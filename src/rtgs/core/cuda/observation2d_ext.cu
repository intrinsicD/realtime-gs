// Indexed observation-field (compact-teacher) point queries on the GPU.
//
// One thread per query point: binary-search the point's tile in the CSR tile_keys, then
// accumulate the exact paired weights/colors of that tile's component row sequentially in
// registers, in ascending component order — the same canonical order the CPU CSR stream
// evaluates. No atomics anywhere, so results are deterministic across runs (unlike the
// stage-1 splatting kernels); only FMA contraction can differ from the CPU reference.
//
// Semantics mirror rtgs.core.observation2d exactly:
//   - displacement dx = (xy - origin_offset) - query_means[c]  (origin_offset = 0 for legacy
//     fields; fit-window origin for mean-residual fields, with query_means = local_means)
//   - q = a*dx^2 + 2*b*dx*dy + c*dy^2 from the field's effective conics
//   - w = exp(-0.5 q), optional support-fade floor subtract + clamp at 0
//   - inclusive AABB support test against support_centers/radii on the ORIGINAL coordinates
//   - points outside the fit window contribute nothing (all-zero row)
//   - w *= amplitude; numerator += w*color, weight_sum += w
//   - color = numerator / (weight_sum + eps) when normalized, else numerator
//   - optional local affine color: color + local_x*grad_x + local_y*grad_y with the
//     rotated, color-scale-normalized local frame (cos/sin and clamped scales precomputed
//     host-side so their values match torch bit-for-bit)

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <vector>

namespace {

constexpr int kThreads = 256;

__global__ void observation_query_kernel(
    const float* __restrict__ xy,
    const int64_t* __restrict__ tile_keys,
    const int64_t* __restrict__ tile_offsets,
    const int* __restrict__ component_ids,
    const float* __restrict__ query_means,
    const float* __restrict__ conics,
    const float* __restrict__ amplitudes,
    const float* __restrict__ colors,
    const float* __restrict__ support_centers,
    const float* __restrict__ support_radii,
    const float* __restrict__ rot_cs,
    const float* __restrict__ color_scales,
    const float* __restrict__ color_grads,
    bool has_grads,
    int n_points,
    int n_tiles,
    long long tiles_x,
    float tile_size,
    float off_x,
    float off_y,
    float fit_x0,
    float fit_y0,
    float fit_x1,
    float fit_y1,
    float fade_floor,
    bool normalize,
    float eps,
    bool want_color,
    float* __restrict__ out_color,
    float* __restrict__ out_numerator,
    float* __restrict__ out_weight_sum) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n_points) {
    return;
  }
  float x = xy[s * 2 + 0];
  float y = xy[s * 2 + 1];
  float num_r = 0.0f;
  float num_g = 0.0f;
  float num_b = 0.0f;
  float den = 0.0f;

  bool valid = x >= fit_x0 && x < fit_x1 && y >= fit_y0 && y < fit_y1;
  if (valid && n_tiles > 0) {
    long long key = static_cast<long long>(floorf(y / tile_size)) * tiles_x +
        static_cast<long long>(floorf(x / tile_size));
    int lo = 0;
    int hi = n_tiles;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (tile_keys[mid] < key) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    if (lo < n_tiles && tile_keys[lo] == key) {
      int64_t begin = tile_offsets[lo];
      int64_t end = tile_offsets[lo + 1];
      float qx = x - off_x;
      float qy = y - off_y;
      for (int64_t e = begin; e < end; ++e) {
        int c = component_ids[e];
        // Inclusive AABB support test on the original coordinates.
        float scx = support_centers[c * 2 + 0];
        float scy = support_centers[c * 2 + 1];
        float srx = support_radii[c * 2 + 0];
        float sry = support_radii[c * 2 + 1];
        if (x < scx - srx || x > scx + srx || y < scy - sry || y > scy + sry) {
          continue;
        }
        float dx = qx - query_means[c * 2 + 0];
        float dy = qy - query_means[c * 2 + 1];
        float a = conics[c * 3 + 0];
        float b = conics[c * 3 + 1];
        float cc = conics[c * 3 + 2];
        float q = a * dx * dx + 2.0f * b * dx * dy + cc * dy * dy;
        float w = expf(-0.5f * q);
        if (fade_floor > 0.0f) {
          w = fmaxf(w - fade_floor, 0.0f);
        }
        w *= amplitudes[c];
        den += w;
        if (want_color) {
          float cr = colors[c * 3 + 0];
          float cg = colors[c * 3 + 1];
          float cb = colors[c * 3 + 2];
          if (has_grads) {
            float cs = rot_cs[c * 2 + 0];
            float sn = rot_cs[c * 2 + 1];
            float lx = (cs * dx + sn * dy) / color_scales[c * 2 + 0];
            float ly = (-sn * dx + cs * dy) / color_scales[c * 2 + 1];
            cr += lx * color_grads[c * 6 + 0] + ly * color_grads[c * 6 + 3];
            cg += lx * color_grads[c * 6 + 1] + ly * color_grads[c * 6 + 4];
            cb += lx * color_grads[c * 6 + 2] + ly * color_grads[c * 6 + 5];
          }
          num_r += w * cr;
          num_g += w * cg;
          num_b += w * cb;
        }
      }
    }
  }

  out_weight_sum[s] = den;
  if (want_color) {
    out_numerator[s * 3 + 0] = num_r;
    out_numerator[s * 3 + 1] = num_g;
    out_numerator[s * 3 + 2] = num_b;
    if (normalize) {
      float denom = den + eps;
      out_color[s * 3 + 0] = num_r / denom;
      out_color[s * 3 + 1] = num_g / denom;
      out_color[s * 3 + 2] = num_b / denom;
    } else {
      out_color[s * 3 + 0] = num_r;
      out_color[s * 3 + 1] = num_g;
      out_color[s * 3 + 2] = num_b;
    }
  }
}

}  // namespace

std::vector<torch::Tensor> rtgs_observation_query_cuda(
    torch::Tensor xy,
    torch::Tensor tile_keys,
    torch::Tensor tile_offsets,
    torch::Tensor component_ids,
    torch::Tensor query_means,
    torch::Tensor conics,
    torch::Tensor amplitudes,
    torch::Tensor colors,
    torch::Tensor support_centers,
    torch::Tensor support_radii,
    torch::Tensor rot_cs,
    torch::Tensor color_scales,
    torch::Tensor color_grads,
    int64_t tiles_x,
    int64_t tile_size,
    double off_x,
    double off_y,
    double fit_x0,
    double fit_y0,
    double fit_x1,
    double fit_y1,
    double fade_floor,
    bool normalize,
    double eps,
    bool want_color) {
  int n_points = static_cast<int>(xy.size(0));
  int n_tiles = static_cast<int>(tile_keys.size(0));
  bool has_grads = color_grads.numel() > 0;
  auto weight_sum = torch::zeros({n_points}, xy.options());
  auto color = want_color ? torch::zeros({n_points, 3}, xy.options())
                          : torch::empty({0, 3}, xy.options());
  auto numerator = want_color ? torch::zeros({n_points, 3}, xy.options())
                              : torch::empty({0, 3}, xy.options());
  if (n_points > 0) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int blocks = (n_points + kThreads - 1) / kThreads;
    observation_query_kernel<<<blocks, kThreads, 0, stream>>>(
        xy.data_ptr<float>(),
        tile_keys.data_ptr<int64_t>(),
        tile_offsets.data_ptr<int64_t>(),
        component_ids.data_ptr<int>(),
        query_means.data_ptr<float>(),
        conics.data_ptr<float>(),
        amplitudes.data_ptr<float>(),
        colors.data_ptr<float>(),
        support_centers.data_ptr<float>(),
        support_radii.data_ptr<float>(),
        rot_cs.data_ptr<float>(),
        color_scales.data_ptr<float>(),
        has_grads ? color_grads.data_ptr<float>() : nullptr,
        has_grads,
        n_points,
        n_tiles,
        static_cast<long long>(tiles_x),
        static_cast<float>(tile_size),
        static_cast<float>(off_x),
        static_cast<float>(off_y),
        static_cast<float>(fit_x0),
        static_cast<float>(fit_y0),
        static_cast<float>(fit_x1),
        static_cast<float>(fit_y1),
        static_cast<float>(fade_floor),
        normalize,
        static_cast<float>(eps),
        want_color,
        want_color ? color.data_ptr<float>() : nullptr,
        want_color ? numerator.data_ptr<float>() : nullptr,
        weight_sum.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {color, numerator, weight_sum};
}
