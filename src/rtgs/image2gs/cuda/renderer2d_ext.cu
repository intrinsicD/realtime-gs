// Batched stage-1 2D gaussian splatting kernels.
//
// One CUDA block per gaussian; 256 threads stride the gaussian's precomputed support
// rectangle. Forward accumulates the additive compositor with atomicAdd (matching the
// torch reference's index_add up to float summation order — not bit-exact across runs).
// Backward recomputes weights per pixel and reduces thread-local analytic gradients with
// one final atomicAdd per thread and component (StructSplat's baseline pattern; a
// block-reduce variant is a later optimization, see docs/ROADMAP.md M4).
//
// Conventions (must match rtgs.image2gs.renderer2d exactly):
//   - pixel centers at (px + 0.5, py + 0.5)
//   - q = a*dx^2 + 2*b*dx*dy + c*dy^2 with conic (a, b, c) = inverse covariance entries
//   - hard support cutoff: contributions with q >= cutoff are exactly zero
//   - out[view] += w * color, den[view] += w, with w = exp(-0.5*q) * weight

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <vector>

namespace {

constexpr int kThreads = 256;

__global__ void accumulate2d_kernel(
    const float* __restrict__ xy,
    const float* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ weights,
    const int* __restrict__ rects,
    const int* __restrict__ view_index,
    int m,
    int height,
    int width,
    float cutoff,
    float* __restrict__ out,
    float* __restrict__ den) {
  int i = blockIdx.x;
  if (i >= m) {
    return;
  }
  int x0 = rects[i * 4 + 0];
  int x1 = rects[i * 4 + 1];
  int y0 = rects[i * 4 + 2];
  int y1 = rects[i * 4 + 3];
  int tx = x1 - x0 + 1;
  int ty = y1 - y0 + 1;
  if (tx <= 0 || ty <= 0) {
    return;
  }
  float mx = xy[i * 2 + 0];
  float my = xy[i * 2 + 1];
  // Rects come from the host and are image-clamped, but a NaN/Inf mean (diverged fit) would
  // still poison q; drop such gaussians like the reference's zero contribution.
  if (!isfinite(mx) || !isfinite(my)) {
    return;
  }
  float a = conics[i * 3 + 0];
  float b = conics[i * 3 + 1];
  float c = conics[i * 3 + 2];
  float wgt = weights[i];
  float cr = colors[i * 3 + 0];
  float cg = colors[i * 3 + 1];
  float cb = colors[i * 3 + 2];
  long long base = static_cast<long long>(view_index[i]) * height * width;

  int total = tx * ty;
  for (int t = threadIdx.x; t < total; t += blockDim.x) {
    int px = x0 + (t % tx);
    int py = y0 + (t / tx);
    float dx = static_cast<float>(px) + 0.5f - mx;
    float dy = static_cast<float>(py) + 0.5f - my;
    float q = a * dx * dx + 2.0f * b * dx * dy + c * dy * dy;
    if (!(q < cutoff)) {
      continue;
    }
    float w = expf(-0.5f * q) * wgt;
    long long flat = base + static_cast<long long>(py) * width + px;
    atomicAdd(&out[flat * 3 + 0], w * cr);
    atomicAdd(&out[flat * 3 + 1], w * cg);
    atomicAdd(&out[flat * 3 + 2], w * cb);
    atomicAdd(&den[flat], w);
  }
}

__global__ void backward2d_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ grad_den,
    const float* __restrict__ xy,
    const float* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ weights,
    const int* __restrict__ rects,
    const int* __restrict__ view_index,
    int m,
    int height,
    int width,
    float cutoff,
    float* __restrict__ grad_xy,
    float* __restrict__ grad_conics,
    float* __restrict__ grad_colors,
    float* __restrict__ grad_weights) {
  int i = blockIdx.x;
  if (i >= m) {
    return;
  }
  int x0 = rects[i * 4 + 0];
  int x1 = rects[i * 4 + 1];
  int y0 = rects[i * 4 + 2];
  int y1 = rects[i * 4 + 3];
  int tx = x1 - x0 + 1;
  int ty = y1 - y0 + 1;
  if (tx <= 0 || ty <= 0) {
    return;
  }
  float mx = xy[i * 2 + 0];
  float my = xy[i * 2 + 1];
  if (!isfinite(mx) || !isfinite(my)) {
    return;
  }
  float a = conics[i * 3 + 0];
  float b = conics[i * 3 + 1];
  float c = conics[i * 3 + 2];
  float wgt = weights[i];
  float cr = colors[i * 3 + 0];
  float cg = colors[i * 3 + 1];
  float cb = colors[i * 3 + 2];
  long long base = static_cast<long long>(view_index[i]) * height * width;

  float gmx = 0.0f;
  float gmy = 0.0f;
  float ga = 0.0f;
  float gb = 0.0f;
  float gc = 0.0f;
  float gcr = 0.0f;
  float gcg = 0.0f;
  float gcb = 0.0f;
  float gw = 0.0f;

  int total = tx * ty;
  for (int t = threadIdx.x; t < total; t += blockDim.x) {
    int px = x0 + (t % tx);
    int py = y0 + (t / tx);
    float dx = static_cast<float>(px) + 0.5f - mx;
    float dy = static_cast<float>(py) + 0.5f - my;
    float q = a * dx * dx + 2.0f * b * dx * dy + c * dy * dy;
    if (!(q < cutoff)) {
      continue;
    }
    float raw = expf(-0.5f * q);
    float w = raw * wgt;
    long long flat = base + static_cast<long long>(py) * width + px;
    float gr = grad_out[flat * 3 + 0];
    float gg = grad_out[flat * 3 + 1];
    float gbch = grad_out[flat * 3 + 2];
    float gd = grad_den[flat];

    // Additive compositor: out += w*color, den += w with w = raw*weight.
    float dw = gr * cr + gg * cg + gbch * cb + gd;
    gcr += gr * w;
    gcg += gg * w;
    gcb += gbch * w;
    gw += raw * dw;
    float dq = -0.5f * w * dw;
    gmx += dq * (-2.0f * a * dx - 2.0f * b * dy);
    gmy += dq * (-2.0f * b * dx - 2.0f * c * dy);
    ga += dq * dx * dx;
    gb += dq * 2.0f * dx * dy;
    gc += dq * dy * dy;
  }

  atomicAdd(&grad_xy[i * 2 + 0], gmx);
  atomicAdd(&grad_xy[i * 2 + 1], gmy);
  atomicAdd(&grad_conics[i * 3 + 0], ga);
  atomicAdd(&grad_conics[i * 3 + 1], gb);
  atomicAdd(&grad_conics[i * 3 + 2], gc);
  atomicAdd(&grad_colors[i * 3 + 0], gcr);
  atomicAdd(&grad_colors[i * 3 + 1], gcg);
  atomicAdd(&grad_colors[i * 3 + 2], gcb);
  atomicAdd(&grad_weights[i], gw);
}

}  // namespace

std::vector<torch::Tensor> rtgs_render2d_forward_cuda(
    torch::Tensor xy,
    torch::Tensor conics,
    torch::Tensor colors,
    torch::Tensor weights,
    torch::Tensor rects,
    torch::Tensor view_index,
    int64_t n_views,
    int64_t height,
    int64_t width,
    double cutoff) {
  int m = static_cast<int>(xy.size(0));
  auto out = torch::zeros({n_views, height, width, 3}, xy.options());
  auto den = torch::zeros({n_views, height, width}, xy.options());
  if (m > 0 && height > 0 && width > 0) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    accumulate2d_kernel<<<m, kThreads, 0, stream>>>(
        xy.data_ptr<float>(),
        conics.data_ptr<float>(),
        colors.data_ptr<float>(),
        weights.data_ptr<float>(),
        rects.data_ptr<int>(),
        view_index.data_ptr<int>(),
        m,
        static_cast<int>(height),
        static_cast<int>(width),
        static_cast<float>(cutoff),
        out.data_ptr<float>(),
        den.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {out, den};
}

std::vector<torch::Tensor> rtgs_render2d_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor grad_den,
    torch::Tensor xy,
    torch::Tensor conics,
    torch::Tensor colors,
    torch::Tensor weights,
    torch::Tensor rects,
    torch::Tensor view_index,
    int64_t n_views,
    int64_t height,
    int64_t width,
    double cutoff) {
  int m = static_cast<int>(xy.size(0));
  auto grad_xy = torch::zeros_like(xy);
  auto grad_conics = torch::zeros_like(conics);
  auto grad_colors = torch::zeros_like(colors);
  auto grad_weights = torch::zeros_like(weights);
  if (m > 0 && height > 0 && width > 0) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    backward2d_kernel<<<m, kThreads, 0, stream>>>(
        grad_out.data_ptr<float>(),
        grad_den.data_ptr<float>(),
        xy.data_ptr<float>(),
        conics.data_ptr<float>(),
        colors.data_ptr<float>(),
        weights.data_ptr<float>(),
        rects.data_ptr<int>(),
        view_index.data_ptr<int>(),
        m,
        static_cast<int>(height),
        static_cast<int>(width),
        static_cast<float>(cutoff),
        grad_xy.data_ptr<float>(),
        grad_conics.data_ptr<float>(),
        grad_colors.data_ptr<float>(),
        grad_weights.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad_xy, grad_conics, grad_colors, grad_weights};
}
