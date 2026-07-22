// Host-side bindings for indexed observation-field (compact-teacher) CUDA queries.
//
// The Python wrapper (rtgs.core.observation2d_cuda) uploads a CPU-built CSR index and the
// field's derived component tensors verbatim, so this layer only validates devices, dtypes,
// and shapes before launching. Query semantics live in observation2d_ext.cu and must mirror
// rtgs.core.observation2d exactly.

#include <torch/extension.h>

#include <vector>

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
    bool want_color);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INT(x) TORCH_CHECK(x.scalar_type() == at::kInt, #x " must be int32")
#define CHECK_LONG(x) TORCH_CHECK(x.scalar_type() == at::kLong, #x " must be int64")

std::vector<torch::Tensor> query(
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
  for (const auto& t : {xy, query_means, conics, amplitudes, colors, support_centers,
                        support_radii, rot_cs, color_scales, color_grads}) {
    CHECK_CUDA(t);
    CHECK_CONTIGUOUS(t);
    CHECK_FLOAT(t);
  }
  CHECK_CUDA(tile_keys);
  CHECK_CUDA(tile_offsets);
  CHECK_CUDA(component_ids);
  CHECK_CONTIGUOUS(tile_keys);
  CHECK_CONTIGUOUS(tile_offsets);
  CHECK_CONTIGUOUS(component_ids);
  CHECK_LONG(tile_keys);
  CHECK_LONG(tile_offsets);
  CHECK_INT(component_ids);
  TORCH_CHECK(xy.dim() == 2 && xy.size(1) == 2, "xy must be (S, 2)");
  const int64_t n = query_means.size(0);
  TORCH_CHECK(query_means.dim() == 2 && query_means.size(1) == 2, "query_means must be (N, 2)");
  TORCH_CHECK(conics.sizes() == torch::IntArrayRef({n, 3}), "conics must be (N, 3)");
  TORCH_CHECK(amplitudes.sizes() == torch::IntArrayRef({n}), "amplitudes must be (N,)");
  TORCH_CHECK(colors.sizes() == torch::IntArrayRef({n, 3}), "colors must be (N, 3)");
  TORCH_CHECK(
      support_centers.sizes() == torch::IntArrayRef({n, 2}), "support_centers must be (N, 2)");
  TORCH_CHECK(
      support_radii.sizes() == torch::IntArrayRef({n, 2}), "support_radii must be (N, 2)");
  TORCH_CHECK(rot_cs.sizes() == torch::IntArrayRef({n, 2}), "rot_cs must be (N, 2)");
  TORCH_CHECK(
      color_scales.sizes() == torch::IntArrayRef({n, 2}), "color_scales must be (N, 2)");
  TORCH_CHECK(
      color_grads.numel() == 0 || color_grads.sizes() == torch::IntArrayRef({n, 6}),
      "color_grads must be empty or (N, 6)");
  TORCH_CHECK(
      tile_offsets.size(0) == tile_keys.size(0) + 1,
      "tile_offsets length must be tile_keys length + 1");
  TORCH_CHECK(tiles_x > 0 && tile_size > 0, "tiles_x and tile_size must be positive");
  return rtgs_observation_query_cuda(
      xy, tile_keys, tile_offsets, component_ids, query_means, conics, amplitudes, colors,
      support_centers, support_radii, rot_cs, color_scales, color_grads, tiles_x, tile_size,
      off_x, off_y, fit_x0, fit_y0, fit_x1, fit_y1, fade_floor, normalize, eps, want_color);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("query", &query, "rtgs indexed observation-field query (CUDA)");
}
