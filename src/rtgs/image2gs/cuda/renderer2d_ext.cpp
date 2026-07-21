// Host-side bindings for the batched stage-1 2D gaussian splatting kernels.
//
// Semantics mirror rtgs.image2gs.renderer2d exactly: additive accumulated blending
// (sum_i weight_i * color_i * exp(-0.5 q_i) for q_i < cutoff), half-pixel centers,
// detached support rectangles computed host-side. Kernel structure follows StructSplat's
// exact CUDA renderer (MIT), adapted to this repository's compositor and batching.

#include <torch/extension.h>

#include <vector>

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
    double cutoff);

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
    double cutoff);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INT(x) TORCH_CHECK(x.scalar_type() == at::kInt, #x " must be int32")

static void check_inputs(
    const torch::Tensor& xy,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& weights,
    const torch::Tensor& rects,
    const torch::Tensor& view_index,
    int64_t n_views,
    int64_t height,
    int64_t width) {
  CHECK_CUDA(xy);
  CHECK_CUDA(conics);
  CHECK_CUDA(colors);
  CHECK_CUDA(weights);
  CHECK_CUDA(rects);
  CHECK_CUDA(view_index);
  CHECK_CONTIGUOUS(xy);
  CHECK_CONTIGUOUS(conics);
  CHECK_CONTIGUOUS(colors);
  CHECK_CONTIGUOUS(weights);
  CHECK_CONTIGUOUS(rects);
  CHECK_CONTIGUOUS(view_index);
  CHECK_FLOAT(xy);
  CHECK_FLOAT(conics);
  CHECK_FLOAT(colors);
  CHECK_FLOAT(weights);
  CHECK_INT(rects);
  CHECK_INT(view_index);
  TORCH_CHECK(xy.dim() == 2 && xy.size(1) == 2, "xy must be (M, 2)");
  TORCH_CHECK(conics.dim() == 2 && conics.size(1) == 3, "conics must be (M, 3)");
  TORCH_CHECK(colors.dim() == 2 && colors.size(1) == 3, "colors must be (M, 3)");
  TORCH_CHECK(weights.dim() == 1, "weights must be (M,)");
  TORCH_CHECK(rects.dim() == 2 && rects.size(1) == 4, "rects must be (M, 4)");
  TORCH_CHECK(view_index.dim() == 1, "view_index must be (M,)");
  const int64_t m = xy.size(0);
  TORCH_CHECK(conics.size(0) == m, "conics M must match xy M");
  TORCH_CHECK(colors.size(0) == m, "colors M must match xy M");
  TORCH_CHECK(weights.size(0) == m, "weights M must match xy M");
  TORCH_CHECK(rects.size(0) == m, "rects M must match xy M");
  TORCH_CHECK(view_index.size(0) == m, "view_index M must match xy M");
  TORCH_CHECK(n_views > 0, "n_views must be positive");
  TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
}

std::vector<torch::Tensor> forward(
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
  check_inputs(xy, conics, colors, weights, rects, view_index, n_views, height, width);
  return rtgs_render2d_forward_cuda(
      xy, conics, colors, weights, rects, view_index, n_views, height, width, cutoff);
}

std::vector<torch::Tensor> backward(
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
  check_inputs(xy, conics, colors, weights, rects, view_index, n_views, height, width);
  CHECK_CUDA(grad_out);
  CHECK_CUDA(grad_den);
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(grad_den);
  CHECK_FLOAT(grad_out);
  CHECK_FLOAT(grad_den);
  TORCH_CHECK(grad_out.numel() == n_views * height * width * 3, "grad_out must be (B, H, W, 3)");
  TORCH_CHECK(grad_den.numel() == n_views * height * width, "grad_den must be (B, H, W)");
  return rtgs_render2d_backward_cuda(
      grad_out, grad_den, xy, conics, colors, weights, rects, view_index, n_views, height,
      width, cutoff);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "rtgs batched 2D splatting forward (CUDA)");
  m.def("backward", &backward, "rtgs batched 2D splatting backward (CUDA)");
}
