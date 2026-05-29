// pybind11 bindings for the raw CUDA kernels.
//
// One module exposing four functions:
//   matmul_tiled(A, B)              -> C
//   row_sum(x)                       -> y
//   softmax_online(x)                -> y
//   decode_attention(q, k, v, scale) -> out
//
// Each forward declaration must match the definition in the corresponding
// .cu file. JIT-compilation through torch.utils.cpp_extension.load passes
// every source listed in `sources=[...]` to a single compile/link step, so
// these signatures live in C++ headers' namespace flash_attn_lab.

#include <torch/extension.h>

namespace flash_attn_lab {

torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B);
torch::Tensor row_sum(torch::Tensor x);
torch::Tensor softmax_online(torch::Tensor x);
torch::Tensor decode_attention(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    double sm_scale);

}  // namespace flash_attn_lab

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "flash-attn-lab raw CUDA kernels";
  m.def("matmul_tiled", &flash_attn_lab::matmul_tiled,
        "Tiled FP32 GEMM with 32x32 shared-memory tiles");
  m.def("row_sum", &flash_attn_lab::row_sum,
        "Row-wise sum reduction with warp-shuffle reductions");
  m.def("softmax_online", &flash_attn_lab::softmax_online,
        "Numerically-stable row-wise online softmax");
  m.def("decode_attention", &flash_attn_lab::decode_attention,
        "Naive fused single-query attention against a KV cache");
}
