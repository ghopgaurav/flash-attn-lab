/*
 * reduce.cu
 *
 * What it computes:
 *   Row-wise sum reduction. Given x of shape (B, N), produces y of shape (B,)
 *   where y[b] = sum_n x[b, n]. One CUDA block per row.
 *
 * Parallelization strategy:
 *   - One block per row b. Block size is BLOCK = 256 threads (= 8 warps).
 *   - Each thread strides through the N dimension with a grid-stride pattern,
 *     accumulating partial sums in a register.
 *   - Per-warp reduction uses __shfl_down_sync, which is the cheapest
 *     intra-warp exchange path (no shared memory traffic). Lane 0 of each
 *     warp writes its warp-sum to a small (warps x 1) shared buffer.
 *   - The first warp then reads those warp-sums and reduces them again with
 *     __shfl_down_sync. Lane 0 writes the final scalar to y[b].
 *
 * Memory access pattern:
 *   - Strided global reads with stride blockDim.x ensure consecutive threads
 *     in a warp address consecutive elements -> fully coalesced.
 *   - Shared-memory traffic is O(num_warps) per block, negligible.
 *
 * Interview Q this answers:
 *   "Explain warp scheduling and divergence."
 *
 *   A warp is 32 threads executed in lockstep on a SIMT pipeline; the
 *   scheduler issues one instruction per warp per cycle (modulo dual-issue).
 *   Divergence happens when threads in the same warp take different
 *   control-flow paths: the hardware serializes the divergent paths,
 *   masking off lanes, which is wasted issue slots. Warp shuffles
 *   (__shfl_down_sync etc.) move data between lanes inside one warp without
 *   touching shared memory, and they require all lanes named in the mask to
 *   participate, which is the lockstep contract made explicit. A
 *   well-written reduction keeps the active mask = full warp (0xffffffff)
 *   for as long as possible and only narrows when fewer lanes have valid
 *   data, to avoid divergence stalls.
 *
 * Status: WORKING, used by tests.
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

namespace flash_attn_lab {

constexpr int BLOCK = 256;
constexpr int WARP = 32;
constexpr int MAX_WARPS = BLOCK / WARP;

__device__ __forceinline__ float warp_reduce_sum(float v) {
  // Full-mask warp shuffle reduction. All 32 lanes must participate.
  #pragma unroll
  for (int offset = WARP / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__global__ void row_sum_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int B, int N) {
  const int b = blockIdx.x;
  if (b >= B) return;

  const float* xrow = x + b * N;

  float acc = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    acc += xrow[i];
  }

  acc = warp_reduce_sum(acc);

  __shared__ float warp_sums[MAX_WARPS];
  const int lane = threadIdx.x & (WARP - 1);
  const int warp_id = threadIdx.x >> 5;

  if (lane == 0) {
    warp_sums[warp_id] = acc;
  }
  __syncthreads();

  // First warp reduces the per-warp sums.
  if (warp_id == 0) {
    float v = (lane < (blockDim.x + WARP - 1) / WARP) ? warp_sums[lane] : 0.0f;
    v = warp_reduce_sum(v);
    if (lane == 0) {
      y[b] = v;
    }
  }
}

torch::Tensor row_sum(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "row_sum is fp32-only in this scaffold");
  TORCH_CHECK(x.dim() == 2, "x must be 2D (B, N)");
  x = x.contiguous();

  const int B = x.size(0);
  const int N = x.size(1);

  auto y = torch::empty({B}, x.options());

  dim3 block(BLOCK);
  dim3 grid(B);
  row_sum_kernel<<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), B, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

}  // namespace flash_attn_lab
