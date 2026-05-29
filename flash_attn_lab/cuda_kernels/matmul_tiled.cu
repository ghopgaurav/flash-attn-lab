/*
 * matmul_tiled.cu
 *
 * What it computes:
 *   C = A @ B for FP32 matrices, with shapes
 *     A: (M, K)   row-major
 *     B: (K, N)   row-major
 *     C: (M, N)   row-major
 *
 * Parallelization strategy:
 *   - 2D grid of blocks; each block computes a 32x32 tile of C.
 *   - Within a block, 32x32 = 1024 threads, one output element per thread
 *     (one warp per row of the output tile).
 *   - The K dimension is consumed in chunks of TILE; in each iteration we
 *     cooperatively load the next 32x32 sub-tile of A and B into shared
 *     memory, sync, multiply-accumulate, sync, and advance.
 *
 * Memory access pattern:
 *   - Global -> shared: each thread loads exactly one element of As and one
 *     of Bs per K-tile. The (row, col) layout makes consecutive threads in a
 *     warp address consecutive 4-byte words in global memory, so loads are
 *     fully coalesced for both A (row-major K stride) and B (row-major N
 *     stride).
 *   - Shared -> registers: the inner k-loop reads As[ty][k] and Bs[k][tx];
 *     the As column read is broadcast across the warp, the Bs row read is
 *     contiguous in shared memory and bank-conflict free for fp32 with 32
 *     banks.
 *
 * Interview Q this answers:
 *   "Explain shared vs global memory and memory coalescing."
 *
 *   Global memory lives in HBM/GDDR; latency is ~hundreds of cycles and
 *   bandwidth is the dominant cost when threads in a warp read non-adjacent
 *   addresses. Shared memory lives on-chip per SM; latency is a few cycles
 *   and is partitioned into 32 banks. Coalescing means consecutive threads
 *   in a warp issue accesses that fall in the same 32/64/128-byte segment,
 *   so the hardware turns 32 thread-loads into one transaction. Tiling
 *   stages a working set in shared memory so each input element is reused
 *   `TILE` times across the inner accumulation, turning a memory-bound naive
 *   matmul into a compute-bound tiled matmul.
 *
 * Status: WORKING, used by tests. Not optimal (no register-blocking, no
 * vectorized loads, no double-buffering); the goal is to be obviously
 * correct and serve as a baseline for the Triton/cuBLAS comparison.
 */

#include <cuda_runtime.h>
#include <torch/extension.h>

namespace flash_attn_lab {

constexpr int TILE = 32;

__global__ void matmul_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;

  float acc = 0.0f;

  const int num_k_tiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < num_k_tiles; ++t) {
    const int a_col = t * TILE + threadIdx.x;
    const int b_row = t * TILE + threadIdx.y;

    As[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}

torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A, B must be CUDA tensors");
  TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32,
              "matmul_tiled is fp32-only in this scaffold");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A, B must be 2D");
  TORCH_CHECK(A.size(1) == B.size(0), "K mismatch");
  A = A.contiguous();
  B = B.contiguous();

  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);

  auto C = torch::empty({M, N}, A.options());

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  matmul_tiled_kernel<<<grid, block>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return C;
}

}  // namespace flash_attn_lab
