/*
 * softmax_online.cu
 *
 * What it computes:
 *   Row-wise numerically-stable softmax in two streamed passes per row:
 *     pass 1: m, l = streaming max + sum-of-exp (the "online" recurrence)
 *     pass 2: y[i] = exp(x[i] - m) / l
 *   Input shape: (B, N), fp32. Output shape: (B, N), fp32.
 *
 * Parallelization strategy:
 *   - One block per row. BLOCK = 256 threads (8 warps).
 *   - Each thread computes a per-thread (m_t, l_t) over its strided slice of
 *     the row. The block then reduces (m_t, l_t) using the online softmax
 *     merge:
 *
 *         m_new = max(m_a, m_b)
 *         l_new = exp(m_a - m_new)*l_a + exp(m_b - m_new)*l_b
 *
 *     This merge is associative, so warp shuffles + a small shared buffer
 *     fold all 8 warps' partials into a single (m, l) pair.
 *   - In pass 2 each thread writes its slice using the shared (m, l).
 *
 * Memory access pattern:
 *   - Strided reads of x with stride blockDim.x are fully coalesced.
 *   - Two passes over the row means 2*N reads + N writes per row in HBM;
 *     this is the canonical "non-fused" baseline we then beat with the
 *     fused FlashAttention kernels.
 *
 * Interview Q this answers:
 *   "Why is naive softmax numerically unstable, and how does the online
 *    algorithm fix it?"
 *
 *   Naive softmax computes exp(x_i) directly. For x_i large (logits often
 *   reach ~1e2 after attention scaling on long sequences), exp overflows
 *   fp32/fp16. The standard fix is to subtract the row max: y_i =
 *   exp(x_i - m) / sum_j exp(x_j - m). This yields the same exact result
 *   and bounds every exp argument to <= 0. The "online" formulation lets
 *   you compute (m, l) in a single streaming pass: when a new value x
 *   arrives, m_new = max(m, x) and l_new = l*exp(m - m_new) + exp(x - m_new).
 *   This merge is associative, which is exactly what allows tiled
 *   FlashAttention to combine partial softmax statistics across KV blocks
 *   without ever materializing the full N-by-N attention matrix.
 *
 * Status: WORKING, used by tests.
 */

#include <cfloat>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace flash_attn_lab {

constexpr int BLOCK = 256;
constexpr int WARP = 32;
constexpr int MAX_WARPS = BLOCK / WARP;

struct ML {
  float m;
  float l;
};

__device__ __forceinline__ ML merge_ml(ML a, ML b) {
  // Online softmax merge (associative).
  if (a.m == -FLT_MAX) return b;
  if (b.m == -FLT_MAX) return a;
  float m = fmaxf(a.m, b.m);
  float l = a.l * __expf(a.m - m) + b.l * __expf(b.m - m);
  return {m, l};
}

__device__ __forceinline__ ML warp_reduce_ml(ML v) {
  #pragma unroll
  for (int offset = WARP / 2; offset > 0; offset >>= 1) {
    float om = __shfl_down_sync(0xffffffff, v.m, offset);
    float ol = __shfl_down_sync(0xffffffff, v.l, offset);
    v = merge_ml(v, ML{om, ol});
  }
  return v;
}

__global__ void softmax_online_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int B, int N) {
  const int b = blockIdx.x;
  if (b >= B) return;

  const float* xrow = x + b * N;
  float* yrow = y + b * N;

  // Pass 1: streaming (m, l).
  ML acc{-FLT_MAX, 0.0f};
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float xi = xrow[i];
    float m_new = fmaxf(acc.m, xi);
    float l_new = acc.l * __expf(acc.m - m_new) + __expf(xi - m_new);
    acc.m = m_new;
    acc.l = l_new;
  }

  acc = warp_reduce_ml(acc);

  __shared__ ML warp_ml[MAX_WARPS];
  const int lane = threadIdx.x & (WARP - 1);
  const int warp_id = threadIdx.x >> 5;
  if (lane == 0) {
    warp_ml[warp_id] = acc;
  }
  __syncthreads();

  __shared__ ML row_ml;
  if (warp_id == 0) {
    int n_warps = (blockDim.x + WARP - 1) / WARP;
    ML v = (lane < n_warps) ? warp_ml[lane] : ML{-FLT_MAX, 0.0f};
    v = warp_reduce_ml(v);
    if (lane == 0) {
      row_ml = v;
    }
  }
  __syncthreads();

  // Pass 2: normalize.
  const float m = row_ml.m;
  const float inv_l = 1.0f / row_ml.l;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    yrow[i] = __expf(xrow[i] - m) * inv_l;
  }
}

torch::Tensor softmax_online(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "softmax_online is fp32-only in this scaffold");
  TORCH_CHECK(x.dim() == 2, "x must be 2D (B, N)");
  x = x.contiguous();

  const int B = x.size(0);
  const int N = x.size(1);

  auto y = torch::empty_like(x);
  dim3 block(BLOCK);
  dim3 grid(B);
  softmax_online_kernel<<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), B, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

}  // namespace flash_attn_lab
