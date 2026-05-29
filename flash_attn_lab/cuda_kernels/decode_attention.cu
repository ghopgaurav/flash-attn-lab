/*
 * decode_attention.cu
 *
 * What it computes:
 *   Single-query attention against a KV cache, fused into one kernel.
 *     q:       (B, H, D)        single query per (batch, head)
 *     k_cache: (B, H, T, D)
 *     v_cache: (B, H, T, D)
 *     out:     (B, H, D)
 *   Per (batch, head):
 *     scores[t] = (q @ k_cache[t]) * sm_scale
 *     probs     = softmax(scores)
 *     out       = probs @ v_cache
 *
 * Parallelization strategy:
 *   - One CUDA block per (batch, head) pair. blockIdx.y == b, blockIdx.x == h.
 *   - BLOCK = 128 threads.
 *   - Stage 1: every thread computes a strided slice of `scores[t] = sum_d q_d * k[t,d]`,
 *     accumulating in fp32.
 *   - Stage 2: row-wise online softmax over `scores` (reuses the same
 *     m-l recurrence used in softmax_online.cu, but inlined here so the
 *     scores tensor stays in shared memory).
 *   - Stage 3: every thread computes a strided slice of
 *     `out[d] = sum_t probs[t] * v[t,d]`, accumulating in fp32, and writes
 *     to the output.
 *   - This is naive: scores live in shared memory at full size T per block,
 *     and we make two passes over the KV cache (stage 1 + stage 3). It is
 *     intentionally NOT a flash-style streaming decode kernel; that is the
 *     content of phase 3 / triton_kernels/decode.py.
 *
 * Memory access pattern:
 *   - For each block, q is broadcast (D floats) to all threads.
 *   - K and V are streamed twice across threads in a warp; reads are
 *     strided over D first then T, which is coalesced for D-major layout.
 *
 * Interview Q this answers:
 *   "Why is decode memory-bound while prefill is compute-bound?"
 *
 *   Prefill multiplies an M-by-D query block by a D-by-N key block, and the
 *   resulting M-by-N scores are reused inside the softmax + V matmul. The
 *   arithmetic-intensity formula 2*M*N*D / (M*D + N*D + M*N) grows like
 *   min(M, N), so for square prefill tiles you get hundreds of flops per
 *   byte and you saturate the tensor cores. In decode M = 1: the kernel
 *   touches the full T-by-D K and V caches (N reads of D bytes each) and
 *   only does 2*T*D flops. Arithmetic intensity is O(1) per byte read, far
 *   below the roofline ridge point of any modern GPU, so HBM bandwidth is
 *   the binding constraint. This is also why decode kernels obsess over
 *   reducing KV-cache reads (paged KV, GQA, KV quantization).
 *
 * Status: WORKING but naive. Correctness is validated against the
 * reference; performance is intentionally a baseline for later iteration.
 */

#include <cfloat>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace flash_attn_lab {

constexpr int DECODE_BLOCK = 128;
constexpr int DECODE_WARP = 32;
constexpr int DECODE_MAX_WARPS = DECODE_BLOCK / DECODE_WARP;

__device__ __forceinline__ float warp_reduce_sum_f32(float v) {
  #pragma unroll
  for (int offset = DECODE_WARP / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

struct ML {
  float m;
  float l;
};

__device__ __forceinline__ ML merge_ml_f32(ML a, ML b) {
  if (a.m == -FLT_MAX) return b;
  if (b.m == -FLT_MAX) return a;
  float m = fmaxf(a.m, b.m);
  float l = a.l * __expf(a.m - m) + b.l * __expf(b.m - m);
  return {m, l};
}

__device__ __forceinline__ ML warp_reduce_ml_f32(ML v) {
  #pragma unroll
  for (int offset = DECODE_WARP / 2; offset > 0; offset >>= 1) {
    float om = __shfl_down_sync(0xffffffff, v.m, offset);
    float ol = __shfl_down_sync(0xffffffff, v.l, offset);
    v = merge_ml_f32(v, ML{om, ol});
  }
  return v;
}

extern __shared__ float decode_smem[];

__global__ void decode_attention_kernel(
    const float* __restrict__ q,        // (B, H, D)
    const float* __restrict__ k_cache,  // (B, H, T, D)
    const float* __restrict__ v_cache,  // (B, H, T, D)
    float* __restrict__ out,            // (B, H, D)
    int B, int H, int T, int D,
    float sm_scale) {
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  if (b >= B || h >= H) return;

  const int q_off = (b * H + h) * D;
  const int kv_off = ((b * H) + h) * T * D;

  const float* qrow = q + q_off;
  const float* krow = k_cache + kv_off;
  const float* vrow = v_cache + kv_off;
  float* orow = out + q_off;

  float* scores = decode_smem;  // size T

  // Stage 1: scores[t] = (q @ k[t]) * sm_scale
  // Each block iterates t = 0..T-1. To get coalesced reads of k[t, :] we
  // stride threadIdx.x across D and reduce.
  for (int t = 0; t < T; ++t) {
    float acc = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      acc += qrow[d] * krow[t * D + d];
    }
    acc = warp_reduce_sum_f32(acc);

    __shared__ float warp_partial[DECODE_MAX_WARPS];
    int lane = threadIdx.x & (DECODE_WARP - 1);
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_partial[warp_id] = acc;
    __syncthreads();
    if (warp_id == 0) {
      int n_warps = (blockDim.x + DECODE_WARP - 1) / DECODE_WARP;
      float v = (lane < n_warps) ? warp_partial[lane] : 0.0f;
      v = warp_reduce_sum_f32(v);
      if (lane == 0) {
        scores[t] = v * sm_scale;
      }
    }
    __syncthreads();
  }

  // Stage 2: online softmax over `scores` (length T) into `scores`.
  ML acc{-FLT_MAX, 0.0f};
  for (int t = threadIdx.x; t < T; t += blockDim.x) {
    float xi = scores[t];
    float m_new = fmaxf(acc.m, xi);
    float l_new = acc.l * __expf(acc.m - m_new) + __expf(xi - m_new);
    acc.m = m_new;
    acc.l = l_new;
  }
  acc = warp_reduce_ml_f32(acc);

  __shared__ ML warp_ml[DECODE_MAX_WARPS];
  int lane = threadIdx.x & (DECODE_WARP - 1);
  int warp_id = threadIdx.x >> 5;
  if (lane == 0) warp_ml[warp_id] = acc;
  __syncthreads();

  __shared__ ML row_ml;
  if (warp_id == 0) {
    int n_warps = (blockDim.x + DECODE_WARP - 1) / DECODE_WARP;
    ML v = (lane < n_warps) ? warp_ml[lane] : ML{-FLT_MAX, 0.0f};
    v = warp_reduce_ml_f32(v);
    if (lane == 0) row_ml = v;
  }
  __syncthreads();

  const float m = row_ml.m;
  const float inv_l = 1.0f / row_ml.l;
  for (int t = threadIdx.x; t < T; t += blockDim.x) {
    scores[t] = __expf(scores[t] - m) * inv_l;
  }
  __syncthreads();

  // Stage 3: out[d] = sum_t probs[t] * v[t, d]
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc_o = 0.0f;
    for (int t = 0; t < T; ++t) {
      acc_o += scores[t] * vrow[t * D + d];
    }
    orow[d] = acc_o;
  }
}

torch::Tensor decode_attention(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    double sm_scale) {
  TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(),
              "all inputs must be CUDA tensors");
  TORCH_CHECK(q.dtype() == torch::kFloat32 && k_cache.dtype() == torch::kFloat32 &&
                  v_cache.dtype() == torch::kFloat32,
              "decode_attention is fp32-only in this scaffold");
  TORCH_CHECK(q.dim() == 3, "q must be (B, H, D)");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4, "k/v cache must be (B, H, T, D)");
  TORCH_CHECK(q.size(0) == k_cache.size(0) && q.size(1) == k_cache.size(1),
              "B/H mismatch between q and k_cache");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k/v cache shape mismatch");

  q = q.contiguous();
  k_cache = k_cache.contiguous();
  v_cache = v_cache.contiguous();

  const int B = q.size(0);
  const int H = q.size(1);
  const int D = q.size(2);
  const int T = k_cache.size(2);

  auto out = torch::empty({B, H, D}, q.options());

  dim3 block(DECODE_BLOCK);
  dim3 grid(H, B);
  size_t shmem_bytes = T * sizeof(float);

  decode_attention_kernel<<<grid, block, shmem_bytes>>>(
      q.data_ptr<float>(),
      k_cache.data_ptr<float>(),
      v_cache.data_ptr<float>(),
      out.data_ptr<float>(),
      B, H, T, D,
      static_cast<float>(sm_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace flash_attn_lab
