#include <cuda_runtime.h>
#include <thrust/reduce.h>

#include <numeric>
#include <sstream>
#include <vector>

#include "myreduce.h"

#define CUPRINT(x, ...) \
  { printf("\33[33m(CUDA) " x "\n\33[0m", ##__VA_ARGS__); }

#define CUDA_CHECK(err)                                                          \
  do {                                                                           \
    cudaError_t err_ = (err);                                                    \
    if (err_ != cudaSuccess) {                                                   \
      std::stringstream ss;                                                      \
      ss << "CUDA error " << int(err_) << " at " << __FILE__ << ":" << __LINE__; \
      throw std::runtime_error(ss.str());                                        \
    }                                                                            \
  } while (false)

template <typename T, unsigned int kBlockSize>
__device__ void WarpReduce(volatile T* sdata, unsigned int tid) {
  if (kBlockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (kBlockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (kBlockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (kBlockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (kBlockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (kBlockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <typename SrcType, typename DstType, unsigned int kBlockSize>
__global__ void ReduceSum(const SrcType* g_idata, DstType* g_odata, size_t n) {
  extern __shared__ DstType sdata[];
  const unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (kBlockSize * 2) + tid;
  DstType my_sum = i < n ? DstType(g_idata[i]) : DstType();
  if (i + kBlockSize < n) my_sum += DstType(g_idata[i + kBlockSize]);
  sdata[tid] = my_sum;
  __syncthreads();
  if (kBlockSize >= 1024) {
    if (tid < 512) sdata[tid] += sdata[tid + 512];
    __syncthreads();
  }
  if (kBlockSize >= 512) {
    if (tid < 256) sdata[tid] += sdata[tid + 256];
    __syncthreads();
  }
  if (kBlockSize >= 256) {
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
  }
  if (kBlockSize >= 128) {
    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();
  }
  if (tid < 32) WarpReduce<DstType, kBlockSize>(sdata, tid);
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

float MyReduce(const float* data, size_t n) {
  constexpr unsigned int thd_num = 512;
  const unsigned int blk_num = (n + thd_num * 2 - 1) / (thd_num * 2);
  const unsigned int smem_size = sizeof(float) * thd_num;
  const size_t total_size = sizeof(float) * n;
  float *g_idata, *g_odata;
  CUDA_CHECK(cudaMalloc(&g_idata, total_size));
  CUDA_CHECK(cudaMalloc(&g_odata, sizeof(float) * blk_num));
  CUDA_CHECK(cudaMemcpy(g_idata, data, total_size, cudaMemcpyHostToDevice));
  ReduceSum<float, float, thd_num><<<blk_num, thd_num, smem_size>>>(g_idata, g_odata, n);
  std::vector<float> odata;
  odata.resize(blk_num);
  CUDA_CHECK(cudaMemcpy(odata.data(), g_odata, sizeof(float) * blk_num, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(g_idata));
  CUDA_CHECK(cudaFree(g_odata));
  return std::accumulate(odata.begin(), odata.end(), 0.0f);
}

float ThReduce(const float* data, size_t n) {
  const size_t total_size = sizeof(float) * n;
  float* g_idata;
  CUDA_CHECK(cudaMalloc(&g_idata, total_size));
  CUDA_CHECK(cudaMemcpy(g_idata, data, total_size, cudaMemcpyHostToDevice));
  auto ret = thrust::reduce(thrust::device, g_idata, g_idata + n);
  CUDA_CHECK(cudaFree(g_idata));
  return ret;
}