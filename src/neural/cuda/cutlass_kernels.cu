/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2023 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "cuda_common.h"
#include <cstdint>

#ifdef USE_CUTLASS


#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/device/gemm_batched.h"


// Fused MHA implementation from cutlass example #41
#include "fused_multi_head_attention/kernel_forward.h"



template <bool bias>
bool fusedMHACutlass(void* output, void* q, void* k, void* v, void* skip,
                     int batch_size, int num_heads, int depth,
                     cudaStream_t stream) {
  cutlass::half_t* mha_q = (cutlass::half_t*)q;
  cutlass::half_t* mha_k = (cutlass::half_t*)k;
  cutlass::half_t* mha_v = (cutlass::half_t*)v;

  constexpr int kQueriesPerBlock = 64;
  constexpr int kKeysPerBlock = 64;
  constexpr bool kSingleValueIteration = true;

  using Attention =
      AttentionKernel<cutlass::half_t,      // scalar_t
                      cutlass::arch::Sm80,  // ArchTag
                      true,                 // Memory is aligned
                      kQueriesPerBlock, kKeysPerBlock, kSingleValueIteration,
                      false,  // Supports dropout
                      bias    // Supports bias
                      >;

  typename Attention::Params p;
  {  // set parameters
    p.query_ptr = mha_q;
    p.key_ptr = mha_k;
    p.value_ptr = mha_v;
    p.logsumexp_ptr = nullptr;  // Only needed for bw
    p.output_accum_ptr = nullptr;
    if (Attention::kNeedsOutputAccumulatorBuffer) {
      // throw Exception("Unhandled case in cutlass MHA");
      return false;
    }
    p.output_ptr = (cutlass::half_t*)output;
    p.attn_bias_ptr = (cutlass::half_t*)skip;

    p.scale = 1.0f / sqrt((float)depth);

    p.num_heads = num_heads;
    p.num_batches = batch_size;
    p.head_dim = depth;
    p.head_dim_value = depth;
    p.num_queries = 64;
    p.num_keys = 64;

    // All tensors are in BMHK shapes
    p.q_strideH = depth;
    p.k_strideH = depth;
    p.v_strideH = depth;
    p.q_strideM = depth * num_heads;
    p.k_strideM = depth * num_heads;
    p.v_strideM = depth * num_heads;
    p.q_strideB = p.q_strideM * 64;
    p.k_strideB = p.k_strideM * 64;
    p.v_strideB = p.v_strideM * 64;
    p.o_strideM = p.head_dim_value * p.num_heads;

    p.bias_strideH = 64 * 64;
    p.bias_strideM = 64;
    p.bias_strideB = num_heads * p.bias_strideH;
  }

  constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
  int smem_bytes = sizeof(typename Attention::SharedStorage);
  if (smem_bytes > 0xc000) {
    cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);
  }
  if (!Attention::check_supported(p)) {
    // throw Exception("Unhandled case in cutlass MHA");
    return false;
  }

  kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);

  // ReportCUDAErrors(cudaGetLastError());
  return true;
}

bool fusedMHA(void* output, void* mha_q, void* mha_k, void* mha_v, void* skip,
              int batch_size, int num_heads, int depth, cudaStream_t stream) {
  if (skip == nullptr) 
    return fusedMHACutlass<false>(output, mha_q, mha_k, mha_v, skip, batch_size,
                                  num_heads, depth, stream);
  else
    return fusedMHACutlass<true>(output, mha_q, mha_k, mha_v, skip, batch_size,
                                  num_heads, depth, stream);
}


namespace lczero {
namespace cudnn_backend {

// function to calculate mean
static float mean(float arr[], int n) {
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += arr[i];
  }
  return sum / n;
}


// function to calculate standard deviation
static float stdDev(float arr[], int n) {
  float m = mean(arr, n);  // get the mean
  float var = 0;           // initialize variance
  for (int i = 0; i < n; i++) {
    var += pow(arr[i] - m, 2);  // add the squared difference from mean
  }
  var /= n;          // divide by number of elements
  return sqrt(var);  // return the square root of variance
}

// Helper fuction to do vector loads/stores
template <typename T>
__device__ __forceinline__ void copyAs(void* dst, const void* src) {
  *((T*)(dst)) = *((const T*)(src));
}

// debug code to dump allocation in GPU memory
template <typename T>
void dumpTensor(const T* memory, int elements, const char* message,
                       bool only_summary = false, bool cpu_tensor = false) {
  const bool fp16 = std::is_same<half, T>::value;
  const bool int8 = std::is_same<int8_t, T>::value;
  printf("\n%s\n", message);
  int elementSize = (int)(fp16 ? sizeof(half) : sizeof(float));
  if (int8) elementSize = sizeof(int8_t);
  int bytes = elements * elementSize;
  void* temp = (void*)memory;
  if (!cpu_tensor) {
    temp = malloc(bytes);
    cudaMemcpy(temp, memory, bytes, cudaMemcpyDeviceToHost);
  }
  float maxval = -std::numeric_limits<float>::max();
  float minval = std::numeric_limits<float>::max();
  int nans = 0;
  int nanss[10]{};

  std::vector<float> fpArr(elements);
  for (int i = 0; i < elements; i++) {
    float val;
    if (int8) {
      int8_t* arr = (int8_t*)temp;
      val = (float)arr[i];
    }
    else if (fp16) {
      half* arr = (half*)temp;
      val = (float)arr[i];
    } else {
      float* arr = (float*)temp;
      val = arr[i];
    }
    fpArr[i] = val;
    maxval = std::max(maxval, val);
    minval = std::min(minval, val);

    if (std::isnan(val)) {
      if (nans < 10) nanss[nans] = i;
      nans++;
    }

    if (!only_summary || i < 2 || i == elements - 1) {
      printf("%8.4f ", val);
      if ((i % 8) == 7) printf("\n");
      // printf("%i;%.6f\n", i, val);
    }
  }
  if (!cpu_tensor) free(temp);
  if (maxval == -std::numeric_limits<float>::max())
    maxval = std::numeric_limits<double>::quiet_NaN();
  if (minval == std::numeric_limits<float>::max())
    minval = std::numeric_limits<double>::quiet_NaN();

  float avg = mean(&fpArr[0], elements);
  float stddev = stdDev(&fpArr[0], elements);
  printf("Max: %.6f, Min: %.6f, Mean: %.6f, StdDev: %.6f, NaNs: %i of %i",
         maxval, minval, avg, stddev, nans, elements);
  if (nans > 0) {
    printf("\nNaN indices: ");
    for (int i = 0; i < nans && i < 10; i++) printf("%i ", nanss[i]);
    if (nans > 10) printf("......");
  }
  printf("\n");
}


// int8 GEMM using CUTLASS
void cutlassMatrixMulBTransposed(const int8_t* A, const int8_t* B, int8_t* Out,
                                 int M, int N, int K, int batchSize,
                                 int AStride, int BStride, int OutStride,
                                 float alphaf, float betaf) {
  //dumpTensor<int8_t>(A, 512, "A after scaling", false);
  //dumpTensor<int8_t>(B, 512, "B after scaling", false);

  using ElementAccumulator = int32_t;    // <- data type of accumulator
  using ElementComputeEpilogue = float;  // <- data type of epilogue operations
  using ElementInputA = int8_t;  // <- data type of elements in input matrix A
  using ElementInputB = int8_t;  // <- data type of elements in input matrix B
  using ElementOutput =
      int8_t;  // <- data type of elements in output matrix Out

  // TODO: figure out why row major for matrix B doesn't work?!!!
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 128>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 128>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 32>;

  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

  // This code section describes the epilogue part of the kernel
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, ElementComputeEpilogue>;

  // Number of pipelines you want to use
  constexpr int NumStages = 3;

  using Gemm = cutlass::gemm::device::GemmBatched<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
      ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(alphaf);
  ElementComputeEpilogue beta = ElementComputeEpilogue(betaf);

  typename Gemm::Arguments arguments{
      {M, N, K}, {A, K},   AStride,   {B, K},        BStride,  {Out, N},
      OutStride, {Out, N}, OutStride, {alpha, beta}, batchSize};

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Initialize CUTLASS kernel with arguments and workspace pointer
  cutlass::Status status = gemm_op.initialize(arguments, nullptr);
  status = gemm_op();
}

// FP16 GEMM using cutlass
void cutlassMatrixMulBTransposed(const half* A, const half* B, half* Out, int M,
    int N, int K, int batchSize, int AStride, int BStride, int OutStride, bool useInt8) {

  half halfOne = (half)1.0f;
  half halfZero = (half)0.0f;

  using ElementAccumulator = cutlass::half_t;  // <- data type of accumulator
  using ElementComputeEpilogue =
      ElementAccumulator;  // <- data type of epilogue operations
  using ElementInputA =
      cutlass::half_t;  // <- data type of elements in input matrix A
  using ElementInputB =
      cutlass::half_t;  // <- data type of elements in input matrix B
  using ElementOutput =
      cutlass::half_t;  // <- data type of elements in output matrix D
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm80;

  // This code section describes the tile size a thread block will compute
  using ShapeMMAThreadBlock =
      cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N
                                               // = 128, K = 32
  // This code section describes tile size a warp will compute
  using ShapeMMAWarp =
      cutlass::gemm::GemmShape<32, 64,
                               32>;  // <- warp tile M = 64, N = 64, K = 32
  // This code section describes the size of MMA op
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M =
                                                           // 8, N = 8, K = 4

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;  // <-
                                                                          // ??

  // This code section describes ?
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,  // <- data type of output matrix
      128 / cutlass::sizeof_bits<
                ElementOutput>::value,  // <- this is the number of elements per
                                        // vectorized memory access. For half
                                        // precision, it's 8 elements. This
                                        // becomes the vector width of math
                                        // instructions in epilogue too
      ElementAccumulator,  // <- data type of accumulator
      float>;  // <- data type for alpha/beta in linear combination function

  constexpr int NumStages = 3;  // stages == 2/4 is also good sometimes

  using Gemm = cutlass::gemm::device::GemmBatched<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
      ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;

  cutlass::Status status = gemm_op({{M, N, K},
                                    {(cutlass::half_t const*)A, K},
                                    AStride,
                                    {(cutlass::half_t const*)B, K},
                                    BStride,
                                    {(cutlass::half_t const*)Out, N},
                                    OutStride,
                                    {(cutlass::half_t*)Out, N},
                                    OutStride,
                                    {halfOne, halfZero},
                                    batchSize});
}


static void calibrateGemm(int8_t* weights_int8, float* input_scaling_factors,
                          float* output_scaling_factors, float* cpuA,
                          float* cpuB, int M, int N, int K, int batchSize) {
  std::vector<float> scaling_factors(K);

  // apply smooth-quant (basically adjust A and B matrices to make quantization
  // easier)
  for (int k = 0; k < K; k++) {
    float absMaxA = 0;
    float absMaxB = 0;
    // scan a column of Matrix A to find the abs max.
    for (int y = 0; y < M; y++) {
      float val = cpuA[y * K + k];
      absMaxA = std::max(absMaxA, abs(val));
    }

    // scan a column of Matrix B (from each batch dimension)
    for (int b = 0; b < batchSize; b++)
      for (int x = 0; x < N; x++) {
        float val = cpuB[b * N * K + x * K + k];
        absMaxB = std::max(absMaxB, abs(val));
      }

    // compute scaling factor:
    float s = sqrt(absMaxA / (absMaxB));

    // sanity check, don't use too small, or too big scaling factors
    if (s < 1)
      s = 1.0f;  // don't try to squeeze activations for improving range of
                 // weights!
    if (s > 10) s = 10.0f;

    scaling_factors[k] = s;

    // printf("\nMaxA: %f, MaxB: %f, scale: %f ", absMaxA, absMaxB, s);

    // scale A and B matrices using the scaling factor
    for (int y = 0; y < M; y++) {
      float val = cpuA[y * K + k];
      val /= s;
      cpuA[y * K + k] = (half)val;
    }

    for (int b = 0; b < batchSize; b++)
      for (int x = 0; x < N; x++) {
        float val = cpuB[b * N * K + x * K + k];
        val *= s;
        cpuB[b * N * K + x * K + k] = (half)val;
      }
  }

  // figure out scaling factors for A and B matrices
  float absMaxA = 0;
  for (int i = 0; i < M * K; i++) {
    float val = cpuA[i];
    absMaxA = std::max(absMaxA, abs(val));
  }

  float AFactor = 127.0 / absMaxA;

  // update the scaling factors based on global max for Activation matrix
  for (int i = 0; i < K; i++) {
    input_scaling_factors[i] = 127.0f / (scaling_factors[i] * absMaxA);
  }

  std::vector<float> BFactor(batchSize);
  for (int b = 0; b < batchSize; b++) {
    float absMaxB = 0;
    for (int i = 0; i < K * N; i++) {
      float val = cpuB[i + b * K * N];
      absMaxB = std::max(absMaxB, abs(val));
    }

    // quantize the weights
    float scaleB = 127.0f / absMaxB;
    BFactor[b] = scaleB;
    for (int i = 0; i < K * N; i++) {
      float val = cpuB[i + b * K * N];
      // quantize and clamp
      val = (val * scaleB);
      if (val > 127) val = 127;
      if (val < -128) val = -128;
      weights_int8[i + b * K * N] = (int8_t)roundf(val);
    }
  }

  // output scaling factors
  for (int i = 0; i < batchSize; i++)
    output_scaling_factors[i] = 127.0 / (AFactor * BFactor[i]);

  // Ankan - for debug/test
  // printf("\nScaling factors - A: %g, B_Q: %g, B_K: %g, B_V: %g \n",
  //       127.0 / absMaxA, BFactor[0], BFactor[1], BFactor[2]);
}

// Same Activation (A) matrix (M x K) is multiplied by batchSize x B matrices /
// weights (K x N transposed) The outputs are:
//  1. quantized weight matrices (weights_int8)
//  2. "per-column" scaling factors (input_scaling_factors) needed to quantize
//  matrix A
//  3. Scaling factors to dequantize the output matrix (just 3 values: factorQ,
//  factorK, factorV)
// M_Batch is the batch size component in "M" dimension
// maxValuesA contains the max values in activation matrix found so far
template <typename DataType>
void calibrateGemmForInt8(int8_t* weights_int8, float* input_scaling_factors,
                          float* output_scaling_factors, float* maxValuesA,
                          const DataType* A, const DataType* B, int M, int N,
                          int K, int batchSize, int M_Batch) {
  auto cpuA = (DataType*)malloc(M_Batch * M * K * sizeof(DataType));
  auto cpuB = (DataType*)malloc(batchSize * K * N * sizeof(DataType));

  ReportCUDAErrors(cudaMemcpy(cpuA, A, M_Batch * M * K * sizeof(DataType),
                              cudaMemcpyDeviceToHost));
  ReportCUDAErrors(cudaMemcpy(cpuB, B, batchSize * K * N * sizeof(DataType),
                              cudaMemcpyDeviceToHost));

  // convert to FP32 (if not already in fp32, and pick one Activation matrix at
  // a time)
  auto fpA = (float*)malloc(M * K * sizeof(float));
  auto fpB = (float*)malloc(batchSize * K * N * sizeof(float));

  for (int i = 0; i < K * N * batchSize; i++) fpB[i] = (float)cpuB[i];

  for (int b = 0; b < M_Batch; b++) {
    for (int i = 0; i < M * K; i++) {
      float val = abs((float)cpuA[b * M * K + i]);
      val = std::max(val, maxValuesA[i]);
      fpA[i] = val;
      maxValuesA[i] = val;  // update the max activation matrix
    }

    // calibrate a single sample
    calibrateGemm(weights_int8, input_scaling_factors, output_scaling_factors,
                  fpA, fpB, M, N, K, batchSize);
  }

  free(fpA);
  free(fpB);
  free(cpuA);
  free(cpuB);
}


// process 8 elements per thread (in x dimension)
__global__ void quantizeMatrix(int8_t* output, const half* input, int height,
                               int width, const float* scale) {
  int x = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  float factor[8];
  half ip[8];
  int8_t op[8];

  copyAs<uint4>(&ip[0], &input[y * width + x]);
  copyAs<uint4>(&factor[0], &scale[x]);
  copyAs<uint4>(&factor[4], &scale[x+4]);

  for (int i = 0; i < 8; i++) {
    float val = roundf((float)ip[i] * factor[i]);
    if (val > 127) val = 127;
    if (val < -128) val = -128;
    op[i] = (int8_t)(val);
  }

  copyAs<uint2>(&output[y * width + x], &op[0]);
}


// The scale is per column
void quantizeActivationMatrix(int8_t* output, const half* input, int height,
                              int width, const float* scale, cudaStream_t stream) {

  dim3 blockDim(16, 16);
  dim3 gridDim(lczero::cudnn_backend::DivUp(width, 16 * 8),
               lczero::cudnn_backend::DivUp(height, 16));
  quantizeMatrix<<<gridDim, blockDim, 0, stream>>>(output, input, height, width,
                                                   scale);
  ReportCUDAErrors(cudaGetLastError());
}


#define MAX_BATCH_DEQUANT 16

struct ScaleParam {
  float scale[MAX_BATCH_DEQUANT];
};

// process 8 elements per thread (in x dimension)
__global__ void deQuantizeMatrix(half* output, const int8_t* input, const half *bias, int height, int width, int stride, ScaleParam s) {
  int x = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z;

  if (x >= width || y >= height) return;

  float factor = s.scale[b];

  int8_t ip[8] = {};
  half op[8] = {};
  half bi[8] = {};

  copyAs<uint2>(&ip[0], &input[b * stride + y * width + x]);
  copyAs<uint4>(&bi[0], &bias[b * width + x]);

  for (int i = 0; i < 8; i++) {
    float val = (float)ip[i];
    val *= factor;
    val += (float)bi[i];
    op[i] = (half) val;
  }

  copyAs<uint4>(&output[b * stride + y * width + x], &op[0]);
}



// the scale (in CPU memory) is per "batch"
// the bias is per column, per batch
void deQuantizeOutputMatrixBiasAdd(half* output, const int8_t* input,
                                   int height, int width, int batchSize,
                                   float* scale, const half* bias,
                                   cudaStream_t stream) {
  dim3 blockDim(16, 16);
  dim3 gridDim(lczero::cudnn_backend::DivUp(width, 16 * 8),
               lczero::cudnn_backend::DivUp(height, 16), batchSize);

  assert(batchSize < MAX_BATCH_DEQUANT);    // otherwise we will need to put them in GPU memory

  int stride = width * height;

  ScaleParam s = {};
  for (int i = 0; i < batchSize; i++) s.scale[i] = scale[i];

  deQuantizeMatrix<<<gridDim, blockDim, 0, stream>>>(output, input, bias, height, width, stride, s);
  ReportCUDAErrors(cudaGetLastError());

}




template void calibrateGemmForInt8<float>(int8_t* weights_int8,
                                          float* input_scaling_factors,
                                          float* output_scaling_factors,
                                          float* maxValuesA, const float* A,
                                          const float* B, int M, int N, int K,
                                          int batchSize, int M_Batch);
template void calibrateGemmForInt8<half>(int8_t* weights_int8,
                                         float* input_scaling_factors,
                                         float* output_scaling_factors,
                                         float* maxValuesA, const half* A,
                                         const half* B, int M, int N, int K,
                                         int batchSize, int M_Batch);

template void dumpTensor<float>(const float* memory, int elements,
                                const char* message, bool only_summary,
                                bool cpu_tensor);

template void dumpTensor<half>(const half* memory, int elements,
                                const char* message, bool only_summary,
                                bool cpu_tensor);

template void dumpTensor<int8_t>(const int8_t* memory, int elements,
                                 const char* message, bool only_summary,
                                 bool cpu_tensor);


};  // namespace cudnn_backend
};  // namespace lczero
#endif
