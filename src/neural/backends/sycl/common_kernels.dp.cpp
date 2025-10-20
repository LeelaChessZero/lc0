/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2024 The LCZero Authors
  Copyright (C) 2023 Intel Corporation

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
   
  SPDX-License-Identifier:GNU General Public License v3.0 or later
*/

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cassert>

#include "sycl_common.h"
#include "neural/backends/shared/activation.h"
#include "neural/tables/attention_policy_map.h"
#include "winograd_helper.h"
#include <cmath>

namespace lczero {
namespace sycldnn_backend {
namespace {
constexpr int kInputPlanes = 112;
}  // namespace

/////////////////////////////////////////////////////////////////////////////
//          Simple CUDA kernels used by certain layers                     //
/////////////////////////////////////////////////////////////////////////////

template <typename T>
void addVectors_kernel(T* c, T* a, T* b, int size, int asize,
                                  int bsize, ActivationFunction activation,
                                  const sycl::nd_item<3> &item_ct1) {
  int i = item_ct1.get_local_id(2) +
          item_ct1.get_local_range(2) * item_ct1.get_group(2);
  if (i < size) {
    float aVal = 0;
    float bVal = 0;
    if (a) aVal = (float)(a[i % asize]);
    if (b) bVal = (float)(b[i % bsize]);

    float cVal = aVal + bVal;

    cVal = activate(cVal, activation);

    c[i] = (T)cVal;
  }
}

// Adds two vectors (possibly of different sizes), also do optional relu
// activation.
template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize,
                ActivationFunction activation, sycl::queue &sycl_queue) {
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  sycl_queue.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
          sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item_ct1) {
        addVectors_kernel(c, a, b, size, asize, bsize, activation, item_ct1);
      });
}

template <typename T>
void addVectorsHNC_NHC_kernel(T* a, T* b, int N, int H, int C,
                              const sycl::nd_item<3> &item_ct1) {
  int i = item_ct1.get_local_id(2) +
          item_ct1.get_local_range(2) * item_ct1.get_group(2);
  if (i < N * H * C) {
    int orig_i = i;
    int c = i % C;
    i /= C;
    int n = i % N;
    i /= N;
    int h = i;
    float aVal = (float)a[orig_i];
    float bVal = (float)b[n * H * C + h * C + c];

    float cVal = aVal + bVal;

    a[orig_i] = (T)cVal;
  }
}

template <typename T>
void addVectorsHNC_NHC(T* a, T* b, int N, int H, int C,
                       sycl::queue &sycl_queue) {
  const int kBlockSize = 256;
  int blocks = DivUp(N * H * C, kBlockSize);
  sycl_queue.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                             sycl::range<3>(1, 1, kBlockSize),
                                         sycl::range<3>(1, 1, kBlockSize)),
                       [=](sycl::nd_item<3> item_ct1) {
                         addVectorsHNC_NHC_kernel(a, b, N, H, C, item_ct1);
                       });
}

template <typename T, ActivationFunction act>
void addBiasBatched_kernel(T* output, const T* input, const T* bias,
                                      int N, int C,
                                      const sycl::nd_item<3> &item_ct1) {
  int batch = item_ct1.get_group(1);
  int n = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
          item_ct1.get_local_id(1);
  if (n >= N) return;
  int c = item_ct1.get_local_id(2) * 4;

  int biasIndex = batch * C + c;
  int tensorIndex = batch * N * C + n * C + c;

  float val[4];
  float b[4];

  // Load from memory
  const bool fp16 = std::is_same<sycl::half, T>::value;
  if (fp16) {
    sycl::half inp[4];
    copyAs<sycl::uint2>(&inp[0], &input[tensorIndex]);
#pragma unroll
    for (int i = 0; i < 4; i++) val[i] = (float)inp[i];

    copyAs<sycl::uint2>(&inp[0], &bias[biasIndex]);
#pragma unroll
    for (int i = 0; i < 4; i++) b[i] = (float)inp[i];
  } else {
    copyAs<sycl::uint4>(&val[0], &input[tensorIndex]);
    copyAs<sycl::uint4>(&b[0], &bias[biasIndex]);
  }

  // Perform bias add and activation
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float x = val[i] + b[i];
    x = activate(x, act);
    val[i] = x;
  }

  // write to memory
  if (fp16) {
    sycl::half op[4];
#pragma unroll
    for (int i = 0; i < 4; i++) op[i] = (sycl::half)val[i];
    copyAs<sycl::uint2>(&output[tensorIndex], &op[0]);
  } else {
    copyAs<sycl::uint4>(&output[tensorIndex], &val[0]);
  }
}

// Input/output tensors are Batch * N * C
// bias tensor is N * C (i.e, different bias for each Batch dimension)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, ActivationFunction activation, sycl::queue &sycl_queue) {
  // process 4 elements per thread to achieve close to peak memory bandwidth
  if (C % 4 != 0) throw Exception("unsupported filter size");
  if (C > 2048) throw Exception("unsupported filter size");

  sycl::range<3> blockDim(1, 1, 1), gridDim(1, 1, 1);
  blockDim[2] = C / 4;
  unsigned int tmp = (512 / blockDim[2]);
  blockDim[1] = sycl::min(sycl::max(tmp, 1u), (unsigned int)N);
  blockDim[0] = 1;
  gridDim[2] = DivUp(N, blockDim[1]);
  gridDim[1] = Batch;
  gridDim[0] = 1;

  switch (activation) {
    case ACTIVATION_NONE:
      //addBiasBatched_kernel<T, ACTIVATION_NONE>
        //  <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C);
        sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                           [=](sycl::nd_item<3> item_ct1) {
                             addBiasBatched_kernel<T, ACTIVATION_NONE>(output, input, bias,
                                                            N, C, item_ct1);
                           });
      break;
    case ACTIVATION_SELU:
      //addBiasBatched_kernel<T, ACTIVATION_SELU>
        //  <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C);

        sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                           [=](sycl::nd_item<3> item_ct1) {
                             addBiasBatched_kernel<T, ACTIVATION_SELU>(output, input, bias,
                                                            N, C, item_ct1);
                           });

      break;
    case ACTIVATION_MISH:
      //addBiasBatched_kernel<T, ACTIVATION_MISH>
        //  <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C);

      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                           [=](sycl::nd_item<3> item_ct1) {
                             addBiasBatched_kernel<T, ACTIVATION_MISH>(output, input, bias,
                                                            N, C, item_ct1);
                           });
      break;
    case ACTIVATION_RELU:
      //addBiasBatched_kernel<T, ACTIVATION_RELU>
        //  <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C);
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                           [=](sycl::nd_item<3> item_ct1) {
                             addBiasBatched_kernel<T, ACTIVATION_RELU>(output, input, bias,
                                                            N, C, item_ct1);
                           });
      break;
    case ACTIVATION_SWISH:
      //addBiasBatched_kernel<T, ACTIVATION_SWISH>
        //  <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C);
      
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                           [=](sycl::nd_item<3> item_ct1) {
                             addBiasBatched_kernel<T, ACTIVATION_SWISH>(output, input, bias,
                                                            N, C, item_ct1);
                           });
      break;
    case ACTIVATION_RELU_2:  // square relu
      //addBiasBatched_kernel<T, ACTIVATION_RELU_2>
        //  <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C);
      
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                           [=](sycl::nd_item<3> item_ct1) {
                             addBiasBatched_kernel<T, ACTIVATION_RELU_2>(output, input, bias,
                                                            N, C, item_ct1);
                           });

      break;
    default:
      throw Exception(
          "unsupported activation in addBiasBatched. Add in switch-case here");
  }
}

template <typename T, ActivationFunction act>
void addBiasBatched_kernel(T* output, const T* input, const T* bias,
                                      int N, int C, int Nstride,
                                      const sycl::nd_item<3> &item_ct1) {
  int batch = item_ct1.get_group(1);
  int n = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
          item_ct1.get_local_id(1);
  if (n >= N) return;
  int c = item_ct1.get_local_id(2) * 4;

  int biasIndex = batch * C + c;
  int tensorIndex = batch * Nstride * C + n * C + c;

  float val[4];
  float b[4];

  // Load from memory
  const bool fp16 = std::is_same<sycl::half, T>::value;
  if (fp16) {
    sycl::half inp[4];
    copyAs<sycl::uint2>(&inp[0], &input[tensorIndex]);
#pragma unroll
    for (int i = 0; i < 4; i++) val[i] = (float)inp[i];

    copyAs<sycl::uint2>(&inp[0], &bias[biasIndex]);
#pragma unroll
    for (int i = 0; i < 4; i++) b[i] = (float)inp[i];
  } else {
    copyAs<sycl::uint4>(&val[0], &input[tensorIndex]);
    copyAs<sycl::uint4>(&b[0], &bias[biasIndex]);
  }

  // Perform bias add and activation
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float x = val[i] + b[i];
    x = activate(x, act);
    val[i] = x;
  }

  // write to memory
  if (fp16) {
    sycl::half op[4];
#pragma unroll
    for (int i = 0; i < 4; i++) op[i] = (sycl::half)val[i];
    copyAs<sycl::uint2>(&output[tensorIndex], &op[0]);
  } else {
    copyAs<sycl::uint4>(&output[tensorIndex], &val[0]);
  }
}

// Input/output tensors are Batch * N * C
// bias tensor is N * C (i.e, different bias for each Batch dimension)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, int Nstride, ActivationFunction activation, sycl::queue &sycl_queue) {
  // process 4 elements per thread to achieve close to peak memory bandwidth
  if (C % 4 != 0) throw Exception("unsupported filter size");
  if (C > 4096) throw Exception("unsupported filter size");

  sycl::range<3> blockDim(1, 1, 1), gridDim(1, 1, 1);
  blockDim[2] = C / 4;
  unsigned int tmp = (512 / blockDim[2]);
  blockDim[1] = sycl::min(sycl::max(tmp, 1u), (unsigned int)N);
  blockDim[0] = 1;
  gridDim[2] = DivUp(N, blockDim[1]);
  gridDim[1] = Batch;
  gridDim[0] = 1;

  switch (activation) {
    case ACTIVATION_NONE:
      //addBiasBatched_kernel<T, ACTIVATION_NONE>
      //    <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C,
       //                                      Nstride);
       sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                           [=](sycl::nd_item<3> item_ct1) {
                             addBiasBatched_kernel<T, ACTIVATION_NONE>(output, input, bias,
                                                            N, C, Nstride, item_ct1);
                           });
      break;
    case ACTIVATION_SELU:
     // addBiasBatched_kernel<T, ACTIVATION_SELU>
       //   <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C,
         //                                    Nstride);
          sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                           [=](sycl::nd_item<3> item_ct1) {
                             addBiasBatched_kernel<T, ACTIVATION_SELU>(output, input, bias,
                                                            N, C, Nstride, item_ct1);
                            });
      break;
    case ACTIVATION_MISH:
      //addBiasBatched_kernel<T, ACTIVATION_MISH>
      //    <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C,
       //                                      Nstride);
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                           [=](sycl::nd_item<3> item_ct1) {
                             addBiasBatched_kernel<T, ACTIVATION_MISH>(output, input, bias,
                                                            N, C, Nstride, item_ct1);
                           });

      break;
    case ACTIVATION_RELU:
      //addBiasBatched_kernel<T, ACTIVATION_RELU>
        //  <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C,
          //                                   Nstride);

      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                           [=](sycl::nd_item<3> item_ct1) {
                             addBiasBatched_kernel<T, ACTIVATION_RELU>(output, input, bias,
                                                            N, C, Nstride, item_ct1);
                            });
      break;
    case ACTIVATION_SWISH:
      //addBiasBatched_kernel<T, ACTIVATION_SWISH>
      //    <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C,
      //                                       Nstride);

       sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                           [=](sycl::nd_item<3> item_ct1) {
                             addBiasBatched_kernel<T, ACTIVATION_SWISH>(output, input, bias,
                                                            N, C, Nstride, item_ct1);
                          });
      break;
    case ACTIVATION_RELU_2:  // square relu
     // addBiasBatched_kernel<T, ACTIVATION_RELU_2>
     //     <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C,
      //                                       Nstride);
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                           [=](sycl::nd_item<3> item_ct1) {
                             addBiasBatched_kernel<T, ACTIVATION_RELU_2>(output, input, bias,
                                                            N, C, Nstride, item_ct1);
                            });

      break;
    default:
      throw Exception(
          "unsupported activation in addBiasBatched. Add in switch-case here");
  }
}

template <typename T>
void addBias_NCHW_kernel(T* c, T* a, T* b, int N, int C, int H,
                                    int W, ActivationFunction activation,
                                    const sycl::nd_item<3> &item_ct1) {
  int i = item_ct1.get_local_id(2) +
          item_ct1.get_local_range(2) * item_ct1.get_group(2);
  int size = N * C * H * W;

  if (i < size) {
    float aVal = (float)a[i];

    // All this math can be optimized, but the kernel is memory bound anyway.
    int biasIndex = (i / (H * W)) % C;
    float bVal = (float)b[biasIndex];
    

    float cVal = aVal + bVal;

    cVal = activate(cVal, activation);

    c[i] = (T)cVal;
  }
}

// Add bias to convolution's output.
template <typename T>
void addBias_NCHW(T* c, T* a, T* b, int N, int C, int H, int W,
                  ActivationFunction activation, sycl::queue &sycl_queue) {
  int size = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  sycl_queue.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
          sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item_ct1) {
        addBias_NCHW_kernel(c, a, b, N, C, H, W, activation, item_ct1);
      });
}

template <typename dT, typename sT>
dT readNCHW(const sT* input_tensor, int n, int c, int h, int w,
                       int Nin, int Cin, int H, int W) {
  if (n >= Nin || c >= Cin) return 0;

  int index;
  index = n;
  index *= Cin;
  index += c;
  index *= H;
  index += h;
  index *= W;
  index += w;

  return (dT)(input_tensor[index]);
}

template <typename dT, typename sT>
void NCHWtoNHWC_kernel(dT* output_tensor, const sT* input_tensor,
                                  int Nin, int Cin, int Nout, int Cout, int H,
                                  int W, const sycl::nd_item<3> &item_ct1) {
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  if (tid >= Nout * Cout * H * W) return;

  int index = tid;

  int c = (index % Cout);
  index /= Cout;
  int w = index % W;
  index /= W;
  int h = index % H;
  index /= H;
  int n = index;

  output_tensor[tid] =
      readNCHW<dT, sT>(input_tensor, n, c, h, w, Nin, Cin, H, W);
}

template <typename DstType, typename SrcType>
void convertNCHWtoNHWC(DstType* output_tensor, const SrcType* input_tensor,
                       int Nin, int Cin, int Nout, int Cout, int H, int W, sycl::queue &sycl_queue) {
  size_t numElements = Nout * Cout * H * W;
  const int blockSize = 256;
  int blocks = DivUp(numElements, blockSize);
  sycl_queue.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, blockSize),
          sycl::range<3>(1, 1, blockSize)),
      [=](sycl::nd_item<3> item_ct1) {
        NCHWtoNHWC_kernel(output_tensor, input_tensor, Nin, Cin, Nout, Cout, H,
                          W, item_ct1);
      });
}

template <typename DstType, typename SrcType>
void copyTypeConverted_kernel(DstType* op, SrcType* ip, int N,
                              const sycl::nd_item<3> &item_ct1) {
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  if (tid >= N) return;

  DstType el = (DstType)ip[tid];
  op[tid] = el;
}

template <typename DstType, typename SrcType>
void copyTypeConverted(DstType* op, SrcType* ip, int N, sycl::queue &sycl_queue) {
  const int kBlockSize = 256;
  int blocks = DivUp(N, kBlockSize);
  sycl_queue.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                             sycl::range<3>(1, 1, kBlockSize),
                                         sycl::range<3>(1, 1, kBlockSize)),
                       [=](sycl::nd_item<3> item_ct1) {
                         copyTypeConverted_kernel(op, ip, N, item_ct1);
                       });
}

template <typename T>
void batchNorm_kernel(T* output, const T* input, const T* skipInput,
                                 int N, int C, int H, int W, const float* means,
                                 const float* varMultipliers,
                                 ActivationFunction activation,
                                 const sycl::nd_item<3> &item_ct1) {
  int index = item_ct1.get_local_id(2) +
              item_ct1.get_local_range(2) * item_ct1.get_group(2);

  int wIndex = 0;
  if (sizeof(T) == sizeof(float))
    wIndex = (index / (H * W)) % C;  // NCHW for fp32.
  else
    wIndex = index % C;  // NHWC for fp16.

  float el = input[index];
  float mean = means[wIndex];
  float varMulti = varMultipliers[wIndex];

  el -= mean;
  el *= varMulti;

  if (skipInput) el += (float)skipInput[index];

  el = activate(el, activation);

  output[index] = (T)el;
}

// Every thread processes single element.
template <typename T>
void batchNorm(T* output, const T* input, const T* skipInput, int N, int C,
               int H, int W, float* means, float* var_multipliers,
               ActivationFunction activation, sycl::queue &sycl_queue) {
  const int total_elements = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(total_elements, kBlockSize);

  sycl_queue.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
          sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item_ct1) {
        batchNorm_kernel(output, input, skipInput, N, C, H, W, means,
                         var_multipliers, activation, item_ct1);
      });
}

void expandPlanes_kernel_Fp32_NCHW(float* output,
                                              const uint64_t* masks,
                                              const float* values, int n,
                                              const sycl::nd_item<3> &item_ct1,
                                              uint64_t *shMasks, float *shVals) {
  // Block size of 256, same mask/val for 64 consecutive threads.
  constexpr int kNumShmemElements = 256 / 64;

  int index = item_ct1.get_local_id(2) +
              item_ct1.get_local_range(2) * item_ct1.get_group(2);

  int planeIndex = index >> 6;

  if (planeIndex >= n) return;

  // Load inputs to shared memory.
  if (item_ct1.get_local_id(2) < kNumShmemElements) {
    shMasks[item_ct1.get_local_id(2)] =
        masks[planeIndex + item_ct1.get_local_id(2)];
    shVals[item_ct1.get_local_id(2)] =
        values[planeIndex + item_ct1.get_local_id(2)];
  }
  /*
  DPCT1113:53: Consider replacing
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
  sycl::nd_item::barrier() if function "expandPlanes_kernel_Fp32_NCHW" is called
  in a multidimensional kernel.
  */
  item_ct1.barrier(sycl::access::fence_space::local_space);

  uint64_t mask = shMasks[item_ct1.get_local_id(2) >> 6];

  int sqIndex = index & 0x3F;
  float op = 0;

  bool set = !!(mask & (1ull << sqIndex));
  if (set) {
    op = shVals[item_ct1.get_local_id(2) >> 6];
  }
  output[index] = op;
}

void expandPlanes_Fp32_NCHW(float* output, const uint64_t* masks,
                            const float* values, int n, sycl::queue &sycl_queue) {
  int threads = n * 8 * 8;  // Each thread writes a single element.
  const int blockSize = 256;
  int blocks = DivUp(threads, blockSize);
  
  sycl_queue.submit([&](sycl::handler& cgh) {
    /*
    DPCT1101:115: 'kNumShmemElements' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments, if
    it is correct.
    */
    sycl::local_accessor<uint64_t, 1> shMasks_acc_ct1(
        sycl::range<1>(4 /*kNumShmemElements*/), cgh);
    /*
    DPCT1101:116: 'kNumShmemElements' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments, if
    it is correct.
    */
    sycl::local_accessor<float, 1> shVals_acc_ct1(
        sycl::range<1>(4 /*kNumShmemElements*/), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, blockSize),
            sycl::range<3>(1, 1, blockSize)),
        [=](sycl::nd_item<3> item_ct1) {
          expandPlanes_kernel_Fp32_NCHW(output, masks, values, n, item_ct1,
                                        shMasks_acc_ct1.get_pointer(),
                                        shVals_acc_ct1.get_pointer());
        });
  });
}

// TODO: Can optimize using shared memory if this becomes a bottleneck.
void expandPlanes_kernel_Fp16_NHWC(sycl::half* output, const uint64_t* masks,
                                   const float* values, int n,
                                   const sycl::nd_item<3>& item_ct1) {
  const int index = item_ct1.get_local_id(2) +
                    item_ct1.get_local_range(2) * item_ct1.get_group(2);
  if (index >= n * 8 * 8) return;

  const int planeIndex = index % kInputPlanes;
  const int boardIndex = index / (kInputPlanes * 8 * 8);
  const int sqIndex = (index / kInputPlanes) & 0x3F;

  uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];

  sycl::half op = 0;
  bool set = !!(mask & (1ull << sqIndex));
  if (set) {
    float val = values[boardIndex * kInputPlanes + planeIndex];
    op = (sycl::half)val;
  }
  output[index] = op;
}

void expandPlanes_Fp16_NHWC(sycl::half* output, const uint64_t* masks,
                            const float* values, int n, sycl::queue &sycl_queue) {
  int threads = n * 8 * 8;  // Each thread writes a single element.
  const int kBlockSize = 256;
  int blocks = DivUp(threads, kBlockSize);
  {
    
    sycl_queue.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
            sycl::range<3>(1, 1, kBlockSize)),
        [=](sycl::nd_item<3> item_ct1) {
          expandPlanes_kernel_Fp16_NHWC(output, masks, values, n, item_ct1);
        });
  }
}

void expandPlanes_kernel_Fp16_NCHW(sycl::half* output, const uint64_t* masks,
                                   const float* values, int n,
                                   const sycl::nd_item<3>& item_ct1,
                                   uint64_t* shMasks, sycl::half* shVals) {
  // block size of 256, same mask/val for 64 consecutive threads
  constexpr int kNumShmemElements = 256 / 64;

  int index = item_ct1.get_local_id(2) +
              item_ct1.get_local_range(2) * item_ct1.get_group(2);

  int planeIndex = index >> 6;

  if (planeIndex >= n) return;

  // load inputs to shared memory
  if (item_ct1.get_local_id(2) < kNumShmemElements) {
    shMasks[item_ct1.get_local_id(2)] =
        masks[planeIndex + item_ct1.get_local_id(2)];
    shVals[item_ct1.get_local_id(2)] =
        values[planeIndex + item_ct1.get_local_id(2)];
  }
  /*
  DPCT1113:56: Consider replacing
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
  sycl::nd_item::barrier() if function "expandPlanes_kernel_Fp16_NCHW" is called
  in a multidimensional kernel.
  */
  item_ct1.barrier(sycl::access::fence_space::local_space);

  uint64_t mask = shMasks[item_ct1.get_local_id(2) >> 6];

  int sqIndex = index & 0x3F;
  sycl::half op = 0;

  bool set = !!(mask & (1ull << sqIndex));
  if (set) {
    op = (sycl::half)shVals[item_ct1.get_local_id(2) >> 6];
  }
  output[index] = op;
}

void expandPlanes_Fp16_NCHW(sycl::half* output, const uint64_t* masks,
                            const float* values, int n, sycl::queue &sycl_queue) {
  int threads = n * 8 * 8;  // each thread writes a single element
  const int blockSize = 256;
  int blocks = DivUp(threads, blockSize);
  {
    
    sycl_queue.submit([&](sycl::handler& cgh) {
      /*
      DPCT1101:117: 'kNumShmemElements' expression was replaced with a value.
      Modify the code to use the original expression, provided in comments, if
      it is correct.
      */
      sycl::local_accessor<uint64_t, 1> shMasks_acc_ct1(
          sycl::range<1>(4 /*kNumShmemElements*/), cgh);
      /*
      DPCT1101:118: 'kNumShmemElements' expression was replaced with a value.
      Modify the code to use the original expression, provided in comments, if
      it is correct.
      */
      sycl::local_accessor<sycl::half, 1> shVals_acc_ct1(
          sycl::range<1>(4 /*kNumShmemElements*/), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, blockSize),
              sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item_ct1) {
            expandPlanes_kernel_Fp16_NCHW(output, masks, values, n, item_ct1,
                                          shMasks_acc_ct1.get_pointer(),
                                          shVals_acc_ct1.get_pointer());
          });
    });
  }
}

template <typename T>
void globalScale_kernel(T* output, const T* input,
                                   const T* scaleBias, const T* prevLayerBias,
                                   int inputSize, int C,
                                   ActivationFunction activation,
                                   const sycl::nd_item<3> &item_ct1) {
  const int kPlaneSize = 64;

  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  if (tid > inputSize) return;

  int nc = tid / kPlaneSize;
  int n = nc / C;
  int c = nc % C;

  float val1 = input[tid];   // Output of residual block to be scaled.
  float val2 = output[tid];  // Skip connection to be added directly.

  if (prevLayerBias) {
    val1 += (float)(prevLayerBias[c]);
  }

  int startIdx = n * 2 * C;  // Scale and bias interleaved.

  float s = scaleBias[startIdx + c];
  s = 1.0f / (1.0f + sycl::exp(-s));  // Sigmoid on scale.

  float b = scaleBias[startIdx + c + C];

  float op = val1 * s + val2 + b;
  op = activate(op, activation);
  output[tid] = (T)op;
}

void globalScale_kernel_fp16_nhwc(sycl::half* output, const sycl::half* input,
                                  const sycl::half* scaleBias,
                                  const sycl::half* prevLayerBias,
                                  int inputSize, int C, int HWC,
                                  ActivationFunction activation,
                                  const sycl::nd_item<3>& item_ct1) {
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  if (tid > inputSize) return;

  int c = tid % C;
  int n = tid / (HWC);

  float val1 = (float)input[tid];   // Output of residual block to be scaled.
  float val2 = (float)output[tid];  // Skip connection to be added directly.
  if (prevLayerBias) {
    val1 += (float)prevLayerBias[c];
  }

  int startIdx = n * 2 * C;  // Scale and bias interleaved.

  float s = scaleBias[startIdx + c];
  s = 1.0f / (1.0f + sycl::exp(-s));  // Sigmoid on scale.

  float b = scaleBias[startIdx + c + C];

  float op = val1 * s + val2 + b;
  op = activate(op, activation);

  output[tid] = (sycl::half)op;
}

// N blocks.
// C threads per block.
// 'HWC' input data processed by thread block.
// Each thread writes a single output.
void globalAvgPool_kernel_NHWC_fp16(sycl::half* output, const sycl::half* input,
                                    const sycl::half* prevLayerBias,
                                    int inputSize, int outputSize,
                                    const sycl::nd_item<3>& item_ct1) {
  const int elementsPerThread = 64;  // 8x8 board.

  int blockStart = item_ct1.get_group(2) * item_ct1.get_local_range(2);

  float S = 0;

#pragma unroll
  for (int i = 0; i < elementsPerThread; i++) {
    int localIndex = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    int inputIndex = blockStart * elementsPerThread + localIndex;
    if (inputIndex < inputSize) S += (float)(input[inputIndex]);
  }

  float avg = S / elementsPerThread;

  // Add bias from previous layer.
  if (prevLayerBias) avg += (float)(prevLayerBias[item_ct1.get_local_id(2)]);

  int opIndex = blockStart + item_ct1.get_local_id(2);
  if (opIndex < outputSize) output[opIndex] = (sycl::half)avg;
}

// Each thread reads 2 inputs (8x8/32), and each warp writes a single output.
template <typename T>
void globalAvgPool_kernel(T* output, const T* input,
                                     const T* prevLayerBias, int inputSize,
                                     int outputSize, int C,
                                     const sycl::nd_item<3> &item_ct1) {
  const int elementsPerWarp = 64;
  const int elementsPerThread = 2;

  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  int laneId = item_ct1.get_local_id(2) & 0x1F;
  int laneStartIndex = (tid - laneId) * elementsPerThread;

  // Compute per-thread sum for elementsPerThread elements.
  float S = 0;

#pragma unroll
  for (int i = 0; i < elementsPerWarp; i += 32) {
    int index = laneStartIndex + laneId + i;
    if (index < inputSize) S += (float)(input[index]);
  }

// Compute warp wide sum (for entire plane - elementsPerWarp elements).
#pragma unroll
  for (int offset = 1; offset < 32; offset *= 2) {
    /*
    DPCT1023:10: The SYCL sub-group does not support mask options for
    dpct::shift_sub_group_left. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_down_sync.
    */
    S += sycl::shift_group_left(item_ct1.get_sub_group(), S, offset);
  }

  float avg = S / elementsPerWarp;
  int opIndex = tid >> 5;

  // First thread in warp has the sum, write it in output.
  if (laneId == 0) {
    if (opIndex < outputSize) {
      if (prevLayerBias) avg += (float)prevLayerBias[opIndex % C];
      output[opIndex] = (T)avg;
    }
  }
}

template <typename T>
void globalAvgPool(int N, int C, T* output, const T* input,
                   const T* prevLayerBias, bool nhwc, sycl::queue &sycl_queue) {
  const int kPlaneSize = 64;

  const bool fp16 = std::is_same<sycl::half, T>::value;
  if (nhwc) {
    assert(fp16);
    // For NHWC fp16, simply launch N blocks, each with C threads.
    /*
    DPCT1049:11: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      
      sycl_queue.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                            sycl::range<3>(1, 1, C)),
          [=](sycl::nd_item<3> item_ct1) {
            globalAvgPool_kernel_NHWC_fp16((sycl::half*)output,
                                           (sycl::half*)input,
                                           (sycl::half*)prevLayerBias,
                                           N * C * kPlaneSize, N * C, item_ct1);
          });
    }
  } else {
    // For NCHW layout (used with fp32),
    // each warp processes a full plane (64 elements), and writes a single
    // average N*C warps are launched.

    const int kTotalWarps = N * C;
    const int kWarpsPerBlock = 8;
    const int kBlockSize = kWarpsPerBlock * 32;

    int blocks = DivUp(kTotalWarps, kWarpsPerBlock);
    sycl_queue.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
            sycl::range<3>(1, 1, kBlockSize)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(SYCL_SUB_GROUP_SIZE)]] {
          globalAvgPool_kernel(output, input, prevLayerBias, N * C * kPlaneSize,
                               N * C, C, item_ct1);
        });
  }
}

template <typename T>
void globalScale(int N, int C, T* output, const T* input, const T* scaleBias,
                 const T* prevLayerBias, bool nhwc,
                 ActivationFunction activation, sycl::queue &sycl_queue) {
  const bool fp16 = std::is_same<sycl::half, T>::value;

  // Each thread writes one output.
  const int kBlockSize = 256;
  const int kBlocks = DivUp(N * 8 * 8 * C, kBlockSize);

  if (nhwc) {
    assert(fp16);
    sycl_queue.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, kBlocks) * sycl::range<3>(1, 1, kBlockSize),
            sycl::range<3>(1, 1, kBlockSize)),
        [=](sycl::nd_item<3> item_ct1) {
          globalScale_kernel_fp16_nhwc(
              (sycl::half*)output, (sycl::half*)input, (sycl::half*)scaleBias,
              (sycl::half*)prevLayerBias, N * C * 8 * 8, C, 8 * 8 * C,
              activation, item_ct1);
        });
  } else {
    sycl_queue.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, kBlocks) * sycl::range<3>(1, 1, kBlockSize),
            sycl::range<3>(1, 1, kBlockSize)),
        [=](sycl::nd_item<3> item_ct1) {
          globalScale_kernel(output, input, scaleBias, prevLayerBias,
                             N * C * 8 * 8, C, activation, item_ct1);
        });
  }
}

template <typename T>
void policyMap_kernel(T* output, const T* input,
                                 const short* indices, int N, int inputSize,
                                 int usedSize, int outputSize,
                                 const sycl::nd_item<3> &item_ct1) {
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  int n = tid / usedSize;
  int i = tid % usedSize;

  if (n >= N) return;

  int j = indices[i];

  if (j >= 0) {
    output[n * outputSize + j] = input[n * inputSize + i];
  }
}

template <typename T>
void PolicyMap(int N, T* output, const T* input, const short* indices,
               int inputSize, int usedSize, int outputSize, sycl::queue &sycl_queue) {
  // Each thread processes one input element
  // Only some of the threads (with valid mapping) write output
  const int kBlockSize = 256;
  const int kBlocks = DivUp(N * usedSize, kBlockSize);

  sycl_queue.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, kBlocks) *
                                             sycl::range<3>(1, 1, kBlockSize),
                                         sycl::range<3>(1, 1, kBlockSize)),
                       [=](sycl::nd_item<3> item_ct1) {
                         policyMap_kernel<T>((T*)output, (T*)input,
                                             (short*)indices, N, inputSize,
                                             usedSize, outputSize, item_ct1);
                       });
}

template <typename T = float, bool use_se, ActivationFunction activation,
          bool use_bias, bool use_skip>
void OutputInputTransform(int N, int C, int se_K, T* output, const T* input,
                          const T* skip, const T* bias, const T* w1,
                          const T* b1, const T* w2, const T* b2, sycl::queue &sycl_queue) {
  // Each thread processes entire chess board
  if (use_se == false) {
    sycl::range<3> grid_dim(1, N, DivUp(C, kOpInpTransformBlockSize));
    {
      
      sycl_queue.parallel_for(
          sycl::nd_range<3>(
              grid_dim * sycl::range<3>(1, 1, kOpInpTransformBlockSize),
              sycl::range<3>(1, 1, kOpInpTransformBlockSize)),
          [=](sycl::nd_item<3> item_ct1) {
            OutputTransform_relu_InputTransform_kernel<float, activation,
                                                       use_bias, use_skip>(
                N, C, output, input, (float*)skip, bias, item_ct1);
          });
    }
  } else if (C > kMaxResBlockFusingChannels) {
    throw Exception(
        "res block fusing opt not supported for the given data type and no "
        "of filters\n");
  } else {
    /*
    DPCT1049:12: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    
    sycl_queue.submit([&](sycl::handler& cgh) {
      /*
      DPCT1101:119: 'kMaxResBlockFusingChannels' expression was replaced
      with a value. Modify the code to use the original expression, provided
      in comments, if it is correct.
      */
      sycl::local_accessor<float, 1> shared_data_acc_ct1(
          sycl::range<1>(384 /*kMaxResBlockFusingChannels*/), cgh);
      /*
      DPCT1101:120: 'kMaxResBlockFusingChannels / 32' expression was
      replaced with a value. Modify the code to use the original expression,
      provided in comments, if it is correct.
      */
      /*
      DPCT1101:121: 'kMaxResBlockFusingSeK' expression was replaced with a
      value. Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<float, 2> shared_sums_acc_ct1(
          sycl::range<2>(12 /*kMaxResBlockFusingChannels / 32*/,
                         128 /*kMaxResBlockFusingSeK*/),
          cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                            sycl::range<3>(1, 1, C)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(SYCL_SUB_GROUP_SIZE)]] {
            OutputTransform_SE_relu_InputTransform_kernel<float, activation,
                                                          use_bias, use_skip>(
                N, C, se_K, output, input, (float*)skip, bias, w1, b1, w2, b2,
                item_ct1, shared_data_acc_ct1.get_pointer(),
                shared_sums_acc_ct1);
          });
    });
  }
}

// softmax along C dimension which is assumed to be 64
// each thread processes two elements. Each warp computes a sum (over 64
// elements)
template <typename T>
void softmax_opt_64_kernel(T* output, const T* input,
                                      const T* input2, int N,
                                      const sycl::nd_item<3> &item_ct1) {
  int index = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
  if (index >= N) return;

  float x[4];
  float ex[2];

  // Load from memory
  const bool fp16 = std::is_same<sycl::half, T>::value;
  if (fp16) {
    sycl::half inp[2];
    copyAs<int>(&inp[0], &input[index * 2]);
    x[0] = (float)inp[0];
    x[1] = (float)inp[1];
    if (input2 != nullptr) {
      copyAs<int>(&inp[0], &input2[index * 2]);
      x[2] = (float)inp[0];
      x[3] = (float)inp[1];
    }
  } else {
    copyAs<sycl::uint2>(&x[0], &input[index * 2]);
    if (input2 != nullptr) {
      copyAs<sycl::uint2>(&x[2], &input2[index * 2]);
    }
  }

  if (input2 != nullptr) {
    x[0] += x[2];
    x[1] += x[3];
  }
  float threadMax = sycl::max(x[0], x[1]);
  float maxval = warpMax(threadMax, item_ct1);
  /*
  DPCT1023:13: The SYCL sub-group does not support mask options for
  dpct::select_from_sub_group. You can specify
  "--use-experimental-features=masked-sub-group-operation" to use the
  experimental helper function to migrate __shfl_sync.
  */
  maxval = sycl::select_from_group(item_ct1.get_sub_group(), maxval, 0);

  ex[0] = sycl::exp(x[0] - maxval);
  ex[1] = sycl::exp(x[1] - maxval);

  float threadSum = ex[0] + ex[1];
  float Sum = warpReduce(threadSum, item_ct1);
  /*
  DPCT1023:14: The SYCL sub-group does not support mask options for
  dpct::select_from_sub_group. You can specify
  "--use-experimental-features=masked-sub-group-operation" to use the
  experimental helper function to migrate __shfl_sync.
  */
  Sum = sycl::select_from_group(item_ct1.get_sub_group(), Sum, 0);

  ex[0] = ex[0] / Sum;
  ex[1] = ex[1] / Sum;

  // Store to memory
  if (fp16) {
    sycl::half op[2];
    op[0] = (sycl::half)ex[0];
    op[1] = (sycl::half)ex[1];
    copyAs<int>(&output[index * 2], &op[0]);
  } else {
    copyAs<sycl::uint2>(&output[index * 2], &ex[0]);
  }
}

// N * C Tensors
// performs softmax along the C dimension
// Each thread processes one element
// Sums are computed in shared memory
// C threads per block, N blocks
template <typename T>
void softmax_kernel(T* output, const T* input, const T* input2,
                    const sycl::nd_item<3> &item_ct1, float &localsum,
                    float &localmax) {
  int n = item_ct1.get_group(2);
  int c = item_ct1.get_local_id(2);
  int C = item_ct1.get_local_range(2);
  int index = n * C + c;
  sycl::atomic_ref<float, sycl::memory_order::relaxed,
                   sycl::memory_scope::work_group> maxval(localmax);
  sycl::atomic_ref<float, sycl::memory_order::relaxed,
                   sycl::memory_scope::work_group> sum(localsum);

  // softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)

  float x = (float)input[index];
  if (input2 != nullptr) x += (float)input2[index];

  if (c == 0) {
    sum = 0;
    maxval = x;
  }

  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Get max across warp first, and then update across C dimension
  float warpmax = warpMax(x, item_ct1);
  if ((c & 0x1F) == 0) maxval.fetch_max(warpmax);

  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  float ex = sycl::exp(x - maxval);

  // compute warp wide sums first
  float val = warpReduce(ex, item_ct1);

  // update shared memory sum across C dimension
  if ((c & 0x1F) == 0)
      sum.fetch_add(val);

  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  float op = ex / sum;

  output[index] = (T)op;
}

template <typename T>
void Softmax(int N, int C, T* output, const T* input, const T* input2, sycl::queue &sycl_queue) {
  if (C == 64) {
    int size = N * 32;  // Total no of threads needed
    const int kBlockSize = 256;
    int blocks = DivUp(size, kBlockSize);
    {
      
      sycl_queue.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
              sycl::range<3>(1, 1, kBlockSize)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(SYCL_SUB_GROUP_SIZE)]] {
            softmax_opt_64_kernel<T>(output, input, input2, size, item_ct1);
          });
    }
  } else {
    /*
    DPCT1049:15: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    sycl_queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 0> sum_acc_ct1(cgh);
      sycl::local_accessor<float, 0> maxval_acc_ct1(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                            sycl::range<3>(1, 1, C)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(SYCL_SUB_GROUP_SIZE)]] {
            softmax_kernel<T>(output, input, input2, item_ct1, sum_acc_ct1,
                              maxval_acc_ct1);
          });
    });
  }
}

[[gnu::always_inline]]
inline float shared_sum_for_layer_norm(
    float x, const sycl::nd_item<3>& item_ct1,
    sycl::local_accessor<float, 2> sum) {
  // compute warp-wide sum
  float s = warpReduce(x, item_ct1);

  // warp-wide sums
  // Max product of the two dimension for the below array is 16 (512/32), but
  // we make each dimension 16 for simplicity. if shared memory capacity is the
  // bottleneck (it's not), we can convert these to single dim array and
  // dynamically index

  // compute sum across C dimension using the warp wide partial sums
  if (item_ct1.get_local_id(2) == 0)
      sum[item_ct1.get_local_id(0)][item_ct1.get_local_id(1)] = s;
  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  if (item_ct1.get_local_id(2) == 0 && item_ct1.get_local_id(1) == 0) {
    float cSum = 0;
    for (int j = 0; j < item_ct1.get_local_range(1); j++) cSum +=
        sum[item_ct1.get_local_id(0)][j];
    sum[item_ct1.get_local_id(0)][0] = cSum;
  }
  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // s now contains the sum across C dimension
  return sum[item_ct1.get_local_id(0)][0];
}

// Each thread processes 4 elements
// 1. Perform Bias add, and skip add
// 2. Perform layer norm (normalize across C dimension)
template <typename T>
void layer_norm_kernel(int N, int C, T* output, const T* input, const T* bias,
                       const T* skip, const T* gammas, const T* betas, float ep,
                       float alpha, ActivationFunction act,
                       const sycl::nd_item<3>& item_ct1,
                       sycl::local_accessor<float, 2> sum) {
  int n = item_ct1.get_group(2) * item_ct1.get_local_range(0) +
          item_ct1.get_local_id(0);
  if (n >= N) return;
  int c = (item_ct1.get_local_id(1) * 32 + item_ct1.get_local_id(2)) * 16;
  bool oobThread = c >= C;

  int biasIndex = c;
  int tensorIndex = n * C + c;

  float val[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  float oth[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  const bool fp16 = std::is_same<sycl::half, T>::value;
  if (!oobThread) {
    // Load from memory (16 elements a time)
    if (fp16) {
      sycl::half inp[8];
      copyAs<sycl::uint4>(&inp[0], &input[tensorIndex]);
      for (int i = 0; i < 8; i++) val[i] = (float)inp[i];
      copyAs<sycl::uint4>(&inp[0], &input[tensorIndex + 8]);
      for (int i = 0; i < 8; i++) val[i + 8] = (float)inp[i];
      copyAs<sycl::uint4>(&inp[0], &bias[biasIndex]);
      for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
      copyAs<sycl::uint4>(&inp[0], &bias[biasIndex + 8]);
      for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
      for (int i = 0; i < 16; i++) val[i] += oth[i];
    } else {
      copyAs<sycl::uint4>(&val[0], &input[tensorIndex]);
      copyAs<sycl::uint4>(&val[4], &input[tensorIndex + 4]);
      copyAs<sycl::uint4>(&val[8], &input[tensorIndex + 8]);
      copyAs<sycl::uint4>(&val[12], &input[tensorIndex + 12]);
      copyAs<sycl::uint4>(&oth[0], &bias[biasIndex]);
      copyAs<sycl::uint4>(&oth[4], &bias[biasIndex + 4]);
      copyAs<sycl::uint4>(&oth[8], &bias[biasIndex + 8]);
      copyAs<sycl::uint4>(&oth[12], &bias[biasIndex + 12]);
      for (int i = 0; i < 16; i++) val[i] += oth[i];
    }
  }

  if (!oobThread) {
    if (skip != nullptr) {
      // Load from memory (16 elements a time)
      if (fp16) {
        sycl::half inp[8];
        copyAs<sycl::uint4>(&inp[0], &skip[tensorIndex]);
        for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
        copyAs<sycl::uint4>(&inp[0], &skip[tensorIndex + 8]);
        for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
      } else {
        copyAs<sycl::uint4>(&oth[0], &skip[tensorIndex]);
        copyAs<sycl::uint4>(&oth[4], &skip[tensorIndex + 4]);
        copyAs<sycl::uint4>(&oth[8], &skip[tensorIndex + 8]);
        copyAs<sycl::uint4>(&oth[12], &skip[tensorIndex + 12]);
      }
    }
  }

  // 1. Compute mean
  float s = 0;
  if (!oobThread)
    if (skip != nullptr) {
      for (int i = 0; i < 16; i++) {
        val[i] = activate(val[i], act) * alpha + oth[i];
        s += val[i];
      }
    } else {
      for (int i = 0; i < 16; i++) {
        val[i] = activate(val[i], act) * alpha;
        s += val[i];
      }
    }

  s = shared_sum_for_layer_norm(s, item_ct1, sum);
  float mean = s / C;

  // 2. Compute varience
  s = 0;
  if (!oobThread)
    for (int i = 0; i < 16; i++) {
      float d = val[i] - mean;
      float d_sq = d * d;
      s += d_sq;
    }
  s = shared_sum_for_layer_norm(s, item_ct1, sum);
  float var = s / C;

  if (!oobThread) {
    // Load from memory (16 elements a time)
    if (fp16) {
      sycl::half inp[8];
      copyAs<sycl::uint4>(&inp[0], &gammas[biasIndex]);
      for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
      copyAs<sycl::uint4>(&inp[0], &gammas[biasIndex + 8]);
      for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
    } else {
      copyAs<sycl::uint4>(&oth[0], &gammas[biasIndex]);
      copyAs<sycl::uint4>(&oth[4], &gammas[biasIndex + 4]);
      copyAs<sycl::uint4>(&oth[8], &gammas[biasIndex + 8]);
      copyAs<sycl::uint4>(&oth[12], &gammas[biasIndex + 12]);
    }
  }

  // 3. Normalize
  for (int i = 0; i < 16; i++) {
    float d = val[i] - mean;
    float norm = d / sycl::sqrt(var + ep);
    float op = norm * oth[i];
    val[i] = op;
  }

  if (!oobThread) {
    // Load from memory (16 elements a time)
    if (fp16) {
      sycl::half inp[8];
      copyAs<sycl::uint4>(&inp[0], &betas[biasIndex]);
      for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
      copyAs<sycl::uint4>(&inp[0], &betas[biasIndex + 8]);
      for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
    } else {
      copyAs<sycl::uint4>(&oth[0], &betas[biasIndex]);
      copyAs<sycl::uint4>(&oth[4], &betas[biasIndex + 4]);
      copyAs<sycl::uint4>(&oth[8], &betas[biasIndex + 8]);
      copyAs<sycl::uint4>(&oth[12], &betas[biasIndex + 12]);
    }
  }

  for (int i = 0; i < 16; i++) {
    val[i] += oth[i];
  }

  if (!oobThread) {
    // Write to memory
    if (fp16) {
      sycl::half op[8];
      for (int i = 0; i < 8; i++) op[i] = (sycl::half)val[i];
      copyAs<sycl::uint4>(&output[tensorIndex], &op[0]);
      for (int i = 0; i < 8; i++) op[i] = (sycl::half)val[i + 8];
      copyAs<sycl::uint4>(&output[tensorIndex + 8], &op[0]);
    } else {
      copyAs<sycl::uint4>(&output[tensorIndex], &val[0]);
      copyAs<sycl::uint4>(&output[tensorIndex + 4], &val[4]);
      copyAs<sycl::uint4>(&output[tensorIndex + 8], &val[8]);
      copyAs<sycl::uint4>(&output[tensorIndex + 12], &val[12]);
    }
  }
}

// add (optional) skip connection to input, and then perform Layer normalization
// normalization is done across C dimension (i.e, sums and std deviations taken
// over elements in C dim)
template <typename T>
void LayerNorm(int N, int C, T* output, const T* input, const T* bias,
               const T* skip, const T* gammas, const T* betas, float ep,
               float alpha, ActivationFunction act, sycl::queue &sycl_queue) {
  // process 4 elements per thread to achieve close to peak memory bandwidth
  if (C % 16 != 0) throw Exception("unsupported filter size");
  if (C > 8192) throw Exception("unsupported filter size");

  sycl::range<3> blockDim(1, 1, 1), gridDim(1, 1, 1);
  blockDim[2] = 32;
  blockDim[1] = DivUp(C / 16, 32);
  blockDim[0] = 1;
  gridDim[2] = N;
  gridDim[1] = 1;
  gridDim[0] = 1;

  /*
  DPCT1049:17: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    
    sycl_queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 2> sum_acc_ct1(sycl::range<2>(16, 16), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(gridDim * blockDim, blockDim),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(SYCL_SUB_GROUP_SIZE)]] {
            layer_norm_kernel<T>(N, C, output, input, bias, skip, gammas, betas,
                                 ep, alpha, act, item_ct1, sum_acc_ct1);
          });
    });
  }
}

// Compute promotion logits in a single kernel
// keys matrix is of N * 64 * C (but we use only last 8 from the 'rows'
// dimension, so N * 8 * C)
// ppo matrix is 4 * C (weights for dense layer / matrix multiplication)
// policy_attn_logits matrix is N * 64 * 64, but we use only 8x8 part of it
// from each batch dimension (so, N * 8 * 8)
// output matrix (promotion logits) is of N * 8 * 24 size
template <typename T>
void promotion_logits_kernel(int C, T* output, const T* keys,
                                        const T* ppo,
                                        const T* policy_attn_logits,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::local_accessor<float, 2> promotion_offsets) {
  constexpr int output_stride = 64 * 64 + 8 * 24;
  int n = item_ct1.get_group(2);     // [0..N)
  int y = item_ct1.get_local_id(1);  // [0..8)
  int x = item_ct1.get_local_id(2);  // [0..24)     // Can split into 8 * 3

  int threadInGroup = item_ct1.get_local_id(1) * 24 + item_ct1.get_local_id(2);

  // phase 1 : compute promotion_offsets by multiplying keys and ppo matrices
  const T* keys_start =
      keys + n * 64 * C + C * 56;  // we are interested only in last 8 out of 64
                                   // 'rows' of keys matrix

  // only 32 threads out of 192 in the group are active in this phase, and each
  // thread computes one element of the promotion_offsets matrix
  // TODO: opt idea1, can use more threads to reduce the length of the loop for
  // the matrix multiply (do parallel reduction of partial sums later)
  //       opt idea2, the below loop for matrix mul has very poor memory access
  //       pattern, can do the loop over 32, and do parallel reductions
  if (threadInGroup < 32) {
    int x = threadInGroup % 4;
    int y = threadInGroup / 4;

    float S = 0;
    for (int i = 0; i < C;
         i++) {  // TODO: modify to loop over 32 instead of C (doing parallel
                 // reductions for the 32 sums)
      float a = (float)keys_start[y * C + i];
      float b =
          (float)ppo[x * C + i];  // weight matrix is transposed (col major)
      S += a * b;
    }

    // write the product (promotion_offsets) in shared memory
    promotion_offsets[x][y] = S;
  }

  /*
  DPCT1065:69: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // phase 2: add the last "row" to the other 3
  // #knight offset is added to the other three
  // promotion_offsets = promotion_offsets[:, :3, :] + promotion_offsets[:, 3:4,
  // :]
  // Only 24 threads in the group are active in this phase
  if (threadInGroup < 32) {
    int x = threadInGroup % 4;
    int y = threadInGroup / 4;
    if (x < 3) {
      promotion_offsets[x][y] += promotion_offsets[3][y];
    }
  }

  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // phase 3: add 8x8 chunk of policy_attn_logits matrix to promotion offsets
  //          the output is 3x8x8 (written as 8 * 24)
  // All threads are active in this phase and they compute one element each
  int w = x / 3;
  int c = x % 3;

  // n_promo_logits = matmul_qk[:, -16:-8, -8:]  # default traversals from rank
  // 7 to rank 8
  float n_promo_logit =
      (float)policy_attn_logits[n * output_stride + (48 + y) * 64 + (56 + w)];
  float promo_offset = promotion_offsets[c][w];

  float op = n_promo_logit + promo_offset;

  output[n * output_stride + threadInGroup] = (T)op;
}

template <typename T>
void ComputePromotionLogits(int N, int C, T* output, const T* keys,
                            const T* ppo, const T* policy_attn_logits, sycl::queue &sycl_queue) {
  // N blocks
  // 8 * 24 threads
  // Each thread computes a single output element
  sycl::range<3> blockDim(1, 8, 24);
  sycl_queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 2> promotion_offsets_acc_ct1(
        sycl::range<2>(4, 8), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, N) * blockDim, blockDim),
        [=](sycl::nd_item<3> item_ct1) {
          promotion_logits_kernel<T>(C, output, keys, ppo, policy_attn_logits,
                                     item_ct1, promotion_offsets_acc_ct1);
        });
  });
}

template <typename T>
void preprocess_for_attention_body_kernel(
    T* output, const T* input, const T* encoding, int input_size,
    int encoding_size, bool is_pe_dense_embedding,
    const sycl::nd_item<3> &item_ct1) {
  int n = item_ct1.get_group(2);
  int hw = item_ct1.get_group(1);
  int c = item_ct1.get_local_id(2) +
          item_ct1.get_local_range(2) * item_ct1.get_group(0);
  if (c >= input_size + encoding_size) return;

  T op;
  if (c >= input_size) {
    // concatenate from position encoding array
    if (is_pe_dense_embedding) {
      op = (T)(encoding[n * 64 * encoding_size + hw * encoding_size + (c - input_size)]);
    } else {
      op = (T)(encoding[64 * hw + (c - input_size)]);
    }
  } else {
    op = input[n * input_size * 64 + c * 64 + hw];  // nchw
  }

  int outputC = input_size + encoding_size;

  // convert to nhwc
  output[n * 64 * outputC + hw * outputC + c] = op;
}

template <typename T>
void inputPreprocessForAttentionBody(T* output, const T* input,
                                     const T* encoding, int N, int input_size,
                                     int encoding_size,
                                     bool is_pe_dense_embedding,
                                     sycl::queue &sycl_queue) {
  // N * 64 blocks
  // (kInputPlanes + kNumPosEncodingChannels) threads
  // Each thread computes a single output element
  sycl::range<3> gridSize = sycl::range<3>(1, 64, N);
  sycl::range<3> blockSize(1, 1, 1);
  blockSize[2] = sycl::min(input_size + encoding_size, 512);
  blockSize[1] = 1;
  blockSize[0] = 1;
  gridSize[0] = DivUp(input_size + encoding_size, blockSize[2]);

  sycl_queue.parallel_for(
      sycl::nd_range<3>(gridSize * blockSize, blockSize),
      [=](sycl::nd_item<3> item_ct1) {
        preprocess_for_attention_body_kernel<T>(output, input, encoding,
                                                input_size, encoding_size,
                                                is_pe_dense_embedding, item_ct1);
      });
}

template <typename T>
void input_gating_kernel(T* output, const T* input, const T* mult,
                                    const T* add, int HW, int C,
                                    const sycl::nd_item<3> &item_ct1) {
  int n_offset = item_ct1.get_group(0) * HW * C;
  int idx = item_ct1.get_local_id(1) * C +
            item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);  // index in input
  int idxT = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2)) *
                 HW +
             item_ct1.get_local_id(
                 1);  // index in transposed weights arrays mult and add.

  if (idx < HW * C) {
    // Combine multiply gating, add gating and weights transpose.
    float op =
        (float)input[n_offset + idx] * (float)mult[idxT] + (float)add[idxT];
    output[n_offset + idx] = (T)op;
  }
}

template <typename T>
void applyInputGating(T* output, const T* input, const T* mult, const T* add,
                      int N, int HW, int C, sycl::queue &sycl_queue) {
  // Multiple blocks to fit into each input area / volume
  // Block x position indicates horizontal section of area
  // Block y position indicates batch
  // Each thread computes a single output element
  sycl::range<3> blockSize(1, 1, 1), gridSize(1, 1, 1);
  blockSize[2] = DivUp(512, HW);
  blockSize[1] = HW;
  blockSize[0] = 1;
  gridSize[2] = DivUp(C, blockSize[2]);
  gridSize[1] = 1;
  gridSize[0] = N;
  
  sycl_queue.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize),
                       [=](sycl::nd_item<3> item_ct1) {
                         input_gating_kernel<T>(output, input, mult, add, HW, C,
                                                item_ct1);
                       });
}

template<typename T, int kWorkPerThread>
static void genOffsetPointers_kernel(T** offsets, int heads, int block_size,
                                     int depth, int d_model, T* k, T* q, T* b1,
                                     T* v, T* b2,
                                     const sycl::nd_item<1>& item_ct) {
  const int i = item_ct.get_global_id(0) * kWorkPerThread;
  if (i >= block_size) return;
  const int h = i % heads;
  const int n = i / heads;
  int w;
  T* res[kWorkPerThread];
  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = k + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = q + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w + block_size] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = b1 + i * 64 * 64 + w * 64 * 64;
    offsets[i + w + 2 * block_size] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = v + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w + 3 * block_size] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] =  b2 + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w + 4 * block_size] = res[w];
  }
}

template <typename T>
void genOffsetPointers(T** offsets, int heads, int max_batch, int depth,
                       int d_model, T* k, T* q, T* b1,
                       T* v, T* b2, sycl::queue& sycl_queue) {
  const int block_size = heads * max_batch;
  // Process two elements per thread to use 128 bit store instructions.
  constexpr int kWorkPerThread = 2;
  constexpr int kWorkGroupSize = 128;
  if (block_size % kWorkPerThread != 0) {
    // Handle odd block sizes.
    sycl::range<1> global(DivUp(block_size, kWorkGroupSize));
    sycl::range<1> local(kWorkGroupSize);
    sycl_queue.parallel_for(sycl::nd_range<1>(global*local, local),
        [=](sycl::nd_item<1> item_ct) {
        genOffsetPointers_kernel<T, 1>(offsets, heads, block_size,
                                       depth, d_model, k, q, b1,
                                       v, b2, item_ct);
        });
  } else {
    // Handle even block size
    sycl::range<1> global(DivUp(block_size, kWorkGroupSize*kWorkPerThread));
    sycl::range<1> local(kWorkGroupSize);
    sycl_queue.parallel_for(sycl::nd_range<1>(global*local, local),
        [=](sycl::nd_item<1> item_ct) {
        genOffsetPointers_kernel<T, kWorkPerThread>(offsets, heads, block_size,
                                                    depth, d_model, k, q, b1,
                                                    v, b2, item_ct);
        });
  }
}

// Template instantiation.
template void copyTypeConverted<sycl::half, float>(sycl::half* op, float* ip, int N, sycl::queue &sycl_queue);
template void copyTypeConverted<float, sycl::half>(float* op, sycl::half* ip, int N, sycl::queue &sycl_queue);
template void copyTypeConverted<float, float>(float* op, float* ip, int N, sycl::queue &sycl_queue);
template void copyTypeConverted<sycl::half, sycl::half>(sycl::half* op, sycl::half* ip, int N, sycl::queue &sycl_queue);

template void batchNorm<float>(float* output, const float* input,
                               const float* skipInput, int N, int C, int H,
                               int W, float* means, float* var_multipliers,
                               ActivationFunction activation, sycl::queue &sycl_queue);

template void batchNorm<sycl::half>(sycl::half* output, const sycl::half* input,
                              const sycl::half* skipInput, int N, int C, int H, int W,
                              float* means, float* var_multipliers,
                              ActivationFunction activation, sycl::queue &sycl_queue);

template void addVectors<float>(float* c, float* a, float* b, int size,
                                int asize, int bsize, ActivationFunction act, sycl::queue &sycl_queue);

template void addVectors<sycl::half>(sycl::half* c, sycl::half* a, sycl::half* b, int size, int asize,
                               int bsize, ActivationFunction act, sycl::queue &sycl_queue);

template void addVectorsHNC_NHC<float>(float* a, float* b, int N, int H, int C, sycl::queue &sycl_queue);
template void addVectorsHNC_NHC<sycl::half>(sycl::half* a, sycl::half* b, int N, int H, int C, sycl::queue &sycl_queue);

template void addBiasBatched<float>(float* output, const float* input,
                                    const float* bias, int Batch, int N, int C,
                                    ActivationFunction activation, sycl::queue &sycl_queue);

template void addBiasBatched<sycl::half>(sycl::half* output, const sycl::half* input,
                                   const sycl::half* bias, int Batch, int N, int C,
                                   ActivationFunction activation, sycl::queue &sycl_queue);

template void addBiasBatched<float>(float* output, const float* input,
                                    const float* bias, int Batch, int N, int C,
                                    int Nstride, ActivationFunction activation, sycl::queue &sycl_queue);

template void addBiasBatched<sycl::half>(sycl::half* output, const sycl::half* input,
                                   const sycl::half* bias, int Batch, int N, int C,
                                   int Nstride, ActivationFunction activation, sycl::queue &sycl_queue);

template void addBias_NCHW<float>(float* c, float* a, float* b, int N, int C,
                                  int H, int W, ActivationFunction activation, sycl::queue &sycl_queue);

template void addBias_NCHW<sycl::half>(sycl::half* c, sycl::half* a, sycl::half* b, int N, int C, int H,
                                 int W, ActivationFunction activation, sycl::queue &sycl_queue);

template void globalAvgPool<float>(int N, int C, float* output,
                                   const float* input,
                                   const float* prevLayerBias, bool nhwc, sycl::queue &sycl_queue);

template void globalAvgPool<sycl::half>(int N, int C, sycl::half* output, const sycl::half* input,
                                  const sycl::half* prevLayerBias, bool nhwc, sycl::queue &sycl_queue);

template void globalScale<float>(int N, int C, float* output,
                                 const float* input, const float* scaleBias,
                                 const float* prevLayerBias, bool nhwc,
                                 ActivationFunction activation, sycl::queue &sycl_queue);

template void globalScale<sycl::half>(int N, int C, sycl::half* output, const sycl::half* input,
                                const sycl::half* scaleBias,
                                const sycl::half* prevLayerBias, bool nhwc,
                                ActivationFunction activation, sycl::queue &sycl_queue);

template void PolicyMap<float>(int N, float* output, const float* input,
                               const short* indices, int inputSize,
                               int usedSize, int outputSize, sycl::queue &sycl_queue);

template void PolicyMap<sycl::half>(int N, sycl::half* output, const sycl::half* input,
                              const short* indices, int inputSize, int usedSize,
                              int outputSize, sycl::queue &sycl_queue);

template void FilterTransform<float>(int N, int C, float* transformedFilter,
                                     const float* filter, sycl::queue &sycl_queue);

template void InputTransform<float, true>(int N, int C,
                                          float* transformed_input,
                                          const float* input, sycl::queue &sycl_queue);

template void InputTransform<float, false>(int N, int C,
                                           float* transformed_input,
                                           const float* input, sycl::queue &sycl_queue);

template void OutputTransform<float, true, ACTIVATION_RELU, true, true, false,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue &sycl_queue);

template void
OutputTransform<float, false, ACTIVATION_RELU, true, true, false, false>(

    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, sycl::queue &sycl_queue);

template void OutputTransform<float, true, ACTIVATION_RELU, true, true, true,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue &sycl_queue);

template void OutputTransform<float, false, ACTIVATION_RELU, true, true, true,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue &sycl_queue);

template void OutputTransform<float, false, ACTIVATION_RELU, true, false, false,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue &sycl_queue);

template void OutputTransform<float, false, ACTIVATION_RELU, true, false, false,
                              true>(int N, int C, int se_K, float* output,
                                    const float* input, const float* skip,
                                    const float* bias, const float* w1,
                                    const float* b1, const float* w2,
                                    const float* b2, sycl::queue &sycl_queue);

template void OutputTransform<float, true, ACTIVATION_RELU, true, true, true,
                              true>(int N, int C, int se_K, float* output,
                                    const float* input, const float* skip,
                                    const float* bias, const float* w1,
                                    const float* b1, const float* w2,
                                    const float* b2, sycl::queue &sycl_queue);

template void OutputTransform<float, true, ACTIVATION_MISH, true, true, false,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue &sycl_queue);

template void OutputTransform<float, false, ACTIVATION_MISH, true, true, false,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue &sycl_queue);

template void OutputTransform<float, true, ACTIVATION_MISH, true, true, true,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue &sycl_queue);

template void OutputTransform<float, false, ACTIVATION_MISH, true, true, true,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue &sycl_queue);

template void OutputTransform<float, false, ACTIVATION_MISH, true, false, false,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue &sycl_queue);

template void OutputTransform<float, false, ACTIVATION_MISH, true, false, false,
                              true>(int N, int C, int se_K, float* output,
                                    const float* input, const float* skip,
                                    const float* bias, const float* w1,
                                    const float* b1, const float* w2,
                                    const float* b2, sycl::queue &sycl_queue);

template void OutputTransform<float, true, ACTIVATION_MISH, true, true, true,
                              true>(int N, int C, int se_K, float* output,
                                    const float* input, const float* skip,
                                    const float* bias, const float* w1,
                                    const float* b1, const float* w2,
                                    const float* b2, sycl::queue &sycl_queue);

template void OutputTransform<float, false, ACTIVATION_NONE, true, false, false,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue &sycl_queue);

template void OutputInputTransform<float, true, ACTIVATION_RELU, true, true>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, sycl::queue &sycl_queue);

template void OutputInputTransform<float, false, ACTIVATION_RELU, true, true>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, sycl::queue &sycl_queue);

template void OutputInputTransform<float, false, ACTIVATION_RELU, true, false>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, sycl::queue &sycl_queue);

template void OutputInputTransform<float, true, ACTIVATION_MISH, true, true>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, sycl::queue &sycl_queue);

template void OutputInputTransform<float, false, ACTIVATION_MISH, true, true>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, sycl::queue &sycl_queue);

template void OutputInputTransform<float, false, ACTIVATION_MISH, true, false>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, sycl::queue &sycl_queue);

template void Softmax<sycl::half>(int N, int C, sycl::half* output, const sycl::half* input,
                            const sycl::half* input2, sycl::queue &sycl_queue);

template void Softmax<float>(int N, int C, float* output, const float* input,
                             const float* input2, sycl::queue &sycl_queue);

template void LayerNorm<sycl::half>(int N, int C, sycl::half* output, const sycl::half* input,
                              const sycl::half* bias, const sycl::half* skip,
                              const sycl::half* gammas, const sycl::half* betas, float ep,
                              float alpha, ActivationFunction act, sycl::queue &sycl_queue);

template void LayerNorm<float>(int N, int C, float* output, const float* input,
                               const float* bias, const float* skip,
                               const float* gammas, const float* betas,
                               float ep, float alpha, ActivationFunction act, sycl::queue &sycl_queue);

template void ComputePromotionLogits<sycl::half>(int N, int C, sycl::half* output,
                                           const sycl::half* keys, const sycl::half* ppo,
                                           const sycl::half* policy_attn_logits, sycl::queue &sycl_queue);

template void ComputePromotionLogits<float>(int N, int C, float* output,
                                            const float* keys, const float* ppo,
                                            const float* policy_attn_logits, sycl::queue &sycl_queue);

template void convertNCHWtoNHWC<sycl::half, float>(sycl::half* output_tensor,
                                             const float* input_tensor, int Nin,
                                             int Cin, int Nout, int Cout, int H,
                                             int W, sycl::queue &sycl_queue);

template void convertNCHWtoNHWC<float, float>(float* output_tensor,
                                              const float* input_tensor,
                                              int Nin, int Cin, int Nout,
                                              int Cout, int H, int W, sycl::queue &sycl_queue);

template void convertNCHWtoNHWC<sycl::half, sycl::half>(sycl::half* output_tensor,
                                            const sycl::half* input_tensor, int Nin,
                                            int Cin, int Nout, int Cout, int H,
                                            int W, sycl::queue &sycl_queue);

template void inputPreprocessForAttentionBody<sycl::half>(
    sycl::half* output, const sycl::half* input, const sycl::half* encoding, int N,
    int input_size, int encoding_size, bool is_pe_dense_embedding,
    sycl::queue &sycl_queue);

template void inputPreprocessForAttentionBody<float>(
    float* output, const float* input, const float* encoding, int N,
    int input_size, int encoding_size, bool is_pe_dense_embedding,
    sycl::queue &sycl_queue);

template void applyInputGating<sycl::half>(sycl::half* output, const sycl::half* input,
                                     const sycl::half* mult, const sycl::half* add, int N,
                                     int C, int output_size, sycl::queue &sycl_queue);

template void applyInputGating<float>(float* output, const float* input,
                                      const float* mult, const float* add,
                                      int N, int C, int output_size, sycl::queue &sycl_queue);

template void genOffsetPointers<float>(float** offsets, int heads, int max_batch, int depth,
                       int d_model, float* k, float* q, float* b1,
                       float* v, float* b2, sycl::queue& sycl_queue);

template void genOffsetPointers<sycl::half>(sycl::half** offsets, int heads, int max_batch, int depth,
                       int d_model, sycl::half* k, sycl::half* q, sycl::half* b1,
                       sycl::half* v, sycl::half* b2, sycl::queue& sycl_queue);
}  // namespace sycldnn_backend
}  // namespace lczero
