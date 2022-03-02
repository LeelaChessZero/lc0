/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

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
#include <algorithm>
#include <cassert>
#include <functional>
#include <list>
#include <memory>
#include <mutex>

#include "cuda_common.h"
#include "inputs_outputs.h"
#include "kernels.h"
#include "layers.h"
#include "neural/factory.h"
#include "neural/network_legacy.h"
#include "neural/shared/policy_map.h"
#include "utils/bititer.h"
#include "utils/exception.h"

//#define DEBUG_RAW_NPS

namespace lczero {
using namespace cudnn_backend;

#if 0
// debug code to dump allocation in GPU memory
void dumpTensor(void *memory, int elements, const char *message, bool fp16 = false)
{
    printf("\n%s\n", message);
    int elementSize = (int) (fp16 ? sizeof(half) : sizeof(float));
    int bytes = elements * elementSize;
    void *temp = malloc(bytes);
    cudaMemcpy(temp, memory, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < elements; i++)
    {
        float val;
        if (fp16) 
        {
            half *arr = (half*)temp;
            val = (float)arr[i];
        }
        else
        {
            float *arr = (float *)temp;
            val = arr[i];
        }
        printf("%8.4f ", val);
        if ((i % 8) == 7) printf("\n");
    }
    free(temp);
    printf("\n");
}
#endif

template <typename DataType>
class CudnnNetwork;

template <typename DataType>
class CudnnNetworkComputation : public NetworkComputation {
 public:
  CudnnNetworkComputation(CudnnNetwork<DataType>* network, bool wdl,
                          bool moves_left);
  ~CudnnNetworkComputation();

  void AddInput(InputPlanes&& input) override {
    const auto iter_mask =
        &inputs_outputs_->input_masks_mem_[batch_size_ * kInputPlanes];
    const auto iter_val =
        &inputs_outputs_->input_val_mem_[batch_size_ * kInputPlanes];

    int i = 0;
    for (const auto& plane : input) {
      iter_mask[i] = plane.mask;
      iter_val[i] = plane.value;
      i++;
    }

    batch_size_++;
  }

  void ComputeBlocking() override;

  int GetBatchSize() const override { return batch_size_; }

  float GetQVal(int sample) const override {
    if (wdl_) {
      auto w = inputs_outputs_->op_value_mem_[3 * sample + 0];
      auto l = inputs_outputs_->op_value_mem_[3 * sample + 2];
      return w - l;
    } else {
      return inputs_outputs_->op_value_mem_[sample];
    }
  }

  float GetDVal(int sample) const override {
    if (wdl_) {
      auto d = inputs_outputs_->op_value_mem_[3 * sample + 1];
      return d;
    } else {
      return 0.0f;
    }
  }

  float GetPVal(int sample, int move_id) const override {
    return inputs_outputs_->op_policy_mem_[sample * kNumOutputPolicy + move_id];
  }

  float GetMVal(int sample) const override {
    if (moves_left_) {
      return inputs_outputs_->op_moves_left_mem_[sample];
    }
    return 0.0f;
  }

 private:
  // Memory holding inputs, outputs.
  std::unique_ptr<InputsOutputs> inputs_outputs_;
  int batch_size_;
  bool wdl_;
  bool moves_left_;

  CudnnNetwork<DataType>* network_;
};

template <typename DataType>
class CudnnNetwork : public Network {
 public:
  CudnnNetwork(const WeightsFile& file, const OptionsDict& options)
      : capabilities_{file.format().network_format().input(),
                      file.format().network_format().moves_left()} {
    LegacyWeights weights(file.weights());
    gpu_id_ = options.GetOrDefault<int>("gpu", 0);

    conv_policy_ = file.format().network_format().policy() ==
                   pblczero::NetworkFormat::POLICY_CONVOLUTION;

    max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);

    showInfo();

    int total_gpus;
    ReportCUDAErrors(cudaGetDeviceCount(&total_gpus));

    if (gpu_id_ >= total_gpus)
      throw Exception("Invalid GPU Id: " + std::to_string(gpu_id_));

    cudaDeviceProp deviceProp = {};
    cudaGetDeviceProperties(&deviceProp, gpu_id_);
    showDeviceInfo(deviceProp);

    // Select GPU to run on (for *the current* thread).
    ReportCUDAErrors(cudaSetDevice(gpu_id_));

    ReportCUDNNErrors(cudnnCreate(&cudnn_));
    ReportCUBLASErrors(cublasCreate(&cublas_));

    // Default layout is nchw.
    nhwc_ = false;
    bool hasTensorCores = false;
    constexpr bool fp16 = std::is_same<half, DataType>::value;

    if (fp16) {
      // Check if the GPU support FP16.

      if ((deviceProp.major == 6 && deviceProp.minor != 1) ||
          (deviceProp.major == 5 && deviceProp.minor == 3)) {
        // FP16 without tensor cores supported on GP100 (SM 6.0) and Jetson
        // (SM 5.3 and 6.2). SM 6.1 GPUs also have FP16, but slower than FP32.
        // nhwc_ remains false.
      } else if (deviceProp.major >= 7) {
        // NHWC layout is faster with Tensor Cores when using cudnn's implicit
        // gemm algorithm.
        // Supported on Volta and Turing (and hopefully future GPUs too).

        // Some GPUs (GTX 16xx) are SM 7.5 but don't have tensor cores
        // enabling TENSOR_OP_MATH or nhwc_ layout for them works but is
        // very very slow (likely because the system emulates it).
        if (!strstr(deviceProp.name, "GTX 16")) {
          hasTensorCores = true;
          nhwc_ = true;
        }
      } else {
        throw Exception("Your GPU doesn't support FP16");
      }

      // Override if forced from backend option
      if (!options.IsDefault<bool>("nhwc")) nhwc_ = options.Get<bool>("nhwc");
    }

    if (hasTensorCores)
      ReportCUBLASErrors(cublasSetMathMode(
          cublas_,
          CUBLAS_TENSOR_OP_MATH));  // Deprecated on CUDA 11.0 and later
    else if (fp16)
      ReportCUBLASErrors(cublasSetMathMode(
          cublas_,
          CUBLAS_PEDANTIC_MATH));  // Explicitly set PEDANTIC_MATH mode to
                                   // avoid cublas bug of making use of tensor
                                   // core math on TU11x GPUs that don't
                                   // support it.

    const int kNumInputPlanes = kInputPlanes;
    const int kNumFilters = (int)weights.input.biases.size();
    numBlocks_ = (int)weights.residual.size();

    // Use our custom winograd for residual tower convolutions for most cases:
    //
    //  1. Should be always faster than cudnn's winograd that we use for fp32,
    //  and for fp16 on GPUs without tensor cores
    //
    //  2. Should also be faster than cudnn's implicit GEMM on GPUs with tensor
    //     cores too, but only for networks with 256 or higher no. of filters.
    //
    //  3. Currently a bug in cublas makes it slower on RTX GPUs with fp16 so
    //  it's disabled. TODO: Enable it once the bug has been fixed and it's
    //  tested to be faster. Putting check for cuda 11 for now.

    if (fp16) {
      int cuda_version;
      cudaRuntimeGetVersion(&cuda_version);
      if (!hasTensorCores)
        use_custom_winograd_ = false;
      else if (kNumFilters >= 256 &&
               !(deviceProp.major == 7 && deviceProp.minor == 5 &&
                 cuda_version < 11000))
        use_custom_winograd_ = true;
      else
        use_custom_winograd_ = false;
    } else {
      use_custom_winograd_ = true;
    }

    // Warn if the memory required for storing transformed weights is
    // going to exceed 40% of total video memory, force custom_winograd off
    // if it's going to exceed 50% of memory.
    size_t residual_single_layer_weight_size =
        3 * 3 * kNumFilters * kNumFilters * sizeof(DataType);
    size_t residual_weight_size =
        residual_single_layer_weight_size * numBlocks_ * 2;
    size_t transformed_residual_weight_size = residual_weight_size * 4;

    if (residual_weight_size > 0.6 * deviceProp.totalGlobalMem) {
      CERR << "Low video memory detected. You may run into OOM errors. Please "
              "consider using a smaller network.";
    }

    const bool custom_winograd_override =
        !options.IsDefault<bool>("custom_winograd");

    if (!custom_winograd_override && use_custom_winograd_ &&
        transformed_residual_weight_size > 0.5 * deviceProp.totalGlobalMem) {
      CERR << "WARNING: Low GPU video memory. Turning off custom_winograd "
              "path. You may still run into OOM errors. "
              "Please consider using a smaller network.";
      use_custom_winograd_ = false;
    }

    // Override if set in backend-opts.
    if (custom_winograd_override)
      use_custom_winograd_ = options.Get<bool>("custom_winograd");

    if (use_custom_winograd_ &&
        transformed_residual_weight_size > 0.4 * deviceProp.totalGlobalMem) {
      CERR << "WARNING: Low GPU video memory. You may still run into OOM "
              "errors. Try with backend-opts=custom_winograd=false, or "
              "using a smaller network.";
    }

    // Winograd needs nchw tensor layout.
    if (use_custom_winograd_) nhwc_ = false;

    use_res_block_winograd_fuse_opt_ = false;
    if (use_custom_winograd_) {
      // Disable res block fusing for fp32 for now.
      // TODO: make it work for filters not a multiple of 32.
      if (kNumFilters % 32 == 0 && fp16) {
        use_res_block_winograd_fuse_opt_ = true;
      }
      // Override if set in backend-opts.
      if (!options.IsDefault<bool>("res_block_fusing")) {
        use_res_block_winograd_fuse_opt_ =
            options.Get<bool>("res_block_fusing");
      }
    }

    const bool use_gemm_ex = deviceProp.major >= 5;

    // 0. Check for SE.
    has_se_ = false;
    if (weights.residual[0].has_se) {
      has_se_ = true;
    }

    const bool mish_net = file.format().network_format().default_activation() ==
                          pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH;

    // 1. Allocate scratch space (used internally by cudnn to run convolutions,
    //     and also for format/layout conversion for weights).
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnTensorDescriptor_t xDesc;
    cudnnCreateFilterDescriptor(&wDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnConvolutionFwdAlgo_t conv_algo;

    const int maxChannels = std::max(kInputPlanes, kNumFilters);

    const cudnnDataType_t datatype = fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
    const cudnnTensorFormat_t layout =
        nhwc_ ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;

    ReportCUDNNErrors(cudnnSetFilter4dDescriptor(
        wDesc, datatype, layout, maxChannels, maxChannels, 3, 3));

    ReportCUDNNErrors(cudnnSetTensor4dDescriptor(
        xDesc, layout, datatype, max_batch_size_, maxChannels, 8, 8));

    ReportCUDNNErrors(cudnnSetConvolution2dDescriptor(
        convDesc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, datatype));

    // It will fall back to non-tensor math if not supported.
    ReportCUDNNErrors(
        cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

    if (nhwc_) {
      conv_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    } else {
      conv_algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
    }

    // Query expected scratch space from cudnn.
    ReportCUDNNErrors(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_, xDesc, wDesc, convDesc, xDesc, conv_algo, &scratch_size_));

    // Have some minumum as we also use this for transforming weights.
    size_t max_weight_size = 128 * 1024 * 1024;

    // parts from scratch allocation are suballocated to hold various weights
    // and biases when transforming winograd weights (one layer at a time), 128
    // MB is way more than that what we need but make sure it's at least 3x of
    // single layer's weight size to be safe.
    if (max_weight_size < 3 * residual_single_layer_weight_size)
      max_weight_size = 3 * residual_single_layer_weight_size;

    if (scratch_size_ < max_weight_size) scratch_size_ = max_weight_size;

    size_t transformed_tensor_size = 0;
    if (use_custom_winograd_) {
      // Need additional space for transformed input/outputs which are 36/16
      // times size (4x4 block transformed into 6x6).
      transformed_tensor_size = (size_t)(max_batch_size_ * kNumFilters * 64 *
                                         (36.0 / 16.0) * sizeof(DataType));
      scratch_size_ = std::max(scratch_size_, 2 * transformed_tensor_size);
    }

    ReportCUDAErrors(cudaMalloc(&scratch_mem_, scratch_size_));
#ifdef DEBUG_RAW_NPS
    CERR << "allocated " << scratch_size_ << " bytes for scratch memory";
#endif

    // 2. Build the network, and copy the weights to GPU memory.

    // Input.
    if (use_custom_winograd_) {
      auto inputConv = std::make_unique<FusedWinogradConvSELayer<DataType>>(
          nullptr, kNumFilters, 8, 8, kNumInputPlanes, mish_net ? MISH : RELU,
          true, false, false, 0, use_gemm_ex, use_res_block_winograd_fuse_opt_);
      inputConv->LoadWeights(&weights.input.weights[0],
                             &weights.input.biases[0], scratch_mem_);
      network_.emplace_back(std::move(inputConv));
    } else {
      auto inputConv = std::make_unique<ConvLayer<DataType>>(
          nhwc_, kNumFilters, 8, 8, 3, kNumInputPlanes, mish_net ? MISH : RELU,
          true);
      inputConv->LoadWeights(&weights.input.weights[0],
                             &weights.input.biases[0], scratch_mem_);
      network_.emplace_back(std::move(inputConv));
    }

    // Residual block.
    for (int block = 0; block < numBlocks_; block++) {
      if (use_custom_winograd_) {
        bool has_se = weights.residual[block].has_se;
        int se_k = (int)weights.residual[block].se.b1.size();

        if (use_res_block_winograd_fuse_opt_) {
          auto layer = std::make_unique<ResidualBlock<DataType>>(
              getLastLayer(), kNumFilters, has_se, se_k, use_gemm_ex,
              block == 0, block == (numBlocks_ - 1), mish_net ? MISH : RELU,
              deviceProp.sharedMemPerBlockOptin);
          layer->LoadWeights0(&weights.residual[block].conv1.weights[0],
                              &weights.residual[block].conv1.biases[0],
                              scratch_mem_);
          layer->LoadWeights1(&weights.residual[block].conv2.weights[0],
                              &weights.residual[block].conv2.biases[0],
                              scratch_mem_);
          if (has_se)
            layer->LoadSEWeights(&weights.residual[block].se.w1[0],
                                 &weights.residual[block].se.b1[0],
                                 &weights.residual[block].se.w2[0],
                                 &weights.residual[block].se.b2[0],
                                 scratch_mem_);
          network_.emplace_back(std::move(layer));
        } else {
          auto conv1 = std::make_unique<FusedWinogradConvSELayer<DataType>>(
              getLastLayer(), kNumFilters, 8, 8, kNumFilters,
              mish_net ? MISH : RELU, true, false, false, 0, use_gemm_ex);
          conv1->LoadWeights(&weights.residual[block].conv1.weights[0],
                             &weights.residual[block].conv1.biases[0],
                             scratch_mem_);
          network_.emplace_back(std::move(conv1));

          auto conv2 = std::make_unique<FusedWinogradConvSELayer<DataType>>(
              getLastLayer(), kNumFilters, 8, 8, kNumFilters,
              mish_net ? MISH : RELU, true, true, has_se, se_k, use_gemm_ex);
          conv2->LoadWeights(&weights.residual[block].conv2.weights[0],
                             &weights.residual[block].conv2.biases[0],
                             scratch_mem_);
          if (has_se)
            conv2->LoadSEWeights(&weights.residual[block].se.w1[0],
                                 &weights.residual[block].se.b1[0],
                                 &weights.residual[block].se.w2[0],
                                 &weights.residual[block].se.b2[0],
                                 scratch_mem_);
          network_.emplace_back(std::move(conv2));
        }

      } else {
        auto conv1 = std::make_unique<ConvLayer<DataType>>(
            getLastLayer(), kNumFilters, 8, 8, 3, kNumFilters,
            mish_net ? MISH : RELU, true);
        conv1->LoadWeights(&weights.residual[block].conv1.weights[0],
                           &weights.residual[block].conv1.biases[0],
                           scratch_mem_);
        network_.emplace_back(std::move(conv1));

        // Relu and bias of second convolution is handled by SELayer.
        bool useReluAndBias = weights.residual[block].has_se ? false : true;

        auto conv2 = std::make_unique<ConvLayer<DataType>>(
            getLastLayer(), kNumFilters, 8, 8, 3, kNumFilters,
            useReluAndBias ? (mish_net ? MISH : RELU) : NONE, useReluAndBias);
        conv2->LoadWeights(
            &weights.residual[block].conv2.weights[0],
            useReluAndBias ? &weights.residual[block].conv2.biases[0] : nullptr,
            scratch_mem_);
        network_.emplace_back(std::move(conv2));

        if (weights.residual[block].has_se) {
          int numFCOut = (int)weights.residual[block].se.b1.size();
          auto se = std::make_unique<SELayer<DataType>>(
              getLastLayer(), numFCOut, false, mish_net ? MISH : RELU);
          se->LoadWeights(&weights.residual[block].se.w1[0],
                          &weights.residual[block].se.b1[0],
                          &weights.residual[block].se.w2[0],
                          &weights.residual[block].se.b2[0],
                          &weights.residual[block].conv2.biases[0],
                          scratch_mem_);
          network_.emplace_back(std::move(se));
        }
      }
    }

    resi_last_ = getLastLayer();

    // Policy head.
    if (conv_policy_) {
      auto conv1 = std::make_unique<ConvLayer<DataType>>(
          resi_last_, kNumFilters, 8, 8, 3, kNumFilters, mish_net ? MISH : RELU,
          true);
      conv1->LoadWeights(&weights.policy1.weights[0],
                         &weights.policy1.biases[0], scratch_mem_);
      network_.emplace_back(std::move(conv1));

      auto pol_channels = weights.policy.biases.size();

      // No relu
      auto conv2 = std::make_unique<ConvLayer<DataType>>(
          getLastLayer(), pol_channels, 8, 8, 3, kNumFilters, NONE, true);
      conv2->LoadWeights(&weights.policy.weights[0], &weights.policy.biases[0],
                         scratch_mem_);
      network_.emplace_back(std::move(conv2));

      auto policymap = std::make_unique<PolicyMapLayer<DataType>>(
          getLastLayer(), kNumOutputPolicy, 1, 1, 73 * 8 * 8, false);
      policymap->LoadWeights(kConvPolicyMap, scratch_mem_);

      network_.emplace_back(std::move(policymap));
    } else {
      auto convPol = std::make_unique<ConvLayer<DataType>>(
          resi_last_, weights.policy.biases.size(), 8, 8, 1, kNumFilters,
          mish_net ? MISH : RELU, true);
      convPol->LoadWeights(&weights.policy.weights[0],
                           &weights.policy.biases[0], scratch_mem_);
      network_.emplace_back(std::move(convPol));

      auto FCPol = std::make_unique<FCLayer<DataType>>(
          getLastLayer(), weights.ip_pol_b.size(), 1, 1, true, NONE);
      FCPol->LoadWeights(&weights.ip_pol_w[0], &weights.ip_pol_b[0],
                         scratch_mem_);
      network_.emplace_back(std::move(FCPol));
    }
    policy_out_ = getLastLayer();

    // Value head.
    {
      auto convVal = std::make_unique<ConvLayer<DataType>>(
          resi_last_, weights.value.biases.size(), 8, 8, 1, kNumFilters,
          mish_net ? MISH : RELU, true);
      convVal->LoadWeights(&weights.value.weights[0], &weights.value.biases[0],
                           scratch_mem_);
      network_.emplace_back(std::move(convVal));

      auto FCVal1 = std::make_unique<FCLayer<DataType>>(
          getLastLayer(), weights.ip1_val_b.size(), 1, 1, true,
          mish_net ? MISH : RELU);
      FCVal1->LoadWeights(&weights.ip1_val_w[0], &weights.ip1_val_b[0],
                          scratch_mem_);
      network_.emplace_back(std::move(FCVal1));

      wdl_ = file.format().network_format().value() ==
             pblczero::NetworkFormat::VALUE_WDL;
      auto fc2_tanh = !wdl_;

      auto FCVal2 = std::make_unique<FCLayer<DataType>>(
          getLastLayer(), weights.ip2_val_b.size(), 1, 1, true,
          fc2_tanh ? TANH : NONE);
      FCVal2->LoadWeights(&weights.ip2_val_w[0], &weights.ip2_val_b[0],
                          scratch_mem_);
      network_.emplace_back(std::move(FCVal2));
    }
    value_out_ = getLastLayer();

    // Moves left head
    moves_left_ = (file.format().network_format().moves_left() ==
                   pblczero::NetworkFormat::MOVES_LEFT_V1) &&
                  options.GetOrDefault<bool>("mlh", true);
    if (moves_left_) {
      auto convMov = std::make_unique<ConvLayer<DataType>>(
          resi_last_, weights.moves_left.biases.size(), 8, 8, 1, kNumFilters,
          mish_net ? MISH : RELU, true);
      convMov->LoadWeights(&weights.moves_left.weights[0],
                           &weights.moves_left.biases[0], scratch_mem_);
      network_.emplace_back(std::move(convMov));

      auto FCMov1 = std::make_unique<FCLayer<DataType>>(
          getLastLayer(), weights.ip1_mov_b.size(), 1, 1, true,
          mish_net ? MISH : RELU);
      FCMov1->LoadWeights(&weights.ip1_mov_w[0], &weights.ip1_mov_b[0],
                          scratch_mem_);
      network_.emplace_back(std::move(FCMov1));

      auto FCMov2 = std::make_unique<FCLayer<DataType>>(getLastLayer(), 1, 1, 1,
                                                        true, RELU);
      FCMov2->LoadWeights(&weights.ip2_mov_w[0], &weights.ip2_mov_b[0],
                          scratch_mem_);
      network_.emplace_back(std::move(FCMov2));
    }
    moves_left_out_ = getLastLayer();

    // 3. Allocate GPU memory for running the network:
    //    - three buffers of max size are enough (one to hold input, second to
    //      hold output and third to hold skip connection's input).

    // size of input to the network
    size_t maxSize = max_batch_size_ * kNumInputPlanes * 64 * sizeof(DataType);

    // take max size of all layers
    for (auto& layer : network_) {
      maxSize = std::max(maxSize, layer->GetOutputSize(max_batch_size_));
    }

    // when this optimization is enabled, we write transformed outputs to
    // intermediate tensor memory
    if (use_res_block_winograd_fuse_opt_ && transformed_tensor_size > maxSize)
      maxSize = transformed_tensor_size;

    for (auto& mem : tensor_mem_) {
      ReportCUDAErrors(cudaMalloc(&mem, maxSize));
      ReportCUDAErrors(cudaMemset(mem, 0, maxSize));
    }

    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyTensorDescriptor(xDesc);

#ifdef DEBUG_RAW_NPS
    CERR << "allocated " << 3 * maxSize
         << " bytes of GPU memory to run the network";
#endif
  }

  void forwardEval(InputsOutputs* io, int batchSize) {
    std::unique_lock<std::mutex> lock(lock_);

#ifdef DEBUG_RAW_NPS
    auto t_start = std::chrono::high_resolution_clock::now();
#endif

    // TODO: consider supporting multi-stream path for cudnn backend too.
    cudaStream_t stream = 0;  // default stream

    // Expand packed planes to full planes.
    uint64_t* ipDataMasks = io->input_masks_mem_gpu_;
    float* ipDataValues = io->input_val_mem_gpu_;

    bool fp16 = std::is_same<half, DataType>::value;
    if (fp16) {
      if (nhwc_)
        expandPlanes_Fp16_NHWC((half*)(tensor_mem_[0]), ipDataMasks,
                               ipDataValues, batchSize * kInputPlanes, stream);
      else
        expandPlanes_Fp16_NCHW((half*)(tensor_mem_[0]), ipDataMasks,
                               ipDataValues, batchSize * kInputPlanes, stream);
    } else {
      expandPlanes_Fp32_NCHW((float*)(tensor_mem_[0]), ipDataMasks,
                             ipDataValues, batchSize * kInputPlanes, stream);
    }

    // debug code example
    // dumpTensor(tensor_mem_[0], 1024, "After expand Planes", fp16);

    float* opPol = io->op_policy_mem_gpu_;
    float* opVal = io->op_value_mem_gpu_;
    float* opMov = io->op_moves_left_mem_gpu_;

    int l = 0;
    // Input.
    network_[l++]->Eval(
        batchSize,
        use_res_block_winograd_fuse_opt_ ? tensor_mem_[1] : tensor_mem_[2],
        tensor_mem_[0], nullptr, scratch_mem_, scratch_size_, cudnn_, cublas_,
        stream);  // input conv

    // Residual block.
    for (int block = 0; block < numBlocks_; block++) {
      if (use_res_block_winograd_fuse_opt_) {
        network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[1], nullptr,
                            scratch_mem_, scratch_size_, cudnn_, cublas_,
                            stream);  // block
      } else {
        network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2], nullptr,
                            scratch_mem_, scratch_size_, cudnn_, cublas_,
                            stream);  // conv1

        if (use_custom_winograd_) {
          network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[0],
                              tensor_mem_[2], scratch_mem_, scratch_size_,
                              cudnn_, cublas_, stream);  // conv2
        } else {
          // For SE Resnet, skip connection is added after SE (and bias is added
          // as part of SE).
          if (has_se_) {
            network_[l++]->Eval(batchSize, tensor_mem_[1], tensor_mem_[0],
                                nullptr, scratch_mem_, scratch_size_, cudnn_,
                                cublas_, stream);  // conv2
          } else {
            network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[0],
                                tensor_mem_[2], scratch_mem_, scratch_size_,
                                cudnn_, cublas_, stream);  // conv2
          }

          if (has_se_) {
            network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[1],
                                tensor_mem_[2], scratch_mem_, scratch_size_,
                                cudnn_, cublas_, stream);  // SE layer
          }
        }
      }
    }

    // Policy head.
    if (conv_policy_) {
      network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2], nullptr,
                          scratch_mem_, scratch_size_, cudnn_, cublas_,
                          stream);  // policy conv1

      network_[l++]->Eval(batchSize, tensor_mem_[1], tensor_mem_[0], nullptr,
                          scratch_mem_, scratch_size_, cudnn_, cublas_,
                          stream);  // policy conv2

      if (fp16) {
        network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[1], nullptr,
                            scratch_mem_, scratch_size_, cudnn_, cublas_,
                            stream);  // policy map layer
        copyTypeConverted(opPol, (half*)(tensor_mem_[0]),
                          batchSize * kNumOutputPolicy,
                          stream);  // POLICY output
      } else {
        network_[l++]->Eval(batchSize, (DataType*)opPol, tensor_mem_[1],
                            nullptr, scratch_mem_, scratch_size_, cudnn_,
                            cublas_,
                            stream);  // policy map layer  // POLICY output
      }
    } else {
      network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2], nullptr,
                          scratch_mem_, scratch_size_, cudnn_, cublas_,
                          stream);  // pol conv

      if (fp16) {
        network_[l++]->Eval(batchSize, tensor_mem_[1], tensor_mem_[0], nullptr,
                            scratch_mem_, scratch_size_, cudnn_, cublas_,
                            stream);  // pol FC

        copyTypeConverted(opPol, (half*)(tensor_mem_[1]),
                          batchSize * kNumOutputPolicy, stream);  // POLICY
      } else {
        network_[l++]->Eval(batchSize, (DataType*)opPol, tensor_mem_[0],
                            nullptr, scratch_mem_, scratch_size_, cudnn_,
                            cublas_, stream);  // pol FC  // POLICY
      }
    }

    // Copy policy output from device memory to host memory.
    ReportCUDAErrors(cudaMemcpyAsync(
        io->op_policy_mem_, io->op_policy_mem_gpu_,
        sizeof(float) * kNumOutputPolicy * batchSize, cudaMemcpyDeviceToHost));

    // value head
    network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2], nullptr,
                        scratch_mem_, scratch_size_, cudnn_, cublas_,
                        stream);  // value conv

    network_[l++]->Eval(batchSize, tensor_mem_[1], tensor_mem_[0], nullptr,
                        scratch_mem_, scratch_size_, cudnn_, cublas_,
                        stream);  // value FC1

    if (fp16) {
      // TODO: consider fusing the bias-add of FC2 with format conversion.
      network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[1], nullptr,
                          scratch_mem_, scratch_size_, cudnn_, cublas_,
                          stream);  // value FC2
      copyTypeConverted(opVal, (half*)(tensor_mem_[0]),
                        wdl_ ? 3 * batchSize : batchSize, stream);  // VALUE
    } else {
      network_[l++]->Eval(batchSize, (DataType*)opVal, tensor_mem_[1], nullptr,
                          scratch_mem_, scratch_size_, cudnn_, cublas_,
                          stream);  // value FC2    // VALUE
    }

    if (moves_left_) {
      // Moves left head
      network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2], nullptr,
                          scratch_mem_, scratch_size_, cudnn_, cublas_,
                          stream);  // moves conv

      network_[l++]->Eval(batchSize, tensor_mem_[1], tensor_mem_[0], nullptr,
                          scratch_mem_, scratch_size_, cudnn_, cublas_,
                          stream);  // moves FC1

      // Moves left FC2
      if (fp16) {
        // TODO: consider fusing the bias-add of FC2 with format conversion.
        network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[1], nullptr,
                            scratch_mem_, scratch_size_, cudnn_, cublas_,
                            stream);
        copyTypeConverted(opMov, (half*)(tensor_mem_[0]), batchSize, stream);
      } else {
        network_[l++]->Eval(batchSize, (DataType*)opMov, tensor_mem_[1],
                            nullptr, scratch_mem_, scratch_size_, cudnn_,
                            cublas_, stream);
      }
    }

    ReportCUDAErrors(cudaDeviceSynchronize());
    // The next thread can start using the GPU now.
    lock.unlock();

    if (wdl_) {
      // Value softmax done cpu side.
      for (int i = 0; i < batchSize; i++) {
        float w = io->op_value_mem_[3 * i + 0];
        float d = io->op_value_mem_[3 * i + 1];
        float l = io->op_value_mem_[3 * i + 2];
        float m = std::max({w, d, l});
        w = std::exp(w - m);
        d = std::exp(d - m);
        l = std::exp(l - m);
        float sum = w + d + l;
        w /= sum;
        l /= sum;
        d = 1.0f - w - l;
        io->op_value_mem_[3 * i + 0] = w;
        io->op_value_mem_[3 * i + 1] = d;
        io->op_value_mem_[3 * i + 2] = l;
      }
    }

#ifdef DEBUG_RAW_NPS
    const int reportingCalls = 100;
    static int numCalls = 0;
    static int sumBatchSize = 0;
    static double totalTime = 0;

    sumBatchSize += batchSize;
    numCalls++;

    auto t_end = std::chrono::high_resolution_clock::now();

    double dt = std::chrono::duration<double>(t_end - t_start).count();
    totalTime += dt;
    if (numCalls == reportingCalls) {
      double avgBatchSize = ((double)sumBatchSize) / numCalls;
      double nps = sumBatchSize / totalTime;
      CERR << "Avg batch size: " << avgBatchSize
           << ", NN eval time: " << totalTime << " seconds per " << sumBatchSize
           << " evals. NPS: " << nps;
      sumBatchSize = 0;
      totalTime = 0;
      numCalls = 0;
    }
#endif
  }

  ~CudnnNetwork() {
    for (auto mem : tensor_mem_) {
      if (mem) ReportCUDAErrors(cudaFree(mem));
    }
    if (scratch_mem_) ReportCUDAErrors(cudaFree(scratch_mem_));
    cudnnDestroy(cudnn_);
    cublasDestroy(cublas_);
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    // Set correct gpu id for this computation (as it might have been called
    // from a different thread).
    ReportCUDAErrors(cudaSetDevice(gpu_id_));
    return std::make_unique<CudnnNetworkComputation<DataType>>(this, wdl_,
                                                               moves_left_);
  }

  std::unique_ptr<InputsOutputs> GetInputsOutputs() {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    if (free_inputs_outputs_.empty()) {
      return std::make_unique<InputsOutputs>(max_batch_size_, wdl_,
                                             moves_left_);
    } else {
      std::unique_ptr<InputsOutputs> resource =
          std::move(free_inputs_outputs_.front());
      free_inputs_outputs_.pop_front();
      return resource;
    }
  }

  void ReleaseInputsOutputs(std::unique_ptr<InputsOutputs> resource) {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    free_inputs_outputs_.push_back(std::move(resource));
  }

  // Apparently nvcc doesn't see constructor invocations through make_unique.
  // This function invokes constructor just to please complier and silence
  // warning. Is never called (but compiler thinks that it could).
  void UglyFunctionToSilenceNvccWarning() { InputsOutputs io(0, false, false); }

 private:
  const NetworkCapabilities capabilities_;
  cudnnHandle_t cudnn_;
  cublasHandle_t cublas_;
  int gpu_id_;
  int max_batch_size_;
  bool wdl_;
  bool moves_left_;

  bool nhwc_;  // do we want to use nhwc layout (fastest with fp16 with tensor
               // cores)

  bool use_custom_winograd_;  // Custom winograd convolution implementation for
                              // convolutions of the residual tower.

  bool use_res_block_winograd_fuse_opt_;  // Fuse operations inside the residual
                                          // tower.

  // Currently only one NN Eval can happen a time (we can fix this if needed
  // by allocating more memory).
  mutable std::mutex lock_;

  int numBlocks_;
  bool has_se_;
  bool conv_policy_;
  std::vector<std::unique_ptr<BaseLayer<DataType>>> network_;
  BaseLayer<DataType>* getLastLayer() { return network_.back().get(); }

  BaseLayer<DataType>* resi_last_;
  BaseLayer<DataType>* policy_out_;
  BaseLayer<DataType>* value_out_;
  BaseLayer<DataType>* moves_left_out_;

  DataType* tensor_mem_[3];
  void* scratch_mem_;
  size_t scratch_size_;

  mutable std::mutex inputs_outputs_lock_;
  std::list<std::unique_ptr<InputsOutputs>> free_inputs_outputs_;

  void showInfo() const {
    int version;
    int ret = cudaRuntimeGetVersion(&version);
    switch (ret) {
      case cudaErrorInitializationError:
        throw Exception("CUDA driver and/or runtime could not be initialized");
      case cudaErrorInsufficientDriver:
        throw Exception("No CUDA driver, or one older than the CUDA library");
      case cudaErrorNoDevice:
        throw Exception("No CUDA-capable devices detected");
    }
    int major = version / 1000;
    int minor = (version - major * 1000) / 10;
    int pl = version - major * 1000 - minor * 10;
    CERR << "CUDA Runtime version: " << major << "." << minor << "." << pl;
    if (version != CUDART_VERSION) {
      major = CUDART_VERSION / 1000;
      minor = (CUDART_VERSION - major * 1000) / 10;
      pl = CUDART_VERSION - major * 1000 - minor * 10;
      CERR << "WARNING: CUDA Runtime version mismatch, was compiled with "
              "version "
           << major << "." << minor << "." << pl;
    }
    version = (int)cudnnGetVersion();
    major = version / 1000;
    minor = (version - major * 1000) / 100;
    pl = version - major * 1000 - minor * 100;
    CERR << "Cudnn version: " << major << "." << minor << "." << pl;
    if (version != CUDNN_VERSION) {
      CERR << "WARNING: CUDNN Runtime version mismatch, was compiled with "
              "version "
           << CUDNN_MAJOR << "." << CUDNN_MINOR << "." << CUDNN_PATCHLEVEL;
    }
    cudaDriverGetVersion(&version);
    major = version / 1000;
    minor = (version - major * 1000) / 10;
    pl = version - major * 1000 - minor * 10;
    CERR << "Latest version of CUDA supported by the driver: " << major << "."
         << minor << "." << pl;
    if (version < CUDART_VERSION) {
      CERR << "WARNING: code was compiled with unsupported CUDA version.";
    }
  }

  void showDeviceInfo(const cudaDeviceProp& deviceProp) const {
    CERR << "GPU: " << deviceProp.name;
    CERR << "GPU memory: " << deviceProp.totalGlobalMem / std::pow(2.0f, 30)
         << " GiB";
    CERR << "GPU clock frequency: " << deviceProp.clockRate / 1e3f << " MHz";
    CERR << "GPU compute capability: " << deviceProp.major << "."
         << deviceProp.minor;

    int version = (int)cudnnGetVersion();
    if (version < 7301 && (deviceProp.major > 7 ||
                           (deviceProp.major == 7 && deviceProp.minor >= 5))) {
      CERR << "WARNING: CUDNN version 7.3.1 or newer is better for this GPU.";
    }
    if (std::is_same<float, DataType>::value && deviceProp.major >= 7) {
      CERR << "WARNING: you will probably get better performance from the "
              "cudnn-fp16 backend.";
    }
  }
};

template <typename DataType>
CudnnNetworkComputation<DataType>::CudnnNetworkComputation(
    CudnnNetwork<DataType>* network, bool wdl, bool moves_left)
    : wdl_(wdl), moves_left_(moves_left), network_(network) {
  batch_size_ = 0;
  inputs_outputs_ = network_->GetInputsOutputs();
}

template <typename DataType>
CudnnNetworkComputation<DataType>::~CudnnNetworkComputation() {
  network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
}

template <typename DataType>
void CudnnNetworkComputation<DataType>::ComputeBlocking() {
  network_->forwardEval(inputs_outputs_.get(), GetBatchSize());
}

template <typename DataType>
std::unique_ptr<Network> MakeCudnnNetwork(const std::optional<WeightsFile>& w,
                                          const OptionsDict& options) {
  if (!w) {
    throw Exception(
        "The cudnn" +
        std::string(std::is_same<half, DataType>::value ? "-fp16" : "") +
        " backend requires a network file.");
  }
  const WeightsFile& weights = *w;
  if (weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_CLASSICAL_WITH_HEADFORMAT &&
      weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_SE_WITH_HEADFORMAT) {
    throw Exception(
        "Network format " +
        std::to_string(weights.format().network_format().network()) +
        " is not supported by CuDNN backend.");
  }
  if (weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_CLASSICAL &&
      weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_CONVOLUTION) {
    throw Exception("Policy format " +
                    std::to_string(weights.format().network_format().policy()) +
                    " is not supported by CuDNN backend.");
  }
  if (weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_CLASSICAL &&
      weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_WDL) {
    throw Exception("Value format " +
                    std::to_string(weights.format().network_format().value()) +
                    " is not supported by CuDNN backend.");
  }
  if (weights.format().network_format().moves_left() !=
          pblczero::NetworkFormat::MOVES_LEFT_NONE &&
      weights.format().network_format().moves_left() !=
          pblczero::NetworkFormat::MOVES_LEFT_V1) {
    throw Exception(
        "Movest left head format " +
        std::to_string(weights.format().network_format().moves_left()) +
        " is not supported by CuDNN backend.");
  }
  if (weights.format().network_format().default_activation() !=
          pblczero::NetworkFormat::DEFAULT_ACTIVATION_RELU &&
      weights.format().network_format().default_activation() !=
          pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH) {
    throw Exception(
        "Default activation " +
        std::to_string(weights.format().network_format().default_activation()) +
        " is not supported by CuDNN backend.");
  }
  return std::make_unique<CudnnNetwork<DataType>>(weights, options);
}

std::unique_ptr<Network> MakeCudnnNetworkAuto(
    const std::optional<WeightsFile>& weights, const OptionsDict& options) {
  int gpu_id = options.GetOrDefault<int>("gpu", 0);
  cudaDeviceProp deviceProp = {};
  // No error checking here, this will be repeated later.
  cudaGetDeviceProperties(&deviceProp, gpu_id);

  // Check if the GPU supports FP16.
  if (deviceProp.major >= 7 ||
      (deviceProp.major == 6 && deviceProp.minor != 1) ||
      (deviceProp.major == 5 && deviceProp.minor == 3)) {
    CERR << "Switching to [cudnn-fp16]...";
    return MakeCudnnNetwork<half>(weights, options);
  }
  CERR << "Switching to [cudnn]...";
  return MakeCudnnNetwork<float>(weights, options);
}

REGISTER_NETWORK("cudnn-auto", MakeCudnnNetworkAuto, 120)
REGISTER_NETWORK("cudnn", MakeCudnnNetwork<float>, 110)
REGISTER_NETWORK("cudnn-fp16", MakeCudnnNetwork<half>, 105)

}  // namespace lczero
