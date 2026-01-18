/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2022 The LCZero Authors

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

#include "inputs_outputs.h"
#include "kernels.h"
#include "layers.h"
#include "neural/factory.h"
#include "neural/network_legacy.h"
#include "neural/tables/attention_policy_map.h"
#include "neural/tables/policy_map.h"
#include "rocm_common.h"
#include "utils/bititer.h"
#include "utils/exception.h"

// #define DEBUG_RAW_NPS

namespace lczero {
using namespace rocm_backend;

template <typename DataType>
class RocmNetwork;

static size_t getMaxAttentionHeadSize(
    const MultiHeadWeights::PolicyHead& weights, int N) {
  const size_t embedding_op_size = weights.ip_pol_b.size();
  const size_t policy_d_model = weights.ip2_pol_b.size();
  assert(policy_d_model == weights.ip3_pol_b.size());

  size_t encoder_d_model = 0;
  size_t encoder_dff = 0;

  if (weights.pol_encoder.size() > 0) {
    encoder_d_model = weights.pol_encoder[0].mha.q_b.size();
    encoder_dff = weights.pol_encoder[0].ffn.dense1_b.size();

    assert(encoder_d_model == weights.pol_encoder[0].mha.k_b.size());
    assert(encoder_d_model == weights.pol_encoder[0].mha.v_b.size());
    assert(embedding_op_size == weights.pol_encoder[0].ffn.dense2_b.size());
  }

  const size_t encoder_heads = weights.pol_encoder_head_count;

  size_t size =
      N * 64 *
      std::max(std::max(embedding_op_size, encoder_dff), policy_d_model);

  // size of matmul_qk matrix = encoder_heads_ * Batch * 64 * 64
  const size_t matmul_qk_size = encoder_heads * N * 64 * 64;
  const size_t output_size = N * (64 * 64 + 8 * 24);
  size = std::max(size, std::max(matmul_qk_size, output_size));

  size_t qkv_size = N * 64 * encoder_d_model;
  // We store qkv in single allocation, and other intermediate tensors are
  // sometimes stored by splitting an allocation into two halves.
  size = std::max(2 * size, 3 * qkv_size);
  return size;
}

static size_t getMaxAttentionBodySize(const MultiHeadWeights& weights, int N) {
  const size_t embedding_op_size = weights.ip_emb_b.size();

  size_t encoder_d_model = 0;
  size_t encoder_dff = 0;

  if (weights.encoder.size() > 0) {
    encoder_d_model = weights.encoder[0].mha.q_b.size();
    encoder_dff = weights.encoder[0].ffn.dense1_b.size();

    assert(encoder_d_model == weights.encoder[0].mha.k_b.size());
    assert(encoder_d_model == weights.encoder[0].mha.v_b.size());
    assert(embedding_op_size == weights.encoder[0].ffn.dense2_b.size());
  }

  const size_t encoder_heads = weights.encoder_head_count;

  size_t size =
      N * 64 *
      std::max(std::max(embedding_op_size, encoder_dff), encoder_d_model);

  // size of matmul_qk matrix = encoder_heads_ * Batch * 64 * 64
  const size_t matmul_qk_size = encoder_heads * N * 64 * 64;
  const size_t output_size = N * (64 * 64 + 8 * 24);
  size = std::max(size, std::max(matmul_qk_size, output_size));

  size_t qkv_size = N * 64 * encoder_d_model;
  // We store qkv in single allocation, and other intermediate tensors are
  // sometimes stored by splitting an allocation into two halves.
  size = std::max(2 * size, 3 * qkv_size);
  return size;
}

template <typename DataType>
class RocmNetworkComputation : public NetworkComputation {
 public:
  RocmNetworkComputation(RocmNetwork<DataType>* network, bool wdl,
                         bool moves_left);
  ~RocmNetworkComputation();

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

  RocmNetwork<DataType>* network_;
};

template <typename DataType>
class RocmNetwork : public Network {
 public:
  RocmNetwork(const WeightsFile& file, const OptionsDict& options)
      : capabilities_{file.format().network_format().input(),
                      file.format().network_format().output(),
                      file.format().network_format().moves_left()} {
    CERR << "RocmNetwork constructor starting...";
    CERR << "Loading weights...";
    MultiHeadWeights weights(file.weights());
    CERR << "Weights loaded";
    gpu_id_ = options.GetOrDefault<int>("gpu", 0);

    const auto nf = file.format().network_format();
    using NF = pblczero::NetworkFormat;
    conv_policy_ = nf.policy() == NF::POLICY_CONVOLUTION;
    attn_policy_ = nf.policy() == NF::POLICY_ATTENTION;
    attn_body_ = nf.network() == NF::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT ||
                 nf.network() == NF::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT;

    max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);
    // min_batch_size_ is chosen as 4 as it is common that for sizes less than
    // 4 that there is no performance gain, but there is variance in the
    // outputs, which means that there is extra non-determinism in some
    // scenarios, including using the multiplexing backend.
    min_batch_size_ =
        options.GetOrDefault<int>("min_batch", std::min(4, max_batch_size_));
    if (max_batch_size_ < min_batch_size_)
      throw Exception("Max batch must not be less than min_batch setting.");

    showInfo();
    CERR << "After showInfo";

    int total_gpus;
    ReportHIPErrors(hipGetDeviceCount(&total_gpus));
    CERR << "Total GPUs: " << total_gpus;

    if (gpu_id_ >= total_gpus)
      throw Exception("Invalid GPU Id: " + std::to_string(gpu_id_));

    hipDeviceProp_t deviceProp = {};
    (void)hipGetDeviceProperties(&deviceProp, gpu_id_);
    showDeviceInfo(deviceProp, gpu_id_);
    CERR << "After device info";

    // Architecture-specific batch size optimization for RDNA 3.5 (gfx1151)
    // Based on benchmarking: FP16 peaks at batch 49-73 (1,750-1,800 nps)
    // Only apply if user hasn't explicitly set min_batch
    std::string arch_name(deviceProp.gcnArchName);
    if (arch_name.find("gfx1151") != std::string::npos &&
        !options.Exists<int>("min_batch")) {
      constexpr bool fp16 = std::is_same<half, DataType>::value;
      if (fp16) {
        // FP16: optimal batch 49-73, use 64 as default
        min_batch_size_ = 64;
        CERR << "RDNA 3.5 detected: optimizing min_batch to " << min_batch_size_
             << " for FP16 performance";
      } else {
        // FP32: optimal batch 25+, use 32 as default
        min_batch_size_ = 32;
        CERR << "RDNA 3.5 detected: optimizing min_batch to " << min_batch_size_
             << " for FP32 performance";
      }
    }

    // Select GPU to run on (for *the current* thread).
    ReportHIPErrors(hipSetDevice(gpu_id_));
    CERR << "Device set";

    ReportMIOPENErrors(miopenCreate(&cudnn_));
    CERR << "MIOpen created";
    ReportROCBLASErrors(rocblas_create_handle(&cublas_));
    CERR << "rocBLAS created";

    // Default layout is nchw.
    nhwc_ = false;
    constexpr bool fp16 = std::is_same<half, DataType>::value;

    if (fp16) {
      // Check if the GPU support FP16.

      // For ROCm: Use NCHW (nhwc_=false) for all GPUs
      // Unlike CUDA+CUDNN which uses NHWC with implicit GEMM convolutions,
      // ROCm with rocBLAS (plain GEMM) works best with NCHW layout
      // RDNA 3.5 has WMMA accelerators but they work with NCHW + rocBLAS
      if (arch_name.find("gfx1151") != std::string::npos ||
          arch_name.find("gfx1150") != std::string::npos) {
        // RDNA 3.5 with WMMA - uses NCHW with rocBLAS (nhwc_ remains false)
        CERR << "RDNA 3.5 FP16 detected: using NCHW layout with rocBLAS";
      } else {
        // For other AMD GPUs, default to NCHW (nhwc_ = false already set)
        CERR << "FP16 supported on this GPU";
      }

      // Override if forced from backend option
      if (options.Exists<bool>("nhwc")) nhwc_ = options.Get<bool>("nhwc");
    }

    // ROCm/rocBLAS does not have math mode settings like CUDA
    // Note: RDNA 3.5+ has WMMA (Wave Matrix Multiply Accumulate) accelerators
    // rocBLAS automatically uses WMMA for FP16 operations, providing 8-9x
    // speedup (CUDA code had cublasSetMathMode calls here, but ROCm handles
    // this automatically)

    CERR << "Getting network dimensions...";
    const int kNumInputPlanes = kInputPlanes;
    CERR << "kNumInputPlanes: " << kNumInputPlanes;
    const int kNumFilters = (int)weights.input.biases.size();
    CERR << "kNumFilters: " << kNumFilters;
    numBlocks_ = (int)weights.residual.size();
    CERR << "numBlocks: " << numBlocks_;

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
      // For gfx1151, custom winograd is not beneficial - use rocBLAS GEMM
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
        options.Exists<bool>("custom_winograd");

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
      if (options.Exists<bool>("res_block_fusing")) {
        use_res_block_winograd_fuse_opt_ =
            options.Get<bool>("res_block_fusing");
      }
    }

    const bool use_gemm_ex = deviceProp.major >= 5;

    // 0. Check for SE.
    has_se_ = false;
    if (numBlocks_ && weights.residual[0].has_se) {
      has_se_ = true;
    }

    const bool mish_net = file.format().network_format().default_activation() ==
                          pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH;

    // 1. Allocate scratch space (used internally by cudnn to run convolutions,
    //     and also for format/layout conversion for weights).
    miopenTensorDescriptor_t wDesc;
    miopenConvolutionDescriptor_t convDesc;
    miopenTensorDescriptor_t xDesc;
    // MIOpen uses miopenCreateTensorDescriptor for both filters and tensors
    miopenCreateTensorDescriptor(&wDesc);
    miopenCreateConvolutionDescriptor(&convDesc);
    miopenCreateTensorDescriptor(&xDesc);

    const int maxChannels = std::max(kInputPlanes, kNumFilters);

    const miopenDataType_t datatype = fp16 ? miopenHalf : miopenFloat;
    miopenConvFwdAlgorithm_t conv_algo;
    miopenTensorLayout_t layout = miopenTensorNCHW;

    // MIOpen miopenSet4dTensorDescriptor: (desc, dataType, n, c, h, w)
    ReportMIOPENErrors(miopenSet4dTensorDescriptor(wDesc, datatype, maxChannels,
                                                   maxChannels, 3, 3));

    ReportMIOPENErrors(miopenSet4dTensorDescriptor(
        xDesc, datatype, max_batch_size_, maxChannels, 8, 8));

    // MIOpen convolution descriptor: (desc, mode, pad_h, pad_w, stride_h,
    // stride_w, dilation_h, dilation_w)
    ReportMIOPENErrors(miopenInitConvolutionDescriptor(
        convDesc, miopenConvolution, 1, 1, 1, 1, 1, 1));

    // MIOpen does not have math mode setting - it automatically uses WMMA for
    // FP16 (CUDA code had cudnnSetConvolutionMathType here, but MIOpen handles
    // this automatically)

    if (nhwc_) {
      conv_algo = miopenConvolutionFwdAlgoGEMM;
    } else {
      conv_algo = miopenConvolutionFwdAlgoWinograd;
    }
    (void)conv_algo;  // Suppress unused warning
    (void)layout;     // Suppress unused warning

    // Query expected scratch space from MIOpen.    // MIOpen API: (handle,
    // wDesc, xDesc, convDesc, yDesc, workSpaceSize)
    ReportMIOPENErrors(miopenConvolutionForwardGetWorkSpaceSize(
        cudnn_, wDesc, xDesc, convDesc, xDesc, &scratch_size_));

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

    // Attention policy head may need more memory
    CERR << "Computing attention head size...";
    const size_t attentionSize =
        getMaxAttentionHeadSize(weights.policy_heads.at("vanilla"),
                                max_batch_size_) *
        sizeof(DataType);
    CERR << "Attention head size: " << attentionSize;
    scratch_size_ = std::max(scratch_size_, attentionSize);

    // Attention body also needs scratch space for Q, K, V
    if (attn_body_) {
      const size_t attentionBodyScratchSize =
          getMaxAttentionBodySize(weights, max_batch_size_) * sizeof(DataType);
      scratch_size_ = std::max(scratch_size_, attentionBodyScratchSize);
    }

    ReportHIPErrors(hipMalloc(&scratch_mem_, scratch_size_));

    // 2. Build the network, and copy the weights to GPU memory.

    ActivationFunction act = mish_net ? ACTIVATION_MISH : ACTIVATION_RELU;
    CERR << "Network construction starting, numBlocks=" << numBlocks_
         << ", attn_body=" << attn_body_;

    // Input conv only used if there are residual blocks in the network
    if (numBlocks_ > 0) {
      CERR << "Building input conv layer";
      // Input.
      if (use_custom_winograd_) {
        auto inputConv = std::make_unique<FusedWinogradConvSELayer<DataType>>(
            nullptr, kNumFilters, 8, 8, kNumInputPlanes, act, true, false,
            false, 0, use_gemm_ex, use_res_block_winograd_fuse_opt_);
        inputConv->LoadWeights(&weights.input.weights[0],
                               &weights.input.biases[0], scratch_mem_);
        network_.emplace_back(std::move(inputConv));
      } else {
        auto inputConv = std::make_unique<ConvLayer<DataType>>(
            nhwc_, kNumFilters, 8, 8, 3, kNumInputPlanes, act, true);
        inputConv->LoadWeights(&weights.input.weights[0],
                               &weights.input.biases[0], scratch_mem_);
        network_.emplace_back(std::move(inputConv));
      }
    }

    // Residual block.
    for (int block = 0; block < numBlocks_; block++) {
      if (use_custom_winograd_) {
        bool has_se = weights.residual[block].has_se;
        int se_k = (int)weights.residual[block].se.b1.size();

        if (use_res_block_winograd_fuse_opt_) {
          auto layer = std::make_unique<ResidualBlock<DataType>>(
              getLastLayer(), kNumFilters, has_se, se_k, use_gemm_ex,
              block == 0, block == (numBlocks_ - 1),
              mish_net ? ACTIVATION_MISH : ACTIVATION_RELU,
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
              mish_net ? ACTIVATION_MISH : ACTIVATION_RELU, true, false, false,
              0, use_gemm_ex);
          conv1->LoadWeights(&weights.residual[block].conv1.weights[0],
                             &weights.residual[block].conv1.biases[0],
                             scratch_mem_);
          network_.emplace_back(std::move(conv1));

          auto conv2 = std::make_unique<FusedWinogradConvSELayer<DataType>>(
              getLastLayer(), kNumFilters, 8, 8, kNumFilters,
              mish_net ? ACTIVATION_MISH : ACTIVATION_RELU, true, true, has_se,
              se_k, use_gemm_ex);
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
            mish_net ? ACTIVATION_MISH : ACTIVATION_RELU, true);
        conv1->LoadWeights(&weights.residual[block].conv1.weights[0],
                           &weights.residual[block].conv1.biases[0],
                           scratch_mem_);
        network_.emplace_back(std::move(conv1));

        // Relu and bias of second convolution is handled by SELayer.
        bool useReluAndBias = weights.residual[block].has_se ? false : true;

        auto conv2 = std::make_unique<ConvLayer<DataType>>(
            getLastLayer(), kNumFilters, 8, 8, 3, kNumFilters,
            useReluAndBias ? (mish_net ? ACTIVATION_MISH : ACTIVATION_RELU)
                           : ACTIVATION_NONE,
            useReluAndBias);
        conv2->LoadWeights(
            &weights.residual[block].conv2.weights[0],
            useReluAndBias ? &weights.residual[block].conv2.biases[0] : nullptr,
            scratch_mem_);
        network_.emplace_back(std::move(conv2));

        if (weights.residual[block].has_se) {
          int numFCOut = (int)weights.residual[block].se.b1.size();
          auto se = std::make_unique<SELayer<DataType>>(
              getLastLayer(), numFCOut, false,
              mish_net ? ACTIVATION_MISH : ACTIVATION_RELU);
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

    if (numBlocks_ > 0) {
      resi_last_ = getLastLayer();
    }

    if (attn_body_) {
      CERR << "Building attention body...";
      Activations activations;
      const auto smolgen_activation =
          file.format().network_format().smolgen_activation();
      activations.smolgen_activation =
          smolgen_activation == pblczero::NetworkFormat::ACTIVATION_DEFAULT
              ? act
              : static_cast<ActivationFunction>(smolgen_activation);
      const auto ffn_activation =
          file.format().network_format().ffn_activation();
      activations.ffn_activation =
          ffn_activation == pblczero::NetworkFormat::ACTIVATION_DEFAULT
              ? act
              : static_cast<ActivationFunction>(ffn_activation);
      activations.default_activation = act;

      CERR << "Creating AttentionBody with numBlocks=" << numBlocks_
           << ", input_c=" << (numBlocks_ > 0 ? kNumFilters : kInputPlanes)
           << ", max_batch=" << max_batch_size_;
      auto attention_body = std::make_unique<AttentionBody<DataType>>(
          weights, scratch_mem_, activations, numBlocks_,
          numBlocks_ > 0 ? kNumFilters : kInputPlanes, max_batch_size_,
          static_cast<InputEmbedding>(
              file.format().network_format().input_embedding()) ==
              InputEmbedding::INPUT_EMBEDDING_PE_DENSE,
          use_gemm_ex);
      network_.emplace_back(std::move(attention_body));

      encoder_last_ = getLastLayer();
    }

    // Policy head.
    {
      MultiHeadWeights::PolicyHead& head = weights.policy_heads.at("vanilla");
      if (attn_policy_) {
        auto AttentionPolicy = std::make_unique<AttentionPolicyHead<DataType>>(
            getLastLayer(), head, scratch_mem_, attn_body_, act,
            max_batch_size_, use_gemm_ex);
        network_.emplace_back(std::move(AttentionPolicy));

        auto policymap = std::make_unique<PolicyMapLayer<DataType>>(
            getLastLayer(), kNumOutputPolicy, 1, 1, 64 * 64 + 8 * 24, true);
        policymap->LoadWeights(kAttnPolicyMap, scratch_mem_);
        network_.emplace_back(std::move(policymap));
      } else if (conv_policy_) {
        auto conv1 = std::make_unique<ConvLayer<DataType>>(
            resi_last_, kNumFilters, 8, 8, 3, kNumFilters,
            mish_net ? ACTIVATION_MISH : ACTIVATION_RELU, true);
        conv1->LoadWeights(&head.policy1.weights[0], &head.policy1.biases[0],
                           scratch_mem_);
        network_.emplace_back(std::move(conv1));

        auto pol_channels = head.policy.biases.size();

        // No relu
        auto conv2 = std::make_unique<ConvLayer<DataType>>(
            getLastLayer(), pol_channels, 8, 8, 3, kNumFilters, ACTIVATION_NONE,
            true);
        conv2->LoadWeights(&head.policy.weights[0], &head.policy.biases[0],
                           scratch_mem_);
        network_.emplace_back(std::move(conv2));

        auto policymap = std::make_unique<PolicyMapLayer<DataType>>(
            getLastLayer(), kNumOutputPolicy, 1, 1, 73 * 8 * 8, false);
        policymap->LoadWeights(kConvPolicyMap, scratch_mem_);

        network_.emplace_back(std::move(policymap));
      } else {
        auto convPol = std::make_unique<ConvLayer<DataType>>(
            resi_last_, head.policy.biases.size(), 8, 8, 1, kNumFilters,
            mish_net ? ACTIVATION_MISH : ACTIVATION_RELU, true);
        convPol->LoadWeights(&head.policy.weights[0], &head.policy.biases[0],
                             scratch_mem_);
        network_.emplace_back(std::move(convPol));

        auto FCPol = std::make_unique<FCLayer<DataType>>(
            getLastLayer(), head.ip_pol_b.size(), 1, 1, true, ACTIVATION_NONE);
        FCPol->LoadWeights(&head.ip_pol_w[0], &head.ip_pol_b[0], scratch_mem_);
        network_.emplace_back(std::move(FCPol));
      }
      policy_out_ = getLastLayer();
    }

    // Value head.
    {
      const MultiHeadWeights::ValueHead& head =
          weights.value_heads.at("winner");
      wdl_ = file.format().network_format().value() ==
             pblczero::NetworkFormat::VALUE_WDL;
      BaseLayer<DataType>* lastlayer = attn_body_ ? encoder_last_ : resi_last_;
      auto value_main = std::make_unique<ValueHead<DataType>>(
          lastlayer, head, scratch_mem_, attn_body_, wdl_, act, max_batch_size_,
          use_gemm_ex);
      network_.emplace_back(std::move(value_main));
    }
    value_out_ = getLastLayer();

    // Moves left head
    moves_left_ = (file.format().network_format().moves_left() ==
                   pblczero::NetworkFormat::MOVES_LEFT_V1) &&
                  options.GetOrDefault<bool>("mlh", true);
    if (moves_left_) {
      if (attn_body_) {
        auto embedded_mov = std::make_unique<EmbeddingLayer<DataType>>(
            encoder_last_, weights.ip_mov_w, weights.ip_mov_b, scratch_mem_,
            act);
        network_.emplace_back(std::move(embedded_mov));
      } else {
        auto convMov = std::make_unique<ConvLayer<DataType>>(
            resi_last_, weights.moves_left.biases.size(), 8, 8, 1, kNumFilters,
            mish_net ? ACTIVATION_MISH : ACTIVATION_RELU, true);
        convMov->LoadWeights(&weights.moves_left.weights[0],
                             &weights.moves_left.biases[0], scratch_mem_);
        network_.emplace_back(std::move(convMov));
      }

      auto FCMov1 = std::make_unique<FCLayer<DataType>>(
          getLastLayer(), weights.ip1_mov_b.size(), 1, 1, true,
          mish_net ? ACTIVATION_MISH : ACTIVATION_RELU);
      FCMov1->LoadWeights(&weights.ip1_mov_w[0], &weights.ip1_mov_b[0],
                          scratch_mem_);
      network_.emplace_back(std::move(FCMov1));

      auto FCMov2 = std::make_unique<FCLayer<DataType>>(getLastLayer(), 1, 1, 1,
                                                        true, ACTIVATION_RELU);
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

    if (attn_policy_ && scratch_size_ > maxSize) {
      maxSize = scratch_size_;
    }

    if (attn_body_) {
      const size_t attentionBodySize =
          getMaxAttentionBodySize(weights, max_batch_size_) * sizeof(DataType);
      maxSize = std::max(maxSize, attentionBodySize);
    }

    for (auto& mem : tensor_mem_) {
      ReportHIPErrors(hipMalloc(&mem, maxSize));
    }

    // MIOpen uses miopenDestroyTensorDescriptor for both filters and tensors
    miopenDestroyTensorDescriptor(wDesc);
    miopenDestroyConvolutionDescriptor(convDesc);
    miopenDestroyTensorDescriptor(xDesc);

    // Create events for fine-grained synchronization
    ReportHIPErrors(hipEventCreate(&value_ready_event_));
    ReportHIPErrors(hipEventCreate(&policy_ready_event_));

#ifdef DEBUG_RAW_NPS
    CERR << "allocated " << 3 * maxSize
         << " bytes of GPU memory to run the network";
#endif
  }

  void forwardEval(InputsOutputs* io, int batchSize) {
    // It is safe to evaluate larger than the batchSize
    // as all buffers are designed to handle max_batch_size
    // and the extra invalid results are never read.
    if (batchSize < min_batch_size_) batchSize = min_batch_size_;
    std::unique_lock<std::mutex> lock(lock_);

#ifdef DEBUG_RAW_NPS
    auto t_start = std::chrono::high_resolution_clock::now();
#endif

    // TODO: consider supporting multi-stream path for cudnn backend too.
    hipStream_t stream = 0;  // default stream

    // Expand packed planes to full planes.
    uint64_t* ipDataMasks = io->input_masks_mem_gpu_;
    float* ipDataValues = io->input_val_mem_gpu_;

    bool fp16 = std::is_same<half, DataType>::value;
    if (fp16) {
      // gfx1151 always uses NCHW layout
      expandPlanes_Fp16_NCHW((half*)(tensor_mem_[0]), ipDataMasks, ipDataValues,
                             batchSize * kInputPlanes, stream);
    } else {
      expandPlanes_Fp32_NCHW((float*)(tensor_mem_[0]), ipDataMasks,
                             ipDataValues, batchSize * kInputPlanes, stream);
    }

    float* opPol = io->op_policy_mem_gpu_;
    // Use FP16 buffer for value output if using FP16 backend
    void* opVal =
        fp16 ? (void*)io->op_value_mem_gpu_fp16_ : (void*)io->op_value_mem_gpu_;
    float* opMov = io->op_moves_left_mem_gpu_;

    int l = 0;

    DataType* flow = tensor_mem_[0];
    DataType* spare1 = tensor_mem_[1];
    DataType* spare2 = tensor_mem_[2];

    if (numBlocks_ > 0) {
      // Input.
      network_[l++]->Eval(
          batchSize,
          use_res_block_winograd_fuse_opt_ ? tensor_mem_[1] : tensor_mem_[2],
          tensor_mem_[0], nullptr, scratch_mem_, scratch_size_, cudnn_, cublas_,
          stream);  // input conv

      // Residual block.
      for (int block = 0; block < numBlocks_; block++) {
        if (use_res_block_winograd_fuse_opt_) {
          network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[1],
                              nullptr, scratch_mem_, scratch_size_, cudnn_,
                              cublas_,
                              stream);  // block
        } else {
          network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2],
                              nullptr, scratch_mem_, scratch_size_, cudnn_,
                              cublas_,
                              stream);  // conv1

          if (use_custom_winograd_) {
            network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[0],
                                tensor_mem_[2], scratch_mem_, scratch_size_,
                                cudnn_, cublas_, stream);  // conv2
          } else {
            // For SE Resnet, skip connection is added after SE (and bias is
            // added as part of SE).
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

      flow = tensor_mem_[2];
      spare1 = tensor_mem_[0];
      spare2 = tensor_mem_[1];
    }

    if (attn_body_) {
      network_[l++]->Eval(batchSize, tensor_mem_[1],
                          (numBlocks_ > 0) ? tensor_mem_[2] : tensor_mem_[0],
                          (numBlocks_ > 0) ? tensor_mem_[0] : tensor_mem_[2],
                          scratch_mem_, scratch_size_, nullptr, cublas_, stream,
                          &head_offset_pointers_);  // Entire attention body

      flow = tensor_mem_[1];
      spare1 = tensor_mem_[0];
      spare2 = tensor_mem_[2];
    }

    // Policy head.
    if (attn_policy_) {
      network_[l++]->Eval(
          batchSize, spare1, flow, spare2, scratch_mem_, scratch_size_, nullptr,
          cublas_, stream,
          &head_offset_pointers_);  // Entire Attention policy head except for
                                    // the policy map
      if (fp16) {
        // For FP16: use direct FP32 output from policy map (avoids FP16→FP32
        // conversion)
        auto* policy_map =
            static_cast<PolicyMapLayer<DataType>*>(network_[l++].get());
        policy_map->EvalFp32Output(batchSize, opPol, (const DataType*)spare1,
                                   stream);
      } else {
        network_[l++]->Eval(batchSize, (DataType*)opPol, spare1, nullptr,
                            scratch_mem_, scratch_size_, nullptr, cublas_,
                            stream);  // policy map layer  // POLICY output
      }

    } else if (conv_policy_) {
      network_[l++]->Eval(batchSize, spare1, flow, nullptr, scratch_mem_,
                          scratch_size_, cudnn_, cublas_,
                          stream);  // policy conv1

      network_[l++]->Eval(batchSize, spare2, spare1, nullptr, scratch_mem_,
                          scratch_size_, cudnn_, cublas_,
                          stream);  // policy conv2

      if (fp16) {
        // For FP16: use direct FP32 output from policy map (avoids FP16→FP32
        // conversion)
        auto* policy_map =
            static_cast<PolicyMapLayer<DataType>*>(network_[l++].get());
        policy_map->EvalFp32Output(batchSize, opPol, (const DataType*)spare2,
                                   stream);
      } else {
        network_[l++]->Eval(batchSize, (DataType*)opPol, spare2, nullptr,
                            scratch_mem_, scratch_size_, cudnn_, cublas_,
                            stream);  // policy map layer  // POLICY output
      }
    } else {
      network_[l++]->Eval(batchSize, spare1, flow, nullptr, scratch_mem_,
                          scratch_size_, cudnn_, cublas_, stream);  // pol conv

      if (fp16) {
        // For FP16: use direct FP32 output from policy map (avoids FP16→FP32
        // conversion)
        auto* policy_map =
            static_cast<PolicyMapLayer<DataType>*>(network_[l++].get());
        policy_map->EvalFp32Output(batchSize, opPol, (const DataType*)spare1,
                                   stream);
      } else {
        network_[l++]->Eval(batchSize, (DataType*)opPol, spare1, nullptr,
                            scratch_mem_, scratch_size_, cudnn_, cublas_,
                            stream);  // pol FC  // POLICY
      }
    }

    // value head
    network_[l++]->Eval(batchSize, (DataType*)opVal, flow, spare2, scratch_mem_,
                        scratch_size_, cudnn_, cublas_, stream);  // value head
    // Record event after value computation (needed for CPU softmax)
    ReportHIPErrors(hipEventRecord(value_ready_event_, stream));

    if (moves_left_) {
      // Moves left head
      network_[l++]->Eval(batchSize, spare1, flow, nullptr, scratch_mem_,
                          scratch_size_, cudnn_, cublas_,
                          stream);  // moves conv or embedding

      network_[l++]->Eval(batchSize, spare2, spare1, nullptr, scratch_mem_,
                          scratch_size_, cudnn_, cublas_, stream);  // moves FC1

      // Moves left FC2
      network_[l++]->Eval(batchSize, (DataType*)opMov, spare2, nullptr,
                          scratch_mem_, scratch_size_, cudnn_, cublas_, stream);
    }

    // Defer policy copy until after all GPU work is queued (Optimization #3)
    ReportHIPErrors(hipMemcpyAsync(io->op_policy_mem_, io->op_policy_mem_gpu_,
                                   sizeof(float) * kNumOutputPolicy * batchSize,
                                   hipMemcpyDeviceToHost, stream));
    // Record event after policy transfer
    ReportHIPErrors(hipEventRecord(policy_ready_event_, stream));

    // Wait ONLY for value (needed for CPU softmax)
    ReportHIPErrors(hipEventSynchronize(value_ready_event_));

    // Release lock early (moves_left and policy transfer can finish async)
    lock.unlock();

    if (wdl_) {
      // Value softmax done cpu side.
      for (int i = 0; i < batchSize; i++) {
        float w, d, l;

        // Convert from FP16 to float if using FP16 backend
        if (fp16) {
          w = (float)io->op_value_mem_fp16_[3 * i + 0];
          d = (float)io->op_value_mem_fp16_[3 * i + 1];
          l = (float)io->op_value_mem_fp16_[3 * i + 2];
        } else {
          w = io->op_value_mem_[3 * i + 0];
          d = io->op_value_mem_[3 * i + 1];
          l = io->op_value_mem_[3 * i + 2];
        }
        float m = std::max({w, d, l});
        w = std::exp(w - m);
        d = std::exp(d - m);
        l = std::exp(l - m);
        float sum = w + d + l;
        w /= sum;
        l /= sum;
        d /= sum;
        io->op_value_mem_[3 * i + 0] = w;
        io->op_value_mem_[3 * i + 1] = d;
        io->op_value_mem_[3 * i + 2] = l;
      }
    }

    // Ensure policy output is ready before returning
    ReportHIPErrors(hipEventSynchronize(policy_ready_event_));

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

  ~RocmNetwork() {
    for (auto mem : tensor_mem_) {
      if (mem) ReportHIPErrors(hipFree(mem));
    }
    if (scratch_mem_) ReportHIPErrors(hipFree(scratch_mem_));
    if (head_offset_pointers_) ReportHIPErrors(hipFree(head_offset_pointers_));

    // Destroy synchronization events
    ReportHIPErrors(hipEventDestroy(value_ready_event_));
    ReportHIPErrors(hipEventDestroy(policy_ready_event_));

    miopenDestroy(cudnn_);
    rocblas_destroy_handle(cublas_);
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    // Set correct gpu id for this computation (as it might have been called
    // from a different thread).
    ReportHIPErrors(hipSetDevice(gpu_id_));
    return std::make_unique<RocmNetworkComputation<DataType>>(this, wdl_,
                                                              moves_left_);
  }

  std::unique_ptr<InputsOutputs> GetInputsOutputs() {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    if (free_inputs_outputs_.empty()) {
      constexpr bool fp16 = std::is_same<half, DataType>::value;
      return std::make_unique<InputsOutputs>(max_batch_size_, wdl_, moves_left_,
                                             0, 0, false, fp16);
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
  void UglyFunctionToSilenceNvccWarning() {
    InputsOutputs io(0, false, false, false);
  }

 private:
  const NetworkCapabilities capabilities_;
  miopenHandle_t cudnn_;
  rocblas_handle cublas_;
  int gpu_id_;
  int max_batch_size_;
  int min_batch_size_;
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
  bool attn_policy_;
  bool attn_body_;
  std::vector<std::unique_ptr<BaseLayer<DataType>>> network_;
  BaseLayer<DataType>* getLastLayer() { return network_.back().get(); }

  BaseLayer<DataType>* resi_last_;
  BaseLayer<DataType>* encoder_last_;
  BaseLayer<DataType>* policy_out_;
  BaseLayer<DataType>* value_out_;
  BaseLayer<DataType>* moves_left_out_;

  DataType* tensor_mem_[3];
  void* scratch_mem_;
  DataType** head_offset_pointers_ = nullptr;
  size_t scratch_size_;

  // Events for fine-grained synchronization
  hipEvent_t value_ready_event_;
  hipEvent_t policy_ready_event_;

  mutable std::mutex inputs_outputs_lock_;
  std::list<std::unique_ptr<InputsOutputs>> free_inputs_outputs_;

  void showInfo() const {
    int version;
    int ret = hipRuntimeGetVersion(&version);
    switch (ret) {
      case hipErrorInitializationError:
        throw Exception("CUDA driver and/or runtime could not be initialized");
      case hipErrorInsufficientDriver:
        throw Exception("No CUDA driver, or one older than the CUDA library");
      case hipErrorNoDevice:
        throw Exception("No CUDA-capable devices detected");
    }
    // HIP Runtime version
    int major = version / 10000000;
    int minor = (version / 100000) % 100;
    int pl = version % 100000;
    CERR << "HIP Runtime version: " << major << "." << minor << "." << pl;

    // MIOpen version
    size_t miopen_major, miopen_minor, miopen_patch;
    miopenGetVersion(&miopen_major, &miopen_minor, &miopen_patch);
    CERR << "MIOpen version: " << miopen_major << "." << miopen_minor << "."
         << miopen_patch;

    // ROCm driver version
    (void)hipDriverGetVersion(&version);
    major = version / 10000000;
    minor = (version / 100000) % 100;
    pl = version % 100000;
    CERR << "ROCm driver version: " << major << "." << minor << "." << pl;
  }

  void showDeviceInfo(const hipDeviceProp_t& deviceProp, int deviceId) const {
    (void)deviceId;
    CERR << "GPU: " << deviceProp.name;
    CERR << "GPU memory: " << deviceProp.totalGlobalMem / std::pow(2.0f, 30)
         << " GiB";
    // Get clock rate
    float clockRateMHz = deviceProp.clockRate / 1e3f;
    CERR << "GPU clock frequency: " << clockRateMHz << " MHz";
    CERR << "GPU architecture: " << deviceProp.gcnArchName;

    // Check MIOpen version
    size_t miopen_major, miopen_minor, miopen_patch;
    miopenGetVersion(&miopen_major, &miopen_minor, &miopen_patch);
    CERR << "MIOpen version for this device: " << miopen_major << "."
         << miopen_minor << "." << miopen_patch;
    if (std::is_same<float, DataType>::value && deviceProp.major >= 7) {
      CERR << "WARNING: you will probably get better performance from the "
              "cudnn-fp16 backend.";
    }
  }
};

template <typename DataType>
RocmNetworkComputation<DataType>::RocmNetworkComputation(
    RocmNetwork<DataType>* network, bool wdl, bool moves_left)
    : wdl_(wdl), moves_left_(moves_left), network_(network) {
  batch_size_ = 0;
  inputs_outputs_ = network_->GetInputsOutputs();
}

template <typename DataType>
RocmNetworkComputation<DataType>::~RocmNetworkComputation() {
  network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
}

template <typename DataType>
void RocmNetworkComputation<DataType>::ComputeBlocking() {
  network_->forwardEval(inputs_outputs_.get(), GetBatchSize());
}

template <typename DataType>
std::unique_ptr<Network> MakeRocmNetwork(const std::optional<WeightsFile>& w,
                                         const OptionsDict& options) {
  if (!w) {
    throw Exception(
        "The cudnn" +
        std::string(std::is_same<half, DataType>::value ? "-fp16" : "") +
        " backend requires a network file.");
  }
  const WeightsFile& weights = *w;
  switch (weights.format().network_format().network()) {
    case pblczero::NetworkFormat::NETWORK_CLASSICAL_WITH_HEADFORMAT:
    case pblczero::NetworkFormat::NETWORK_SE_WITH_HEADFORMAT:
      break;
    case pblczero::NetworkFormat::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT:
    case pblczero::NetworkFormat::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT:
      break;
    default:
      throw Exception("Network format " +
                      pblczero::NetworkFormat::NetworkStructure_Name(
                          weights.format().network_format().network()) +
                      " is not supported by HIP backend.");
  }
  if (weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_CLASSICAL &&
      weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_CONVOLUTION &&
      weights.format().network_format().policy() !=
          pblczero::NetworkFormat::POLICY_ATTENTION) {
    throw Exception("Policy format " +
                    pblczero::NetworkFormat::PolicyFormat_Name(
                        weights.format().network_format().policy()) +
                    " is not supported by HIP backend.");
  }
  if (weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_CLASSICAL &&
      weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_WDL) {
    throw Exception("Value format " +
                    pblczero::NetworkFormat::ValueFormat_Name(
                        weights.format().network_format().value()) +
                    " is not supported by HIP backend.");
  }
  if (weights.format().network_format().moves_left() !=
          pblczero::NetworkFormat::MOVES_LEFT_NONE &&
      weights.format().network_format().moves_left() !=
          pblczero::NetworkFormat::MOVES_LEFT_V1) {
    throw Exception("Moves left head format " +
                    pblczero::NetworkFormat::MovesLeftFormat_Name(
                        weights.format().network_format().moves_left()) +
                    " is not supported by HIP backend.");
  }
  if (weights.format().network_format().default_activation() !=
          pblczero::NetworkFormat::DEFAULT_ACTIVATION_RELU &&
      weights.format().network_format().default_activation() !=
          pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH) {
    throw Exception(
        "Default activation " +
        pblczero::NetworkFormat::DefaultActivation_Name(
            weights.format().network_format().default_activation()) +
        " is not supported by HIP backend.");
  }
  return std::make_unique<RocmNetwork<DataType>>(weights, options);
}

std::unique_ptr<Network> MakeRocmNetworkAuto(
    const std::optional<WeightsFile>& weights, const OptionsDict& options) {
  int gpu_id = options.GetOrDefault<int>("gpu", 0);
  hipDeviceProp_t deviceProp = {};
  // No error checking here, this will be repeated later.
  (void)hipGetDeviceProperties(&deviceProp, gpu_id);

  // Check if the GPU supports FP16.
  if (deviceProp.major >= 7 ||
      (deviceProp.major == 6 && deviceProp.minor != 1) ||
      (deviceProp.major == 5 && deviceProp.minor == 3)) {
    CERR << "Switching to [rocm-fp16]...";
    return MakeRocmNetwork<half>(weights, options);
  }
  CERR << "Switching to [rocm]...";
  return MakeRocmNetwork<float>(weights, options);
}

REGISTER_NETWORK("rocm-auto", MakeRocmNetworkAuto, 115)
REGISTER_NETWORK("rocm", MakeRocmNetwork<float>, 110)
REGISTER_NETWORK("rocm-fp16", MakeRocmNetwork<half>, 105)

}  // namespace lczero
