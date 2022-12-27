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

#include "cuda_common.h"
#include "inputs_outputs.h"
#include "kernels.h"
#include "layers.h"
#include "neural/factory.h"
#include "neural/network_legacy.h"
#include "neural/shared/attention_policy_map.h"
#include "neural/shared/policy_map.h"
#include "utils/bititer.h"
#include "utils/exception.h"

namespace lczero {
using namespace cudnn_backend;

template <typename DataType>
class CudaNetwork;

static size_t getMaxAttentionHeadSize(const LegacyWeights& weights, int N) {
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

  size_t size = N * 64 * std::max(std::max(embedding_op_size, encoder_dff),
                                  policy_d_model);

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

static size_t getMaxAttentionBodySize(const LegacyWeights& weights, int N) {
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
      N * 64 * std::max(embedding_op_size, encoder_d_model);

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
class CudaNetworkComputation : public NetworkComputation {
 public:
  CudaNetworkComputation(CudaNetwork<DataType>* network, bool wdl,
                         bool moves_left);
  ~CudaNetworkComputation();

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

  CudaNetwork<DataType>* network_;
};

template <typename DataType>
class CudaNetwork : public Network {
 public:
  CudaNetwork(const WeightsFile& file, const OptionsDict& options)
      : capabilities_{file.format().network_format().input(),
                      file.format().network_format().moves_left()} {
    LegacyWeights weights(file.weights());
    gpu_id_ = options.GetOrDefault<int>("gpu", 0);

    conv_policy_ = file.format().network_format().policy() ==
                   pblczero::NetworkFormat::POLICY_CONVOLUTION;

    attn_policy_ = file.format().network_format().policy() ==
                   pblczero::NetworkFormat::POLICY_ATTENTION;

    max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);

    showInfo();

    int total_gpus;
    ReportCUDAErrors(cudaGetDeviceCount(&total_gpus));

    if (gpu_id_ >= total_gpus)
      throw Exception("Invalid GPU Id: " + std::to_string(gpu_id_));

    cudaDeviceProp deviceProp = {};
    cudaGetDeviceProperties(&deviceProp, gpu_id_);
    showDeviceInfo(deviceProp);

    l2_cache_size_ = deviceProp.l2CacheSize;

    allow_cache_opt_ = options.GetOrDefault<bool>("cache_opt", true);

    // Select GPU to run on (for *the current* thread).
    ReportCUDAErrors(cudaSetDevice(gpu_id_));

    multi_stream_ = options.GetOrDefault<bool>("multi_stream", false);

    // layout used by cuda backend is nchw.
    has_tensor_cores_ = false;
    constexpr bool fp16 = std::is_same<half, DataType>::value;

    if (fp16) {
      // Check if the GPU support FP16.

      if ((deviceProp.major == 6 && deviceProp.minor != 1) ||
          (deviceProp.major == 5 && deviceProp.minor == 3)) {
        // FP16 without tensor cores supported on GP100 (SM 6.0) and Jetson
        // (SM 5.3 and 6.2). SM 6.1 GPUs also have FP16, but slower than FP32.
        ;
      } else if (deviceProp.major >= 7) {
        // Some GPUs (GTX 16xx) are SM 7.5 but don't have tensor cores
        // enabling TENSOR_OP_MATH for them works but is very very slow
        // (likely because the system emulates it).
        if (!strstr(deviceProp.name, "GTX 16")) {
          has_tensor_cores_ = true;
        }
      } else {
        throw Exception("Your GPU doesn't support FP16");
      }
    }

    if (!multi_stream_) {
      ReportCUBLASErrors(cublasCreate(&cublas_));
      if (has_tensor_cores_)
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
    }

    const int kNumInputPlanes = kInputPlanes;
    const int kNumFilters = (int)weights.input.biases.size();
    numBlocks_ = (int)weights.residual.size();
    numFilters_ = kNumFilters;

    num_encoder_blocks_ = (int) weights.encoder.size();
    attn_body_ = (num_encoder_blocks_ > 0);
    if (attn_body_) {
      assert(weights.ip_emb_b.size() > 0);
    }


    // Warn if the memory required for storing transformed weights is
    // going to exceed 40% of total video memory, force custom_winograd off
    // if it's going to exceed 50% of memory.
    size_t residual_single_layer_weight_size =
        3 * 3 * kNumFilters * kNumFilters * sizeof(DataType);
    size_t residual_weight_size =
        residual_single_layer_weight_size * numBlocks_ * 2;
    size_t transformed_residual_weight_size = residual_weight_size * 4;

    if (transformed_residual_weight_size > 0.4 * deviceProp.totalGlobalMem) {
      CERR << "WARNING: Low GPU video memory. You may run into OOM errors. Try "
              "using a smaller network.";
    }

    // Disable res block fusing for fp32 for now (not worth it)
    // TODO: make it work for filters not a multiple of 32.
    // Note that when used with SE, the optimization
    // works only when filter count is <= 384 (pre-Ampere), or less than 512
    // (Ampere)
    // It turns dynamically off based on filter count (see
    // ResidualBlock<DataType>::Eval)
    if (kNumFilters % 32 == 0 && std::is_same<half, DataType>::value) {
      use_res_block_winograd_fuse_opt_ = true;
    } else {
      use_res_block_winograd_fuse_opt_ = false;
    }
    // Override if set in backend-opts.
    if (!options.IsDefault<bool>("res_block_fusing")) {
      use_res_block_winograd_fuse_opt_ = options.Get<bool>("res_block_fusing");
    }

    const bool use_gemm_ex = deviceProp.major >= 5;

    // 0. Check for SE.
    has_se_ = false;
    if (numBlocks_ && weights.residual[0].has_se) {
      has_se_ = true;
    }

    // Have some minumum as we also use this for transforming weights.
    size_t max_weight_size = 128 * 1024 * 1024;

    // parts from scratch allocation are suballocated to hold various weights
    // and biases when transforming winograd weights (one layer at a time), 128
    // MB is way more than that what we need but make sure it's at least 3x of
    // single layer's weight size to be safe.
    if (max_weight_size < 3 * residual_single_layer_weight_size)
      max_weight_size = 3 * residual_single_layer_weight_size;

    scratch_size_ = max_weight_size;

    // Need additional space for transformed input/outputs which are 36/16
    // times size (4x4 block transformed into 6x6).
    if (numBlocks_ > 0) {
      const size_t transformed_tensor_size =
          (size_t)(max_batch_size_ * kNumFilters * 64 * (36.0 / 16.0) *
                   sizeof(DataType));
      scratch_size_ = std::max(scratch_size_, 2 * transformed_tensor_size);
    }

    // Attention policy head or body may need more memory
    const size_t attentionPolicySize =
        getMaxAttentionHeadSize(weights, max_batch_size_) * sizeof(DataType);

    const size_t attentionBodySize =
        getMaxAttentionBodySize(weights, max_batch_size_) * sizeof(DataType);
    scratch_size_ = std::max(scratch_size_,
                             std::max(attentionPolicySize, attentionBodySize));

    ReportCUDAErrors(cudaMalloc(&scratch_mem_, scratch_size_));

    const bool mish_net = file.format().network_format().default_activation() ==
                          pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH;

    ActivationFunction act = mish_net ? MISH : RELU;

    // 2. Build the network, and copy the weights to GPU memory.

    // Input conv only used if there are residual blocks in the network
    if (numBlocks_ > 0) {
      // Input.
      {
        auto inputConv = std::make_unique<FusedWinogradConvSELayer<DataType>>(
            nullptr, kNumFilters, 8, 8, kNumInputPlanes, act,
            true, false, false, 0, use_gemm_ex,
            use_res_block_winograd_fuse_opt_);
        inputConv->LoadWeights(&weights.input.weights[0],
                               &weights.input.biases[0], scratch_mem_);
        network_.emplace_back(std::move(inputConv));
      }

      // Residual block.
      for (int block = 0; block < numBlocks_; block++) {
        bool has_se = weights.residual[block].has_se;
        int se_k = (int)weights.residual[block].se.b1.size();

        if (use_res_block_winograd_fuse_opt_) {
          auto layer = std::make_unique<ResidualBlock<DataType>>(
              getLastLayer(), kNumFilters, has_se, se_k, use_gemm_ex,
              block == 0, block == (numBlocks_ - 1), act,
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
              getLastLayer(), kNumFilters, 8, 8, kNumFilters, act, true, false,
              false, 0, use_gemm_ex);
          conv1->LoadWeights(&weights.residual[block].conv1.weights[0],
                             &weights.residual[block].conv1.biases[0],
                             scratch_mem_);
          network_.emplace_back(std::move(conv1));

          auto conv2 = std::make_unique<FusedWinogradConvSELayer<DataType>>(
              getLastLayer(), kNumFilters, 8, 8, kNumFilters, act, true, true,
              has_se, se_k, use_gemm_ex);
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
      }
      resi_last_ = getLastLayer();
    }

    if (attn_body_) {
      auto attention_body = std::make_unique<AttentionBody<DataType>>(
          weights, scratch_mem_, act, numBlocks_,
          numBlocks_ > 0 ? kNumFilters : kInputPlanes, max_batch_size_);
      network_.emplace_back(std::move(attention_body));

      encoder_last_ = getLastLayer();
    }

    // Policy head.
    if (attn_policy_) {
      auto AttentionPolicy = std::make_unique<AttentionPolicyHead<DataType>>(
          getLastLayer(), weights, scratch_mem_, attn_body_, act, max_batch_size_);
      network_.emplace_back(std::move(AttentionPolicy));

      auto policymap = std::make_unique<PolicyMapLayer<DataType>>(
          getLastLayer(), kNumOutputPolicy, 1, 1, 64 * 64 + 8 * 24, true);
      policymap->LoadWeights(kAttnPolicyMap, scratch_mem_);
      network_.emplace_back(std::move(policymap));

    } else if (conv_policy_) {
      assert(!attn_body_);  // not supported with attention body
      auto conv1 = std::make_unique<FusedWinogradConvSELayer<DataType>>(
          resi_last_, kNumFilters, 8, 8, kNumFilters, act,
          true, false, false, 0, use_gemm_ex);
      conv1->LoadWeights(&weights.policy1.weights[0],
                         &weights.policy1.biases[0], scratch_mem_);
      network_.emplace_back(std::move(conv1));

      auto pol_channels = weights.policy.biases.size();

      // No relu
      auto conv2 = std::make_unique<FusedWinogradConvSELayer<DataType>>(
          getLastLayer(), pol_channels, 8, 8, kNumFilters, NONE, true, false,
          false, 0, use_gemm_ex);
      conv2->LoadWeights(&weights.policy.weights[0], &weights.policy.biases[0],
                         scratch_mem_);
      network_.emplace_back(std::move(conv2));

      auto policymap = std::make_unique<PolicyMapLayer<DataType>>(
          getLastLayer(), kNumOutputPolicy, 1, 1, 73 * 8 * 8, false);
      policymap->LoadWeights(kConvPolicyMap, scratch_mem_);

      network_.emplace_back(std::move(policymap));
    } else {
      assert(!attn_body_);  // not supported with attention body
      auto convPol = std::make_unique<Conv1Layer<DataType>>(
          resi_last_, weights.policy.biases.size(), 8, 8, kNumFilters, act,
          true, use_gemm_ex);
      convPol->LoadWeights(&weights.policy.weights[0],
                           &weights.policy.biases[0], scratch_mem_);
      network_.emplace_back(std::move(convPol));

      auto FCPol = std::make_unique<FCLayer<DataType>>(
          getLastLayer(), weights.ip_pol_b.size(), 1, 1, true, NONE);
      FCPol->LoadWeights(&weights.ip_pol_w[0], &weights.ip_pol_b[0],
                         scratch_mem_);
      network_.emplace_back(std::move(FCPol));
    }

    // Value head.
    {
      if (attn_body_) {
        auto embedded_val = std::make_unique<EmbeddingLayer<DataType>>(
            encoder_last_, weights.ip_val_w, weights.ip_val_b, scratch_mem_,
            act);
        network_.emplace_back(std::move(embedded_val));
      } else {
        auto convVal = std::make_unique<Conv1Layer<DataType>>(
            resi_last_, weights.value.biases.size(), 8, 8, kNumFilters, act,
            true, use_gemm_ex);
        convVal->LoadWeights(&weights.value.weights[0],
                             &weights.value.biases[0], scratch_mem_);
        network_.emplace_back(std::move(convVal));
      }

      auto FCVal1 = std::make_unique<FCLayer<DataType>>(
          getLastLayer(), weights.ip1_val_b.size(), 1, 1, true, act);
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
        auto convMov = std::make_unique<Conv1Layer<DataType>>(
            resi_last_, weights.moves_left.biases.size(), 8, 8, kNumFilters,
            act, true, use_gemm_ex);
        convMov->LoadWeights(&weights.moves_left.weights[0],
                             &weights.moves_left.biases[0], scratch_mem_);
        network_.emplace_back(std::move(convMov));
      }
      auto FCMov1 = std::make_unique<FCLayer<DataType>>(
          getLastLayer(), weights.ip1_mov_b.size(), 1, 1, true, act);
      FCMov1->LoadWeights(&weights.ip1_mov_w[0], &weights.ip1_mov_b[0],
                          scratch_mem_);
      network_.emplace_back(std::move(FCMov1));

      auto FCMov2 = std::make_unique<FCLayer<DataType>>(getLastLayer(), 1, 1, 1,
                                                        true, RELU);
      FCMov2->LoadWeights(&weights.ip2_mov_w[0], &weights.ip2_mov_b[0],
                          scratch_mem_);
      network_.emplace_back(std::move(FCMov2));
    }

    // 3. Allocate GPU memory for running the network:
    //    - three buffers of max size are enough (one to hold input, second to
    //      hold output and third to hold skip connection's input).

    // size of input to the network
    size_t maxSize = max_batch_size_ * kNumInputPlanes * 64 * sizeof(DataType);

    // take max size of all layers
    for (auto& layer : network_) {
      maxSize = std::max(maxSize, layer->GetOutputSize(max_batch_size_));
    }

    if ((attn_policy_ || use_res_block_winograd_fuse_opt_ || attn_body_) &&
        (scratch_size_ > maxSize)) {
      maxSize = scratch_size_;
    }

    if (!multi_stream_) {
      for (auto& mem : tensor_mem_) {
        ReportCUDAErrors(cudaMalloc(&mem, maxSize));
        ReportCUDAErrors(cudaMemset(mem, 0, maxSize));
      }
    }

    tensor_mem_size_ = multi_stream_ ? maxSize : 0;

    // pre-allocate one InputsOutputs object
    // The first call to allocate memory, create cublas,
    // strem, etc takes really long (600 ms)
    std::unique_ptr<InputsOutputs> io = GetInputsOutputs();
  }

  void forwardEval(InputsOutputs* io, int batchSize) {
    if (!multi_stream_) lock_.lock();

#ifdef DEBUG_RAW_NPS
    auto t_start = std::chrono::high_resolution_clock::now();
#endif

    // Expand packed planes to full planes.
    uint64_t* ipDataMasks = io->input_masks_mem_gpu_;
    float* ipDataValues = io->input_val_mem_gpu_;

    DataType* tensor_mem[3];
    void* scratch_mem;
    cudaStream_t stream;
    cublasHandle_t cublas;
    if (multi_stream_) {
      // We use tensor and scratch memory from InputOutputs (so that multiple
      // requests can run in parallel)
      for (int i = 0; i < 3; i++) tensor_mem[i] = (DataType*)io->tensor_mem_[i];
      scratch_mem = io->scratch_mem_;
      stream = io->stream_;
      cublas = io->cublas_;
    } else {
      for (int i = 0; i < 3; i++) tensor_mem[i] = tensor_mem_[i];
      scratch_mem = scratch_mem_;
      stream = 0;  // default stream
      cublas = cublas_;
    }

    bool fp16 = std::is_same<half, DataType>::value;
    if (fp16) {
      expandPlanes_Fp16_NCHW((half*)(tensor_mem[0]), ipDataMasks, ipDataValues,
                             batchSize * kInputPlanes, stream);
    } else {
      expandPlanes_Fp32_NCHW((float*)(tensor_mem[0]), ipDataMasks, ipDataValues,
                             batchSize * kInputPlanes, stream);
    }

    float* opPol = io->op_policy_mem_gpu_;
    float* opVal = io->op_value_mem_gpu_;
    float* opMov = io->op_moves_left_mem_gpu_;


    // Figure out if the memory requirment for running the res block would fit
    // in the L2 cache.
    bool enableCacheOpt = false;
    DataType* skip_connection =
        use_res_block_winograd_fuse_opt_ ? tensor_mem[1] : tensor_mem[2];

#if CUDART_VERSION >= 11000
    const int pre_transform_tensor_size =
        batchSize * numFilters_ * 8 * 8 * sizeof(DataType);
    const int transformed_tensor_size = pre_transform_tensor_size * 36 / 16;
    const int res_block_mem =
        transformed_tensor_size * 2 + pre_transform_tensor_size;

    cudaStreamAttrValue stream_attribute = {};
    stream_attribute.accessPolicyWindow.base_ptr = tensor_mem[2];
    stream_attribute.accessPolicyWindow.num_bytes = res_block_mem;
    stream_attribute.accessPolicyWindow.hitRatio = 1.0f;
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    if (allow_cache_opt_ && use_res_block_winograd_fuse_opt_ &&
        (res_block_mem <= scratch_size_) && (res_block_mem <= l2_cache_size_)) {
      // we can use a single alloc to hold all the required tensors, and enable
      // persistent L2 caching on it
      ReportCUDAErrors(cudaStreamSetAttribute(
          stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));

      enableCacheOpt = true;
      skip_connection =
          tensor_mem[2] + 2 * transformed_tensor_size / sizeof(DataType);
    }
#endif

    int l = 0;

    DataType* flow = tensor_mem[0];
    DataType* spare1 = tensor_mem[1];
    DataType* spare2 = tensor_mem[2];

    if (numBlocks_ > 0) {
      // Input.
      network_[l++]->Eval(batchSize, skip_connection, tensor_mem[0], nullptr,
                        scratch_mem, scratch_size_, nullptr, cublas,
                        stream);  // input conv

      // Residual block.
      for (int block = 0; block < numBlocks_; block++) {
        if (use_res_block_winograd_fuse_opt_) {
          network_[l++]->Eval(batchSize, tensor_mem[2], skip_connection, nullptr,
                            enableCacheOpt ? nullptr : scratch_mem,
                            scratch_size_, nullptr, cublas,
                            stream);  // block
        } else {
          network_[l++]->Eval(batchSize, tensor_mem[0], tensor_mem[2], nullptr,
                              scratch_mem, scratch_size_, nullptr, cublas,
                              stream);  // conv1

          network_[l++]->Eval(batchSize, tensor_mem[2], tensor_mem[0],
                              tensor_mem[2], scratch_mem, scratch_size_,
                              nullptr, cublas, stream);  // conv2
        }
      }

      flow = tensor_mem[2];
      spare1 = tensor_mem[0];
      spare2 = tensor_mem[1];
    }


    if (attn_body_) {
      network_[l++]->Eval(batchSize, tensor_mem[1],
                          (numBlocks_ > 0) ? tensor_mem[2] : tensor_mem[0],
                          (numBlocks_ > 0) ? tensor_mem[0] : tensor_mem[2],
                          scratch_mem, scratch_size_, nullptr,
                          cublas, stream);  // Entire attention body of the network

      flow = tensor_mem[1];
      spare1 = tensor_mem[0];
      spare2 = tensor_mem[2];
    }

#if CUDART_VERSION >= 11000
    if (enableCacheOpt) {
      // reset the cache settings
      stream_attribute.accessPolicyWindow.num_bytes = 0;
      cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow,
                             &stream_attribute);
      cudaCtxResetPersistingL2Cache();
    }
#endif

    // Policy head.
    if (attn_policy_) {
      network_[l++]->Eval(
          batchSize, spare1, flow, spare2, scratch_mem,
          scratch_size_, nullptr, cublas,
          stream);  // Entire Attention policy head except for the policy map
      if (fp16) {
        network_[l++]->Eval(batchSize, spare2, spare1, nullptr,
                            scratch_mem, scratch_size_, nullptr, cublas,
                            stream);  // policy map layer
        copyTypeConverted(opPol, (half*)spare2,
                          batchSize * kNumOutputPolicy,
                          stream);  // POLICY output
      } else {
        network_[l++]->Eval(batchSize, (DataType*)opPol, spare1, nullptr,
                            scratch_mem, scratch_size_, nullptr, cublas,
                            stream);  // policy map layer  // POLICY output
      }

    } else if (conv_policy_) {
      network_[l++]->Eval(batchSize, spare1, flow, nullptr,
                          scratch_mem, scratch_size_, nullptr, cublas,
                          stream);  // policy conv1

      network_[l++]->Eval(batchSize, spare2, spare1, nullptr,
                          scratch_mem, scratch_size_, nullptr, cublas,
                          stream);  // policy conv2

      if (fp16) {
        network_[l++]->Eval(batchSize, spare1, spare2, nullptr,
                            scratch_mem, scratch_size_, nullptr, cublas,
                            stream);  // policy map layer
        copyTypeConverted(opPol, (half*)(spare1),
                          batchSize * kNumOutputPolicy,
                          stream);  // POLICY output
      } else {
        network_[l++]->Eval(batchSize, (DataType*)opPol, spare2, nullptr,
                            scratch_mem, scratch_size_, nullptr, cublas,
                            stream);  // policy map layer  // POLICY output
      }
    } else {
      network_[l++]->Eval(batchSize, spare1, flow, nullptr,
                          scratch_mem, scratch_size_, nullptr, cublas,
                          stream);  // pol conv

      if (fp16) {
        network_[l++]->Eval(batchSize, spare2, spare1, nullptr,
                            scratch_mem, scratch_size_, nullptr, cublas,
                            stream);  // pol FC

        copyTypeConverted(opPol, (half*)(spare2),
                          batchSize * kNumOutputPolicy, stream);  // POLICY
      } else {
        network_[l++]->Eval(batchSize, (DataType*)opPol, spare1, nullptr,
                            scratch_mem, scratch_size_, nullptr, cublas,
                            stream);  // pol FC  // POLICY
      }
    }

    // Copy policy output from device memory to host memory.
    ReportCUDAErrors(
        cudaMemcpyAsync(io->op_policy_mem_, io->op_policy_mem_gpu_,
                        sizeof(float) * kNumOutputPolicy * batchSize,
                        cudaMemcpyDeviceToHost, stream));

    // value head
    network_[l++]->Eval(batchSize, spare1, flow, nullptr,
                        scratch_mem, scratch_size_, nullptr, cublas,
                        stream);  // value conv or embedding

    network_[l++]->Eval(batchSize, spare2, spare1, nullptr,
                        scratch_mem, scratch_size_, nullptr, cublas,
                        stream);  // value FC1

    if (wdl_) {
      if (fp16) {
        network_[l++]->Eval(batchSize, spare1, spare2, nullptr,
                            scratch_mem, scratch_size_, nullptr, cublas,
                            stream);  // value FC2    // VALUE
        copyTypeConverted(opVal, (half*)spare1, 3 * batchSize,
                          stream);  // VALUE
      } else {
        network_[l++]->Eval(batchSize, (DataType*)opVal, spare2, nullptr,
                            scratch_mem, scratch_size_, nullptr, cublas,
                            stream);  // value FC2    // VALUE
      }
    } else {
      if (fp16) {
        // TODO: consider fusing the bias-add of FC2 with format conversion.
        network_[l++]->Eval(batchSize, spare1, spare2, nullptr,
                            scratch_mem, scratch_size_, nullptr, cublas,
                            stream);  // value FC2
        copyTypeConverted(opVal, (half*)(spare1), batchSize,
                          stream);  // VALUE
      } else {
        network_[l++]->Eval(batchSize, (DataType*)opVal, spare2, nullptr,
                            scratch_mem, scratch_size_, nullptr, cublas,
                            stream);  // value FC2    // VALUE
      }
    }

    if (moves_left_) {
      // Moves left head
      network_[l++]->Eval(batchSize, spare1, flow, nullptr,
                          scratch_mem, scratch_size_, nullptr, cublas,
                          stream);  // moves conv or embedding

      network_[l++]->Eval(batchSize, spare2, spare1, nullptr,
                          scratch_mem, scratch_size_, nullptr, cublas,
                          stream);  // moves FC1

      // Moves left FC2
      if (fp16) {
        // TODO: consider fusing the bias-add of FC2 with format conversion.
        network_[l++]->Eval(batchSize, spare1, spare2, nullptr,
                            scratch_mem, scratch_size_, nullptr, cublas,
                            stream);
        copyTypeConverted(opMov, (half*)(spare1), batchSize, stream);
      } else {
        network_[l++]->Eval(batchSize, (DataType*)opMov, spare2, nullptr,
                            scratch_mem, scratch_size_, nullptr, cublas,
                            stream);
      }
    }

    if (multi_stream_) {
      ReportCUDAErrors(cudaStreamSynchronize(stream));
    } else {
      ReportCUDAErrors(cudaDeviceSynchronize());
      // The next thread can start using the GPU now.
      lock_.unlock();
    }

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
  }

  ~CudaNetwork() {
    if (scratch_mem_) ReportCUDAErrors(cudaFree(scratch_mem_));
    if (!multi_stream_) {
      for (auto mem : tensor_mem_) {
        if (mem) ReportCUDAErrors(cudaFree(mem));
      }
      cublasDestroy(cublas_);
    }
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    // Set correct gpu id for this computation (as it might have been called
    // from a different thread).
    ReportCUDAErrors(cudaSetDevice(gpu_id_));
    return std::make_unique<CudaNetworkComputation<DataType>>(this, wdl_,
                                                              moves_left_);
  }

  std::unique_ptr<InputsOutputs> GetInputsOutputs() {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    if (free_inputs_outputs_.empty()) {
      return std::make_unique<InputsOutputs>(
          max_batch_size_, wdl_, moves_left_, tensor_mem_size_, scratch_size_,
          !has_tensor_cores_ && std::is_same<half, DataType>::value);
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
  int gpu_id_;
  int l2_cache_size_;
  int max_batch_size_;
  bool wdl_;
  bool moves_left_;
  bool use_res_block_winograd_fuse_opt_;  // fuse operations inside the residual
                                          // tower
  bool multi_stream_;                     // run multiple parallel network evals
  bool allow_cache_opt_;                  // try to fit residual block activations in L2 cache

  // Currently only one NN Eval can happen a time (we can fix this if needed
  // by allocating more memory).
  mutable std::mutex lock_;

  int numBlocks_;
  int numFilters_;
  bool has_se_;
  bool conv_policy_;
  bool attn_policy_;
  bool attn_body_;
  int num_encoder_blocks_;
  std::vector<std::unique_ptr<BaseLayer<DataType>>> network_;
  BaseLayer<DataType>* getLastLayer() { return network_.back().get(); }

  BaseLayer<DataType>* resi_last_;
  BaseLayer<DataType>* encoder_last_;

  size_t tensor_mem_size_;
  size_t scratch_size_;

  // this copy is used only for initialization when multi-stream is enabled
  void* scratch_mem_;

  bool has_tensor_cores_;

  // not used when multi-steam is enabled
  cublasHandle_t cublas_;
  DataType* tensor_mem_[3];

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
         << " Gb";
    CERR << "GPU clock frequency: " << deviceProp.clockRate / 1e3f << " MHz";
    CERR << "GPU compute capability: " << deviceProp.major << "."
         << deviceProp.minor;
    CERR << "L2 cache capacity: " << deviceProp.l2CacheSize;
    if (std::is_same<float, DataType>::value && deviceProp.major >= 7) {
      CERR << "WARNING: you will probably get better performance from the "
              "cuda-fp16 backend.";
    }
  }
};

template <typename DataType>
CudaNetworkComputation<DataType>::CudaNetworkComputation(
    CudaNetwork<DataType>* network, bool wdl, bool moves_left)
    : wdl_(wdl), moves_left_(moves_left), network_(network) {
  batch_size_ = 0;
  inputs_outputs_ = network_->GetInputsOutputs();
}

template <typename DataType>
CudaNetworkComputation<DataType>::~CudaNetworkComputation() {
  network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
}

template <typename DataType>
void CudaNetworkComputation<DataType>::ComputeBlocking() {
  network_->forwardEval(inputs_outputs_.get(), GetBatchSize());
}

template <typename DataType>
std::unique_ptr<Network> MakeCudaNetwork(const std::optional<WeightsFile>& w,
                                         const OptionsDict& options) {
  if (!w) {
    throw Exception(
        "The cuda" +
        std::string(std::is_same<half, DataType>::value ? "-fp16" : "") +
        " backend requires a network file.");
  }
  const WeightsFile& weights = *w;
  if (weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_CLASSICAL_WITH_HEADFORMAT &&
      weights.format().network_format().network() !=
          pblczero::NetworkFormat::NETWORK_SE_WITH_HEADFORMAT) {
    throw Exception("Network format " +
                    pblczero::NetworkFormat::NetworkStructure_Name(
                        weights.format().network_format().network()) +
                    " is not supported by the CUDA backend.");
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
                    " is not supported by the CUDA backend.");
  }
  if (weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_CLASSICAL &&
      weights.format().network_format().value() !=
          pblczero::NetworkFormat::VALUE_WDL) {
    throw Exception("Value format " +
                    pblczero::NetworkFormat::ValueFormat_Name(
                        weights.format().network_format().value()) +
                    " is not supported by the CUDA backend.");
  }
  if (weights.format().network_format().moves_left() !=
          pblczero::NetworkFormat::MOVES_LEFT_NONE &&
      weights.format().network_format().moves_left() !=
          pblczero::NetworkFormat::MOVES_LEFT_V1) {
    throw Exception("Moves left head format " +
                    pblczero::NetworkFormat::MovesLeftFormat_Name(
                        weights.format().network_format().moves_left()) +
                    " is not supported by the CUDA backend.");
  }
  if (weights.format().network_format().default_activation() !=
          pblczero::NetworkFormat::DEFAULT_ACTIVATION_RELU &&
      weights.format().network_format().default_activation() !=
          pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH) {
    throw Exception(
        "Default activation " +
        pblczero::NetworkFormat::DefaultActivation_Name(
            weights.format().network_format().default_activation()) +
        " is not supported by the CUDA backend.");
  }
  return std::make_unique<CudaNetwork<DataType>>(weights, options);
}

std::unique_ptr<Network> MakeCudaNetworkAuto(
    const std::optional<WeightsFile>& weights, const OptionsDict& options) {
  int gpu_id = options.GetOrDefault<int>("gpu", 0);
  cudaDeviceProp deviceProp = {};
  // No error checking here, this will be repeated later.
  cudaGetDeviceProperties(&deviceProp, gpu_id);

  // Check if the GPU supports FP16.
  if (deviceProp.major >= 7 ||
      (deviceProp.major == 6 && deviceProp.minor != 1) ||
      (deviceProp.major == 5 && deviceProp.minor == 3)) {
    CERR << "Switching to [cuda-fp16]...";
    return MakeCudaNetwork<half>(weights, options);
  }
  CERR << "Switching to [cuda]...";
  return MakeCudaNetwork<float>(weights, options);
}

REGISTER_NETWORK("cuda-auto", MakeCudaNetworkAuto, 104)
REGISTER_NETWORK("cuda", MakeCudaNetwork<float>, 103)
REGISTER_NETWORK("cuda-fp16", MakeCudaNetwork<half>, 102)

}  // namespace lczero
