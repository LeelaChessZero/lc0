/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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
#pragma once

#include <dxgi.h>

#include <memory>

#include "dx_common.h"
#include "shader_wrapper.h"

namespace lczero {
namespace dx_backend {

class DxContext;
constexpr int kMaxSupportedBatchSize = 256;

// The Layer objects only hold memory for weights, biases, etc
// memory for input and output tensors is provided by caller of Eval.

class BaseLayer {
 public:
  int GetC() const { return C; }
  int GetH() const { return H; }
  int GetW() const { return W; }

  BaseLayer(int c, int h, int w, BaseLayer* ip, DxContext* dx_context,
            bool fp16);
  virtual ~BaseLayer() = default;
  size_t GetOutputSize(int N) const {
    return (fp16_ ? sizeof(dx_half) : sizeof(float)) * N * C * H * W;
  }

  // input2 is optional (skip connection).
  virtual void Eval(int N, DXAlloc output, DXAlloc input, DXAlloc input2,
                    DXAlloc scratch, DXAlloc scratch2,
                    ID3D12GraphicsCommandList4* command_list) = 0;

 protected:
  BaseLayer* input_;
  DxContext* dx_context_;

  bool fp16_;

  // Output tensor dimensions.
  int C;
  int H;
  int W;
};

// Holds Metacommand objects and their scratch space for all allowed batch
// sizes.
class GemmMetaCommand {
 private:
  // Need to create a Metacommand object for each batch size unfortunately!
  // Some hw vendors don't support arbitary sizes anyway, so we create only
  // multiples of 8 in no. of rows (when M is 0).

  static constexpr int kMetacommandGranulity = 8;
  static constexpr int kMaxMetacommands =
      (kMaxSupportedBatchSize * 4) / kMetacommandGranulity;
  ID3D12MetaCommand* meta_commands_[kMaxMetacommands];

  DXAlloc scratch_data_persistent_[kMaxMetacommands];
  DXAlloc scratch_data_temporary_[kMaxMetacommands];

  bool rows_known_;
  bool create_succeeded_;

 public:
  GemmMetaCommand(DxContext* dx_context, int M, int N, int K, int gemm_batch,
                  bool fp16, bool a_transpose, bool b_transpose);
  ~GemmMetaCommand();

  void PerformGemm(int rows, DXAlloc A, DXAlloc B, DXAlloc Output,
                   ID3D12GraphicsCommandList4* command_list);

  bool IsAvailable() { return create_succeeded_; }
};

class ConvMetaCommand {
 private:
  // Metacommand objects for each multiple of 8 batch size
  static constexpr int kMetacommandGranulity = 8;
  static constexpr int kMaxMetacommands =
      kMaxSupportedBatchSize / kMetacommandGranulity;
  ID3D12MetaCommand* meta_commands_[kMaxMetacommands];

  DXAlloc scratch_data_persistent_[kMaxMetacommands];
  DXAlloc scratch_data_temporary_[kMaxMetacommands];
  bool create_succeeded_;
  bool use_bias_;

 public:
  ConvMetaCommand(DxContext* dx_context, int C, int K, int H, int W, int F,
                  bool relu, bool bias, bool fp16);
  ~ConvMetaCommand();

  void PerformConv(int batch, DXAlloc input, DXAlloc filter, DXAlloc bias,
                   DXAlloc output, ID3D12GraphicsCommandList4* command_list);

  bool IsAvailable() { return create_succeeded_; }
};

class ConvLayer : public BaseLayer {
  using BaseLayer::C;
  using BaseLayer::GetC;
  using BaseLayer::GetH;
  using BaseLayer::GetW;
  using BaseLayer::H;
  using BaseLayer::W;

 public:
  ConvLayer(bool fp16, GemmMetaCommand* meta_command_gemm,
            ConvMetaCommand* meta_command_conv, DxContext* dx_context,
            BaseLayer* ip, int C, int H, int W, int size, int Cin, bool bias,
            bool relu, bool skipAdd = false, bool se = false, int se_k = 0);
  ~ConvLayer();

  // returns space in uploadBuffer used for loading weights
  void LoadWeights(float* filter, float* bias, DxContext* dx_context);
  void LoadSEWeights(float* w1, float* b1, float* w2, float* b2);
  void Eval(int N, DXAlloc output, DXAlloc input, DXAlloc input2,
            DXAlloc scratch, DXAlloc scratch2,
            ID3D12GraphicsCommandList4* command_list) override;

 private:
  const int c_input_;
  const int filter_size_;
  const bool use_relu_;
  const bool use_bias_;
  const bool skip_add_;
  const bool has_se_;
  const int se_k_;

  DXAlloc biases_;
  DXAlloc weights_;
  DXAlloc transformed_weights_;  // After winograd transform.

  // Weights and Biases for (optional) SE.
  DXAlloc w1_;
  DXAlloc w2_;
  DXAlloc b1_;
  DXAlloc b2_;

  ShaderWrapper* shader_wrapper_;
  GemmMetaCommand* meta_command_gemm_;
  ConvMetaCommand* meta_command_conv_;
};

class FCLayer : public BaseLayer {
 public:
  FCLayer(bool fp16, DxContext* dx_context, BaseLayer* ip, int C, int H, int W,
          bool bias, bool relu, bool tanh);
  ~FCLayer();

  // returns space in uploadBuffer used for loading weights
  void LoadWeights(float* cpu_weight, float* cpu_bias, DxContext* dx_context);
  void Eval(int N, DXAlloc output, DXAlloc input, DXAlloc input2,
            DXAlloc scratch, DXAlloc scratch2,
            ID3D12GraphicsCommandList4* command_list) override;

 private:
  const bool use_bias_;

  // Only one of the below 2 activation functions should be enabled.
  const bool use_relu_;
  const bool use_tanh_;

  DXAlloc biases_;
  DXAlloc weights_;
  ShaderWrapper* shader_wrapper_;
  std::unique_ptr<GemmMetaCommand> meta_command_;
};

class PolicyMapLayer : public BaseLayer {
 public:
  PolicyMapLayer(bool fp16, DxContext* dx_context, BaseLayer* ip, int C, int H,
                 int W, int used_size);
  ~PolicyMapLayer();
  void LoadWeights(const short* cpu_weights);
  void Eval(int N, DXAlloc output, DXAlloc input, DXAlloc input2,
            DXAlloc scratch, DXAlloc scratch2,
            ID3D12GraphicsCommandList4* command_list) override;

 private:
  const int used_size_;
  DXAlloc weights_;
};

}  // namespace dx_backend
}  // namespace lczero
