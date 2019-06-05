/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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

#include <dxgi.h>
#include "dx_common.h"
#include "shader_wrapper.h"

namespace lczero {
class DxContext;

namespace dx_backend {
constexpr int kMaxSupportedBatchSize = 256;

// The Layer objects only hold memory for weights, biases, etc
// memory for input and output tensors is provided by caller of Eval.

class BaseLayer {
 public:
  int GetC() const { return C; }
  int GetH() const { return H; }
  int GetW() const { return W; }

  BaseLayer(int c, int h, int w, BaseLayer* ip, DxContext* pContext);
  virtual ~BaseLayer() = default;
  size_t GetOutputSize(int N) const { return sizeof(dx_half) * N * C * H * W; }

  // input2 is optional (skip connection).
  virtual void Eval(int N, dx_alloc_handle output, dx_alloc_handle input,
                    dx_alloc_handle input2, dx_command_stream cmdStream) = 0;

 protected:
  BaseLayer* input_;
  DxContext* dx_context_;

  int C;  // Output tensor dimensions.
  int H;
  int W;
};

// holds Metacommand objects and it's scratch space for all allowed batch sizes
class ConvMetaCommand {
 private:
  // one for every batch size unfortunately!

  // official d3d12 metacommand doesn't support alpha/beta factors :-/ (needed
  // for fused residual add)
  ID3D12MetaCommand* pMetaCommands[kMaxSupportedBatchSize];
  //ID3D12NvMetaCommand* pMetaCommands[kMaxSupportedBatchSize];

  DXAlloc scratch_data_[kMaxSupportedBatchSize];

 public:
  ConvMetaCommand(DxContext* pContext, int Cin,
                  int Cout, int H, int W, int filterSize, bool skipAdd,
                  bool hasBias, bool hasRelu);
  ~ConvMetaCommand();

  ID3D12MetaCommand* getMetaCommand(int N) { return pMetaCommands[N - 1]; }
  D3D12_GPU_DESCRIPTOR_HANDLE getScratchHandle(int N) {
    return scratch_data_[N - 1].descHandle;
  }
};

class ConvLayer : public BaseLayer {
  using BaseLayer::C;
  using BaseLayer::GetC;
  using BaseLayer::GetH;
  using BaseLayer::GetW;
  using BaseLayer::H;
  using BaseLayer::W;

 public:
  ConvLayer(ConvMetaCommand* pMetaCommand, DxContext* pContext,
            BaseLayer* ip, int C, int H, int W,
            int size, int Cin, bool bias, bool relu, bool skipAdd = false);
  ~ConvLayer();

  // returns space in uploadBuffer used for loading weights
  void LoadWeights(float* pfilter, float* pBias, DxContext *pContext);
  void Eval(int N, dx_alloc_handle output, dx_alloc_handle input,
            dx_alloc_handle input2, dx_command_stream cmdStream) override;

 private:
  const int c_input_;
  const int filter_size_;
  const bool use_relu_;
  const bool use_bias_;
  const bool skip_add_;

  dx_alloc_handle biases_;
  dx_alloc_handle weights_;

  ConvMetaCommand* meta_command_;
};

class FCLayer : public BaseLayer {
 public:
  FCLayer(DxContext* pContext, BaseLayer* ip,
          int C, int H, int W, bool bias, bool relu, bool tanh, bool softmax,
          bool fp32Out);
  ~FCLayer();

  // returns space in uploadBuffer used for loading weights
  void LoadWeights(float* cpuWeight, float* cpuBias, DxContext *pContext);
  void Eval(int N, dx_alloc_handle output, dx_alloc_handle input,
            dx_alloc_handle input2, dx_command_stream cmdStream) override;

 private:
  const bool use_bias_;

  // Only one of the below 3 activation functions should be enabled.
  const bool use_relu_;
  const bool use_tanh_;
  const bool use_softmax_;

  dx_alloc_handle biases_;
  dx_alloc_handle weights_;
  ShaderWrapper* shader_wrapper_;
};

}  // namespace dx_backend
}  // namespace lczero
