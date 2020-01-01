/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2019 The LCZero Authors

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
*/

#pragma once
#include "dx_common.h"

namespace lczero {
namespace dx_backend {

class ShaderWrapper {
 private:
  ID3D12RootSignature* root_sign_;
  static constexpr int kUavSlots = 8;

  // Various shaders used by the backend:
  //
  // 1. Expand planes: Used to convert packed bit-board representation to
  //                   'planes' that is input to NN
  // 2. Winograd Input transform.
  // 3. Winograd Output transform (compile seperate versions only if better for perf):
  //      - Without anything.
  //      - Just optional relu.
  //      - Fused with skip connection add and relu.
  //      - Fused with SE and skip connection add and relu?
  // 4. Policy Map layer. (can also be done on CPU side)
  // 5. Fully connected softmax when using wdl. (maybe can be done on CPU side too)
  // 6. 1x1 convolution custom kernel (used by policy and value heads).
  // (We have a conv metacommand, but dealing with nhwc/nchw memory layouts for
  //  different vendors/datatypes is probably harder than just writing a kernel ourselves
  //  which shouldn't be the bottleneck anyway).

  //
  // Two copies of all of the above as we need both fp16 and fp32 versions.

  ID3D12PipelineState* expand_planes_state_fp16_;
  ID3D12PipelineState* winograd_input_transform_fp16_;
  ID3D12PipelineState* winograd_output_transform_fp16_;
  ID3D12PipelineState* conv_1x1_fp16_;

  ID3D12PipelineState* expand_planes_state_fp32_;
  ID3D12PipelineState* winograd_input_transform_fp32_;
  ID3D12PipelineState* winograd_output_transform_fp32_;
  ID3D12PipelineState* conv_1x1_fp32_;

  // Another simple shader (same shaders handles both fp32 and fp16) to add
  // bias, apply relu/tanh, etc.
  ID3D12PipelineState* add_vectors_;

  // Fused SE shaders for various standard channel counts
  ID3D12PipelineState* winograd_output_transform_fp16_se_128_;
  ID3D12PipelineState* winograd_output_transform_fp16_se_256_;
  ID3D12PipelineState* winograd_output_transform_fp16_se_320_;
  ID3D12PipelineState* winograd_output_transform_fp16_se_384_;
  ID3D12PipelineState* winograd_output_transform_fp16_se_512_;
  ID3D12PipelineState* winograd_output_transform_fp16_se_640_;
  ID3D12PipelineState* winograd_output_transform_fp16_se_768_;
  ID3D12PipelineState* winograd_output_transform_fp16_se_1024_;

  ID3D12PipelineState* winograd_output_transform_fp32_se_128_;
  ID3D12PipelineState* winograd_output_transform_fp32_se_256_;
  ID3D12PipelineState* winograd_output_transform_fp32_se_320_;
  ID3D12PipelineState* winograd_output_transform_fp32_se_384_;
  ID3D12PipelineState* winograd_output_transform_fp32_se_512_;
  ID3D12PipelineState* winograd_output_transform_fp32_se_640_;
  ID3D12PipelineState* winograd_output_transform_fp32_se_768_;
  ID3D12PipelineState* winograd_output_transform_fp32_se_1024_;

 public:
  void init(ID3D12Device* pDevice);
  void destroy();

  void expandPlanes(ID3D12GraphicsCommandList5* command_list,
                    DXAlloc output_tensor, DXAlloc masks, DXAlloc values,
                    int batchSize, bool fp16);

  void inputTransform(ID3D12GraphicsCommandList5* command_list,
                      DXAlloc transformed_input, DXAlloc input, int N, int C,
                      bool fp16);

  void outputTransform(ID3D12GraphicsCommandList5* command_list, DXAlloc output,
                       DXAlloc transformed_output, DXAlloc skip_connection,
                       DXAlloc bias, DXAlloc se_w1, DXAlloc se_b1,
                       DXAlloc se_w2, DXAlloc se_b2, int N, int K, bool relu,
                       bool bias_add, bool skip_add, bool fused_se, int se_k,
                       bool fp16);

  void conv1x1(ID3D12GraphicsCommandList5* command_list, DXAlloc output,
               DXAlloc input, DXAlloc weight, DXAlloc bias, int N, int C, int K,
               bool relu, bool useBias, bool fp16);

  void addVectors(ID3D12GraphicsCommandList5* command_list, DXAlloc C,
                  DXAlloc A, DXAlloc B, int c_size, int b_size, int a_size,
                  bool relu, bool tanh, bool fp16);
};

}  // namespace dx_backend
}  // namespace lczero
