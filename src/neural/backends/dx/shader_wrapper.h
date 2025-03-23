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
  // 3. Winograd Output transform.
  //      - Fused with bias add, skip connection add, and relu.
  //      - Fused with SE, bias add, skip connection add and relu.
  // 4. Policy Map layer. (can also be done on CPU side)
  // 5. 1x1 convolution custom kernel (used by policy and value heads).
  //      - TODO: Try replacing this with conv metacommand when available.
  //
  // For best performance it would seem that we would need two copies of all the
  // shaders - fp16 and fp32 versions. However 2 copies not needed for now, as:
  //   i) We should use typed UAVs for resource access as they seem to be
  //   faster. With Typed UAVs, the shader automatically recieves datatype converted
  //   values (e.g: in fp32 even when the allocation was in fp16)
  //   ii) Most of these operations are memory bound - except for the GEMM, but:
  //  iii) Due to driver/compiler bugs or lack of optimizations fp16 path seems
  //  slower on both Nvidia and AMD for most of the shaders - even GEMM on AMD
  //  is way slower with fp16 math than with fp32.

  // Only expand planes has different shaders for different datatypes.
  //  - Mostly a meaningless 'early' optimization as this shouldn't be the bottleneck.
  ID3D12PipelineState* expand_planes_fp16_;
  ID3D12PipelineState* expand_planes_fp32_;

  ID3D12PipelineState* winograd_input_transform_;
  ID3D12PipelineState* winograd_output_transform_;
  ID3D12PipelineState* conv_1x1_;
  ID3D12PipelineState* policy_map_;

  // Gemm shaders (used when gemm Metacommand isn't supported by the HW vendor)
  ID3D12PipelineState* gemm_;
 
  // Another simple shader to add bias, apply relu/tanh, etc.
  ID3D12PipelineState* add_vectors_;

  // Fused SE shaders for various standard channel counts.
  ID3D12PipelineState* se_128_;
  ID3D12PipelineState* se_256_;
  ID3D12PipelineState* se_320_;
  ID3D12PipelineState* se_384_;
  ID3D12PipelineState* se_512_;
  ID3D12PipelineState* se_640_;
  ID3D12PipelineState* se_768_;
  ID3D12PipelineState* se_1024_;

  // Winograd output transform fused with SE for various standard channel
  // counts.
  ID3D12PipelineState* winograd_output_transform_se_128_;
  ID3D12PipelineState* winograd_output_transform_se_256_;
  ID3D12PipelineState* winograd_output_transform_se_320_;
  ID3D12PipelineState* winograd_output_transform_se_384_;
  ID3D12PipelineState* winograd_output_transform_se_512_;
  ID3D12PipelineState* winograd_output_transform_se_640_;
  ID3D12PipelineState* winograd_output_transform_se_768_;
  ID3D12PipelineState* winograd_output_transform_se_1024_;

 public:
  void Init(ID3D12Device* pDevice);
  void Destroy();

  void ExpandPlanes(ID3D12GraphicsCommandList4* command_list,
                    DXAlloc output_tensor, DXAlloc masks, DXAlloc values,
                    int batchSize, bool fp16);

  void InputTransform(ID3D12GraphicsCommandList4* command_list,
                      DXAlloc transformed_input, DXAlloc input, int N, int C,
                      bool fp16);

  void OutputTransform(ID3D12GraphicsCommandList4* command_list, DXAlloc output,
                       DXAlloc transformed_output, DXAlloc skip_connection,
                       DXAlloc bias, DXAlloc se_w1, DXAlloc se_b1,
                       DXAlloc se_w2, DXAlloc se_b2, int N, int K, bool relu,
                       bool bias_add, bool skip_add, bool fused_se, int se_k,
                       bool fp16);

  void Se(ID3D12GraphicsCommandList4* command_list, DXAlloc output,
          DXAlloc input, DXAlloc skip_connection, DXAlloc bias, DXAlloc se_w1,
          DXAlloc se_b1, DXAlloc se_w2, DXAlloc se_b2, int N, int K, bool relu,
          bool bias_add, bool skip_add, int se_k, bool fp16);

  void Conv1x1(ID3D12GraphicsCommandList4* command_list, DXAlloc output,
               DXAlloc input, DXAlloc weight, DXAlloc bias, int N, int C, int K,
               bool relu, bool useBias, bool fp16);

  void AddVectors(ID3D12GraphicsCommandList4* command_list, DXAlloc C,
                  DXAlloc A, DXAlloc B, int c_size, int b_size, int a_size,
                  bool relu, bool tanh, bool fp16);

  void PolicyMap(ID3D12GraphicsCommandList4* command_list, DXAlloc output,
                 DXAlloc input, DXAlloc weights, int N, int input_size,
                 int output_size, int used_size, bool fp16);

  void MatrixMultiply(ID3D12GraphicsCommandList4* command_list, DXAlloc output,
                      DXAlloc A, DXAlloc B, int M, int N, int K, int batch,
                      bool fp16);
};

}  // namespace dx_backend
}  // namespace lczero
