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

#pragma once
#include "dx_common.h"

namespace lczero {
namespace dx_backend {

class ShaderWrapper {
 private:
  ID3D12RootSignature* root_sign_;

  ID3D12PipelineState* expand_planes_state_;
  ID3D12PipelineState* policy_fc_state_;
  ID3D12PipelineState* value_fc1_state_;
  ID3D12PipelineState* value_fc2_state_;
  ID3D12PipelineState* policy_softmax_state_;
  ID3D12PipelineState* skip_add_state_;

 public:
  void init(ID3D12Device* pDevice);
  void destroy();
  void expandPlanes(dx_command_stream stream, DXAlloc opTensor, DXAlloc masks,
                    DXAlloc values, int batchSize);
  void policyFC(dx_command_stream stream, DXAlloc output, DXAlloc input,
                DXAlloc weights, DXAlloc biases, int batchSize);

  void valueFC1(dx_command_stream stream, DXAlloc output, DXAlloc input,
                DXAlloc weights, DXAlloc biases, int batchSize);

  void valueFC2(dx_command_stream stream, DXAlloc output, DXAlloc input,
                DXAlloc weights, DXAlloc biases, int batchSize);

  void skipAddRelu(dx_command_stream stream, DXAlloc input1, DXAlloc input2,
                   DXAlloc output, bool performRelu, int numElements);
};

}  // namespace dx_backend
}  // namespace lczero
