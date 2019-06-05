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
#include "shader_wrapper.h"
#include <cassert>
#include <cstring>
#include "comdef.h"
#include "neural/network.h"
#include "shaders/shaders.h"

// Use single shader for policy FC and softmax.
// With optimized Metacommand path (whenever it's available) the non-fused path
// should be way faster.
// the non-fused version has a few bugs too!
#define FUSED_POLICY 1

namespace lczero {
namespace dx_backend {

void ShaderWrapper::init(ID3D12Device* pDevice) {
  // 1. Create root signature - common for all shaders

  // 5 slots
  // slot 0 to 3 -> root UAV slots 0 to 3 (all in space 0)
  // slot 4      -> root constants (16 constants - should be enough)

  D3D12_ROOT_PARAMETER rootParameter[5];
  for (int i = 0; i < 4; i++) {
    rootParameter[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rootParameter[i].Descriptor.RegisterSpace = 0;
    rootParameter[i].Descriptor.ShaderRegister = i;
    rootParameter[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }

  rootParameter[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  rootParameter[4].Constants.RegisterSpace = 0;
  rootParameter[4].Constants.ShaderRegister = 0;
  rootParameter[4].Constants.Num32BitValues = 16;
  rootParameter[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

  D3D12_ROOT_SIGNATURE_DESC rootSigDesc = {5, rootParameter, 0, NULL,
                                           D3D12_ROOT_SIGNATURE_FLAG_NONE};

  ID3DBlob* pSerializedLayout = NULL;
  D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
                              &pSerializedLayout, NULL);

  ReportDxErrors(pDevice->CreateRootSignature(
      1, pSerializedLayout->GetBufferPointer(),
      pSerializedLayout->GetBufferSize(), IID_PPV_ARGS(&root_sign_)));

  pSerializedLayout->Release();

  // Create PSO objects for each shader
  // PSO basically holds the compiled shader object (and other state which we
  // don't use)

  // 2. PSO for the expand planes shader
  D3D12_COMPUTE_PIPELINE_STATE_DESC stateDesc = {};
  stateDesc.CS = {g_ExpandPlanes_kernel_Fp16_NHWC,
                  sizeof(g_ExpandPlanes_kernel_Fp16_NHWC)};
  stateDesc.pRootSignature = root_sign_;
  ReportDxErrors(pDevice->CreateComputePipelineState(
      &stateDesc, IID_PPV_ARGS(&expand_planes_state_)));

  // 3. PSO for policy FC (+softmax) layer
#if FUSED_POLICY == 1
  stateDesc.CS = {g_PolicyFC_With_Softmax_kernel,
                  sizeof(g_PolicyFC_With_Softmax_kernel)};
#else
  stateDesc.CS = {g_PolicyFC, sizeof(g_PolicyFC)};
#endif
  ReportDxErrors(pDevice->CreateComputePipelineState(
      &stateDesc, IID_PPV_ARGS(&policy_fc_state_)));

  stateDesc.CS = {g_PolicySoftmax, sizeof(g_PolicySoftmax)};
  ReportDxErrors(pDevice->CreateComputePipelineState(
      &stateDesc, IID_PPV_ARGS(&policy_softmax_state_)));

  // 4. PSO for value FC layers
  stateDesc.CS = {g_ValueFC1, sizeof(g_ValueFC1)};
  ReportDxErrors(pDevice->CreateComputePipelineState(
      &stateDesc, IID_PPV_ARGS(&value_fc1_state_)));

  stateDesc.CS = {g_ValueFC2, sizeof(g_ValueFC2)};
  ReportDxErrors(pDevice->CreateComputePipelineState(
      &stateDesc, IID_PPV_ARGS(&value_fc2_state_)));

  // 5. PSO for skip connection add/relu operation
  stateDesc.CS = {g_SkipAdd, sizeof(g_SkipAdd)};  // ANKAN : TODO!
  ReportDxErrors(pDevice->CreateComputePipelineState(
      &stateDesc, IID_PPV_ARGS(&skip_add_state_)));
}

void ShaderWrapper::destroy() {
  policy_fc_state_->Release();
  expand_planes_state_->Release();
  value_fc1_state_->Release();
  value_fc2_state_->Release();
  skip_add_state_->Release();
  policy_softmax_state_->Release();

  root_sign_->Release();
}

void ShaderWrapper::expandPlanes(dx_command_stream stream, DXAlloc opTensor,
                                 DXAlloc masks, DXAlloc values, int batchSize) {
  const int N = batchSize * kInputPlanes;
  int Consts[] = {N, kInputPlanes};
  stream->SetComputeRootSignature(root_sign_);
  stream->SetPipelineState(expand_planes_state_);
  stream->SetComputeRootUnorderedAccessView(0, opTensor.gpuVA);
  stream->SetComputeRootUnorderedAccessView(1, masks.gpuVA);
  stream->SetComputeRootUnorderedAccessView(2, values.gpuVA);
  stream->SetComputeRoot32BitConstants(4, 2, &Consts, 0);

  // Each thread writes two elements
  int threads = N * 8 * 8 / 2;
  const int kBlockSize = 256;
  int blocks = DivUp(threads, kBlockSize);
  stream->Dispatch(blocks, 1, 1);
}

// TODO: really need to switch to matrix multiply metacommand when it's
// available although the fully connected are very small, our hand written
// shaders are relatively very slow.
void ShaderWrapper::policyFC(dx_command_stream stream, DXAlloc output,
                             DXAlloc input, DXAlloc weights, DXAlloc biases,
                             int batchSize) {
  int Consts[] = {batchSize};
  stream->SetComputeRootSignature(root_sign_);
  stream->SetPipelineState(policy_fc_state_);
  stream->SetComputeRootUnorderedAccessView(0, output.gpuVA);
  stream->SetComputeRootUnorderedAccessView(1, input.gpuVA);
  stream->SetComputeRootUnorderedAccessView(2, weights.gpuVA);
  stream->SetComputeRootUnorderedAccessView(3, biases.gpuVA);
  stream->SetComputeRoot32BitConstants(4, 1, &Consts, 0);

#if FUSED_POLICY == 1
  // Each thread writes two elements
  // block size is kNumOutputPolicy/2
  // gird size is 'batchSize' no of blocks
  stream->Dispatch(batchSize, 1, 1);
#else
// TODO : move these to a common include file
#define blockWidth 16
#define blockHeight 2

#define elementsPerThreadX 4
#define elementsPerThreadY 4

#define elementsPerBlockX (blockWidth * elementsPerThreadX)
#define elementsPerBlockY (blockHeight * elementsPerThreadY)

  int blocksX = DivUp(1858, elementsPerBlockX);  // TODO: remove hardcoding
  int blocksY = DivUp(batchSize, elementsPerBlockY);

  stream->Dispatch(blocksX, blocksY, 1);

  stream->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(output.pResource));

  // run softmax pass
  stream->SetPipelineState(policy_softmax_state_);
  stream->Dispatch(batchSize, 1, 1);
#endif
}

void ShaderWrapper::valueFC1(dx_command_stream stream, DXAlloc output,
                             DXAlloc input, DXAlloc weights, DXAlloc biases,
                             int batchSize) {
  int Consts[] = {batchSize};
  stream->SetComputeRootSignature(root_sign_);
  stream->SetPipelineState(value_fc1_state_);
  stream->SetComputeRootUnorderedAccessView(0, output.gpuVA);
  stream->SetComputeRootUnorderedAccessView(1, input.gpuVA);
  stream->SetComputeRootUnorderedAccessView(2, weights.gpuVA);
  stream->SetComputeRootUnorderedAccessView(3, biases.gpuVA);
  stream->SetComputeRoot32BitConstants(4, 1, &Consts, 0);

  // Each thread writes two elements.
  // Block size is 128/2 = 64.
  // Gird size is 'batchSize' no of blocks.
  stream->Dispatch(batchSize, 1, 1);
}

void ShaderWrapper::valueFC2(dx_command_stream stream, DXAlloc output,
                             DXAlloc input, DXAlloc weights, DXAlloc biases,
                             int batchSize) {
  int Consts[] = {batchSize};
  stream->SetComputeRootSignature(root_sign_);
  stream->SetPipelineState(value_fc2_state_);
  stream->SetComputeRootUnorderedAccessView(0, output.gpuVA);
  stream->SetComputeRootUnorderedAccessView(1, input.gpuVA);
  stream->SetComputeRootUnorderedAccessView(2, weights.gpuVA);
  stream->SetComputeRootUnorderedAccessView(3, biases.gpuVA);
  stream->SetComputeRoot32BitConstants(4, 1, &Consts, 0);

  // Each thread writes a single element.
  int blockSize = 32;
  int numBlocks = DivUp(batchSize, blockSize);
  stream->Dispatch(numBlocks, 1, 1);
}

void ShaderWrapper::skipAddRelu(dx_command_stream stream, DXAlloc input1,
                                DXAlloc input2, DXAlloc output,
                                bool performRelu, int numElements) {
  int Consts[] = {numElements, performRelu};
  stream->SetComputeRootSignature(root_sign_);
  stream->SetPipelineState(skip_add_state_);
  stream->SetComputeRootUnorderedAccessView(0, output.gpuVA);
  stream->SetComputeRootUnorderedAccessView(1, input1.gpuVA);
  stream->SetComputeRootUnorderedAccessView(2, input2.gpuVA);
  stream->SetComputeRoot32BitConstants(4, 2, &Consts, 0);

  // Each thread writes 2 elements
  int blockSize = 512;
  int numBlocks = DivUp(numElements/2, blockSize);
  stream->Dispatch(numBlocks, 1, 1);
}

}  // namespace dx_backend
}  // namespace lczero
