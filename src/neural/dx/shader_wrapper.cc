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
#include "shader_wrapper.h"
#include <cassert>
#include <cstring>
#include "comdef.h"
#include "neural/network.h"
#include "shaders/shader_shared.h"
#include "shaders/shaders.h"
#include "shaders/shaders_se.h"

namespace lczero {
namespace dx_backend {

// Helper macros to reduce copy-paste.
#define CREATE_SE_PSO(datatype, channels)                            \
  state_desc.CS = {                                                  \
      g_output_transform_shader_##datatype##_se_##channels,          \
      sizeof(g_output_transform_shader_##datatype##_se_##channels)}; \
  ReportDxErrors(device->CreateComputePipelineState(                 \
      &state_desc,                                                   \
      IID_PPV_ARGS(                                                  \
          &winograd_output_transform_##datatype##_se_##channels##_))); \

#define SET_SE_PSO(channels)                                 \
  command_list->SetPipelineState(                            \
      fp16 ? winograd_output_transform_fp16_se_##channels##_ \
           : winograd_output_transform_fp32_se_##channels##_); \

void ShaderWrapper::init(ID3D12Device* device) {
  // Create root signature - common for all shaders.

  // 9 slots
  // slot 0 to 7 -> root UAV slots 0 to 7 (all in space 0)
  // slot 8      -> root constants (16 constants - should be enough)

  D3D12_ROOT_PARAMETER root_parameter[kUavSlots + 1];
  for (int i = 0; i < kUavSlots; i++) {
    root_parameter[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    root_parameter[i].Descriptor.RegisterSpace = 0;
    root_parameter[i].Descriptor.ShaderRegister = i;
    root_parameter[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }

  root_parameter[kUavSlots].ParameterType =
      D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  root_parameter[kUavSlots].Constants.RegisterSpace = 0;
  root_parameter[kUavSlots].Constants.ShaderRegister = 0;
  root_parameter[kUavSlots].Constants.Num32BitValues = 16;
  root_parameter[kUavSlots].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

  D3D12_ROOT_SIGNATURE_DESC root_sig_desc = {
      kUavSlots + 1, root_parameter, 0, NULL, D3D12_ROOT_SIGNATURE_FLAG_NONE};

  ID3DBlob* serialized_layout = NULL;
  D3D12SerializeRootSignature(&root_sig_desc, D3D_ROOT_SIGNATURE_VERSION_1_0,
                              &serialized_layout, NULL);

  ReportDxErrors(device->CreateRootSignature(
      1, serialized_layout->GetBufferPointer(),
      serialized_layout->GetBufferSize(), IID_PPV_ARGS(&root_sign_)));

  serialized_layout->Release();

  // Create PSO objects for each shader.
  // PSO basically holds the compiled shader object.

  // Expand planes shaders.
  D3D12_COMPUTE_PIPELINE_STATE_DESC state_desc = {};
  state_desc.pRootSignature = root_sign_;

  state_desc.CS = {g_ExpandPlanes_shader_fp16,
                   sizeof(g_ExpandPlanes_shader_fp16)};
  ReportDxErrors(device->CreateComputePipelineState(
      &state_desc, IID_PPV_ARGS(&expand_planes_state_fp16_)));

  state_desc.CS = {g_ExpandPlanes_shader_fp32,
                   sizeof(g_ExpandPlanes_shader_fp32)};
  ReportDxErrors(device->CreateComputePipelineState(
      &state_desc, IID_PPV_ARGS(&expand_planes_state_fp32_)));

  // Winograd Input Transform shaders.
  state_desc.CS = {g_input_transform_shader_fp16,
                   sizeof(g_input_transform_shader_fp16)};
  ReportDxErrors(device->CreateComputePipelineState(
      &state_desc, IID_PPV_ARGS(&winograd_input_transform_fp16_)));

  state_desc.CS = {g_input_transform_shader_fp32,
                   sizeof(g_input_transform_shader_fp32)};
  ReportDxErrors(device->CreateComputePipelineState(
      &state_desc, IID_PPV_ARGS(&winograd_input_transform_fp32_)));

  // Winograd Output Transform shaders.
  state_desc.CS = {g_output_transform_shader_fp16,
                   sizeof(g_output_transform_shader_fp16)};
  ReportDxErrors(device->CreateComputePipelineState(
      &state_desc, IID_PPV_ARGS(&winograd_output_transform_fp16_)));

  state_desc.CS = {g_output_transform_shader_fp32,
                   sizeof(g_output_transform_shader_fp32)};
  ReportDxErrors(device->CreateComputePipelineState(
      &state_desc, IID_PPV_ARGS(&winograd_output_transform_fp32_)));

  // 1x1 convolution shaders.
  state_desc.CS = {g_conv_1x1_shader_fp16, sizeof(g_conv_1x1_shader_fp16)};
  ReportDxErrors(device->CreateComputePipelineState(
      &state_desc, IID_PPV_ARGS(&conv_1x1_fp16_)));

  state_desc.CS = {g_conv_1x1_shader_fp32, sizeof(g_conv_1x1_shader_fp32)};
  ReportDxErrors(device->CreateComputePipelineState(
      &state_desc, IID_PPV_ARGS(&conv_1x1_fp32_)));

  // policy map shaders.
  state_desc.CS = {g_policy_map_shader_fp16, sizeof(g_policy_map_shader_fp16)};
  ReportDxErrors(device->CreateComputePipelineState(
      &state_desc, IID_PPV_ARGS(&policy_map_fp16_)));

  state_desc.CS = {g_policy_map_shader_fp32, sizeof(g_policy_map_shader_fp32)};
  ReportDxErrors(device->CreateComputePipelineState(
      &state_desc, IID_PPV_ARGS(&policy_map_fp32_)));

  // Add vectors shader.
  state_desc.CS = {g_add_vectors_shader, sizeof(g_add_vectors_shader)};
  ReportDxErrors(device->CreateComputePipelineState(
      &state_desc, IID_PPV_ARGS(&add_vectors_)));

  // Various output-transform fused with SE shaders
  CREATE_SE_PSO(fp16, 128)
  CREATE_SE_PSO(fp16, 256)
  CREATE_SE_PSO(fp16, 320)
  CREATE_SE_PSO(fp16, 384)
  CREATE_SE_PSO(fp16, 512)
  CREATE_SE_PSO(fp16, 640)
  CREATE_SE_PSO(fp16, 768)
  CREATE_SE_PSO(fp16, 1024)

  CREATE_SE_PSO(fp32, 128)
  CREATE_SE_PSO(fp32, 256)
  CREATE_SE_PSO(fp32, 320)
  CREATE_SE_PSO(fp32, 384)
  CREATE_SE_PSO(fp32, 512)
  CREATE_SE_PSO(fp32, 640)
  CREATE_SE_PSO(fp32, 768)
  CREATE_SE_PSO(fp32, 1024)
}

void ShaderWrapper::destroy() {
  expand_planes_state_fp16_->Release();
  winograd_input_transform_fp16_->Release();
  winograd_output_transform_fp16_->Release();
  conv_1x1_fp16_->Release();

  expand_planes_state_fp32_->Release();
  winograd_input_transform_fp32_->Release();
  winograd_output_transform_fp32_->Release();
  conv_1x1_fp32_->Release();

  root_sign_->Release();
}

void ShaderWrapper::expandPlanes(ID3D12GraphicsCommandList5* command_list,
                                 DXAlloc output_tensor, DXAlloc masks,
                                 DXAlloc values, int batchSize, bool fp16) {
  const int N = batchSize * kInputPlanes;
  int consts[] = {N, kInputPlanes};
  command_list->SetComputeRootSignature(root_sign_);
  command_list->SetPipelineState(fp16 ? expand_planes_state_fp16_
                                      : expand_planes_state_fp32_);
  command_list->SetComputeRootUnorderedAccessView(0, output_tensor.gpuVA);
  command_list->SetComputeRootUnorderedAccessView(1, masks.gpuVA);
  command_list->SetComputeRootUnorderedAccessView(2, values.gpuVA);
  command_list->SetComputeRoot32BitConstants(kUavSlots, 2, &consts, 0);

  int elements = batchSize * kInputPlanes * 8 * 8;
  int blocks = DivUp(elements, kExpandPlanesElementsPerBlock);
  command_list->Dispatch(blocks, 1, 1);
}

void ShaderWrapper::inputTransform(ID3D12GraphicsCommandList5* command_list,
                                   DXAlloc transformed_input, DXAlloc input,
                                   int N, int C, bool fp16) {
  int consts[] = {N, C};
  command_list->SetComputeRootSignature(root_sign_);
  command_list->SetPipelineState(fp16 ? winograd_input_transform_fp16_
                                      : winograd_input_transform_fp32_);
  command_list->SetComputeRootUnorderedAccessView(0, input.gpuVA);
  command_list->SetComputeRootUnorderedAccessView(1, transformed_input.gpuVA);
  command_list->SetComputeRoot32BitConstants(kUavSlots, 2, &consts, 0);

  int blocks = DivUp(N * C, kWinogradTransformShaderBlockSize);
  command_list->Dispatch(blocks, 1, 1);
}

void ShaderWrapper::outputTransform(ID3D12GraphicsCommandList5* command_list,
                                    DXAlloc output, DXAlloc transformed_output,
                                    DXAlloc skip_connection, DXAlloc bias,
                                    DXAlloc se_w1, DXAlloc se_b1, DXAlloc se_w2,
                                    DXAlloc se_b2, int N, int K, bool relu,
                                    bool bias_add, bool skip_add, bool fused_se,
                                    int se_k, bool fp16) {
  int consts[] = {N, K, relu, bias_add, skip_add, fused_se, se_k};
  command_list->SetComputeRootSignature(root_sign_);
  command_list->SetComputeRootUnorderedAccessView(0, transformed_output.gpuVA);
  command_list->SetComputeRootUnorderedAccessView(1, output.gpuVA);
  command_list->SetComputeRootUnorderedAccessView(2, bias.gpuVA);
  command_list->SetComputeRootUnorderedAccessView(3, skip_connection.gpuVA);
  command_list->SetComputeRootUnorderedAccessView(4, se_w1.gpuVA);
  command_list->SetComputeRootUnorderedAccessView(5, se_b1.gpuVA);
  command_list->SetComputeRootUnorderedAccessView(6, se_w2.gpuVA);
  command_list->SetComputeRootUnorderedAccessView(7, se_b2.gpuVA);
  command_list->SetComputeRoot32BitConstants(kUavSlots, 7, &consts, 0);

  int blocks = 0;
  if (fused_se) {
    blocks = N;
    if (K <= 128)
      SET_SE_PSO(128)
    else if (K <= 256)
      SET_SE_PSO(256)
    else if (K <= 320)
      SET_SE_PSO(320)
    else if (K <= 384)
      SET_SE_PSO(384)
    else if (K <= 512)
      SET_SE_PSO(512)
    else if (K <= 640)
      SET_SE_PSO(640)
    else if (K <= 768)
      SET_SE_PSO(768)
    else if (K <= 1024)
      SET_SE_PSO(1024)
    else
      throw Exception("Unsupported channel count for SE");

  } else {
    blocks = DivUp(N * K, kWinogradTransformShaderBlockSize);
    command_list->SetPipelineState(fp16 ? winograd_output_transform_fp16_
                                        : winograd_output_transform_fp32_);
  }

  command_list->Dispatch(blocks, 1, 1);
}

void ShaderWrapper::conv1x1(ID3D12GraphicsCommandList5* command_list,
                            DXAlloc output, DXAlloc input, DXAlloc weight,
                            DXAlloc bias, int N, int C, int K, bool relu,
                            bool useBias, bool fp16) {
  int consts[] = {N, K, C, useBias, relu};
  command_list->SetComputeRootSignature(root_sign_);
  command_list->SetPipelineState(fp16 ? conv_1x1_fp16_ : conv_1x1_fp32_);
  command_list->SetComputeRootUnorderedAccessView(0, output.gpuVA);
  command_list->SetComputeRootUnorderedAccessView(1, input.gpuVA);
  command_list->SetComputeRootUnorderedAccessView(2, weight.gpuVA);
  command_list->SetComputeRootUnorderedAccessView(3, bias.gpuVA);
  command_list->SetComputeRoot32BitConstants(kUavSlots, 5, &consts, 0);
  command_list->Dispatch(K, N, 1);
}

void ShaderWrapper::addVectors(ID3D12GraphicsCommandList5* command_list,
                               DXAlloc C, DXAlloc A, DXAlloc B, int c_size,
                               int a_size, int b_size, bool relu, bool tanh,
                               bool fp16) {
  if (fp16) {
    // Shader handles 2 elements per thread in fp16 mode.
    assert(a_size % 2 == 0);
    assert(b_size % 2 == 0);
    assert(c_size % 2 == 0);
    a_size /= 2;
    b_size /= 2;
    c_size /= 2;
  }
  int consts[] = {a_size, b_size, c_size, relu, tanh, fp16};
  command_list->SetComputeRootSignature(root_sign_);
  command_list->SetPipelineState(add_vectors_);
  command_list->SetComputeRootUnorderedAccessView(0, A.gpuVA);
  command_list->SetComputeRootUnorderedAccessView(1, B.gpuVA);
  command_list->SetComputeRootUnorderedAccessView(2, C.gpuVA);
  command_list->SetComputeRoot32BitConstants(kUavSlots, 6, &consts, 0);

  int blocks = DivUp(c_size, kAddVectorsBlockSize);
  command_list->Dispatch(blocks, 1, 1);
}

void ShaderWrapper::PolicyMap(ID3D12GraphicsCommandList5* command_list,
                              DXAlloc output, DXAlloc input, DXAlloc weights,
                              int N, int input_size, int output_size,
                              int used_size, bool fp16) {
  int consts[] = {N, input_size, used_size, output_size};
  command_list->SetComputeRootSignature(root_sign_);
  command_list->SetPipelineState(fp16 ? policy_map_fp16_ : policy_map_fp32_);
  command_list->SetComputeRootUnorderedAccessView(0, input.gpuVA);
  command_list->SetComputeRootUnorderedAccessView(1, output.gpuVA);
  command_list->SetComputeRootUnorderedAccessView(2, weights.gpuVA);
  command_list->SetComputeRoot32BitConstants(kUavSlots, 4, &consts, 0);

  int blocks = DivUp(N*used_size, kPolicyMapBlockSize);
  command_list->Dispatch(blocks, 1, 1);
}

}  // namespace dx_backend
}  // namespace lczero
