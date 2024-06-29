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
#include "shader_wrapper.h"
#include <cassert>
#include <cstring>
#include "comdef.h"
#include "neural/network.h"
#include "shaders/shader_shared.h"
#include "shaders/shaders.h"

#define ARR_ELEMENT_COUNT(x) (sizeof(x) / sizeof(x[0]))
namespace lczero {
namespace dx_backend {

// Helper macros to reduce copy-paste.
#define CREATE_WINOGRAD_SE_PSO(channels)                                  \
  state_desc.CS = {g_output_transform_shader_fp32_se_##channels,          \
                   sizeof(g_output_transform_shader_fp32_se_##channels)}; \
  ReportDxErrors(device->CreateComputePipelineState(                      \
      &state_desc,                                                        \
      IID_PPV_ARGS(&winograd_output_transform_se_##channels##_)));

#define CREATE_SE_PSO(channels)                               \
  state_desc.CS = {g_se_##channels, sizeof(g_se_##channels)}; \
  ReportDxErrors(device->CreateComputePipelineState(          \
      &state_desc, IID_PPV_ARGS(&se_##channels##_)));


#define SET_WINOGRAD_SE_PSO(channels)                         \
  command_list->SetPipelineState(                             \
      winograd_output_transform_se_##channels##_);

void ShaderWrapper::Init(ID3D12Device* device) {
  // Create root signature - common for all shaders.

  // 8+1+8 slots
  // slot 0 to 7  -> root UAV slots 0 to 7 (all in space 0)
  // slot 8       -> root constants (16 constants - should be enough)
  // slot 9 to 16 -> descriptor UAVs of same allocations as slots 0-7, bound
  //                 at shader slots 8-15

  D3D12_ROOT_PARAMETER root_parameter[kUavSlots + 1 + kUavSlots];
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

  D3D12_DESCRIPTOR_RANGE descRange[kUavSlots] = {};
  for (int i = 0; i < kUavSlots; i++) {
    descRange[i].BaseShaderRegister = i + kUavSlots;
    descRange[i].NumDescriptors = 1;
    descRange[i].OffsetInDescriptorsFromTableStart = 0;
    descRange[i].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    descRange[i].RegisterSpace = 0;

    root_parameter[i + kUavSlots + 1].ParameterType =
        D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameter[i + kUavSlots + 1].DescriptorTable.NumDescriptorRanges = 1;
    root_parameter[i + kUavSlots + 1].DescriptorTable.pDescriptorRanges =
        &descRange[i];
    root_parameter[i + kUavSlots + 1].ShaderVisibility =
        D3D12_SHADER_VISIBILITY_ALL;
  }

  D3D12_ROOT_SIGNATURE_DESC root_sig_desc = {kUavSlots + 1 + kUavSlots,
                                             root_parameter, 0, NULL,
                                             D3D12_ROOT_SIGNATURE_FLAG_NONE};

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
      &state_desc, IID_PPV_ARGS(&expand_planes_fp16_)));

  state_desc.CS = {g_ExpandPlanes_shader_fp32,
                   sizeof(g_ExpandPlanes_shader_fp32)};
  ReportDxErrors(device->CreateComputePipelineState(
      &state_desc, IID_PPV_ARGS(&expand_planes_fp32_)));

  // Winograd Input Transform shader.
  state_desc.CS = {g_input_transform_shader_fp32,
                   sizeof(g_input_transform_shader_fp32)};
  ReportDxErrors(device->CreateComputePipelineState(
      &state_desc, IID_PPV_ARGS(&winograd_input_transform_)));

  // Winograd Output Transform shader.
  state_desc.CS = {g_output_transform_shader_fp32,
                   sizeof(g_output_transform_shader_fp32)};
  ReportDxErrors(device->CreateComputePipelineState(
      &state_desc, IID_PPV_ARGS(&winograd_output_transform_)));

  // 1x1 convolution shader.
  state_desc.CS = {g_conv_1x1_shader_fp32, sizeof(g_conv_1x1_shader_fp32)};
  ReportDxErrors(device->CreateComputePipelineState(
      &state_desc, IID_PPV_ARGS(&conv_1x1_)));

  // policy map shader.
  state_desc.CS = {g_policy_map_shader_fp32, sizeof(g_policy_map_shader_fp32)};
  ReportDxErrors(device->CreateComputePipelineState(
      &state_desc, IID_PPV_ARGS(&policy_map_)));

  state_desc.CS = {g_MatrixMul_Fp32, sizeof(g_MatrixMul_Fp32)};
  ReportDxErrors(device->CreateComputePipelineState(&state_desc,
                                                    IID_PPV_ARGS(&gemm_)));

  // Add vectors shader.
  state_desc.CS = {g_add_vectors_shader, sizeof(g_add_vectors_shader)};
  ReportDxErrors(device->CreateComputePipelineState(
      &state_desc, IID_PPV_ARGS(&add_vectors_)));

  // Various SE shaders.
  CREATE_SE_PSO(128);
  CREATE_SE_PSO(256);
  CREATE_SE_PSO(320);
  CREATE_SE_PSO(384);
  CREATE_SE_PSO(512);
  CREATE_SE_PSO(640);
  CREATE_SE_PSO(768);
  CREATE_SE_PSO(1024);

  // Various output-transform fused with SE shaders.
  CREATE_WINOGRAD_SE_PSO(128)
  CREATE_WINOGRAD_SE_PSO(256)
  CREATE_WINOGRAD_SE_PSO(320)
  CREATE_WINOGRAD_SE_PSO(384)
  CREATE_WINOGRAD_SE_PSO(512)
  CREATE_WINOGRAD_SE_PSO(640)
  CREATE_WINOGRAD_SE_PSO(768)
  CREATE_WINOGRAD_SE_PSO(1024)
}

void ShaderWrapper::Destroy() {
  expand_planes_fp16_->Release();
  expand_planes_fp32_->Release();

  winograd_input_transform_->Release();
  winograd_output_transform_->Release();
  conv_1x1_->Release();
  policy_map_->Release();
  gemm_->Release();
  add_vectors_->Release();

  se_128_->Release();
  se_256_->Release();
  se_320_->Release();
  se_384_->Release();
  se_512_->Release();
  se_640_->Release();
  se_768_->Release();
  se_1024_->Release();

  winograd_output_transform_se_128_->Release();
  winograd_output_transform_se_256_->Release();
  winograd_output_transform_se_320_->Release();
  winograd_output_transform_se_384_->Release();
  winograd_output_transform_se_512_->Release();
  winograd_output_transform_se_640_->Release();
  winograd_output_transform_se_768_->Release();
  winograd_output_transform_se_1024_->Release();

  root_sign_->Release();
}

void ShaderWrapper::ExpandPlanes(ID3D12GraphicsCommandList4* command_list,
                                 DXAlloc output_tensor, DXAlloc masks,
                                 DXAlloc values, int batchSize, bool fp16) {
  const int N = batchSize * kInputPlanes;
  int consts[] = {N, kInputPlanes};
  command_list->SetComputeRootSignature(root_sign_);
  command_list->SetPipelineState(fp16 ? expand_planes_fp16_
                                      : expand_planes_fp32_);
  command_list->SetComputeRootUnorderedAccessView(0, output_tensor.gpu_va);
  command_list->SetComputeRootUnorderedAccessView(1, masks.gpu_va);
  command_list->SetComputeRootUnorderedAccessView(2, values.gpu_va);
  command_list->SetComputeRoot32BitConstants(kUavSlots, 2, &consts, 0);

  int elements = batchSize * kInputPlanes * 8 * 8;
  int blocks = DivUp(elements, kExpandPlanesElementsPerBlock);
  command_list->Dispatch(blocks, 1, 1);
}

void ShaderWrapper::InputTransform(ID3D12GraphicsCommandList4* command_list,
                                   DXAlloc transformed_input, DXAlloc input,
                                   int N, int C, bool /*fp16*/) {
  int consts[] = {N, C};
  command_list->SetComputeRootSignature(root_sign_);
  command_list->SetPipelineState(winograd_input_transform_);
  command_list->SetComputeRootUnorderedAccessView(0, input.gpu_va);
  command_list->SetComputeRootUnorderedAccessView(1, transformed_input.gpu_va);
  command_list->SetComputeRoot32BitConstants(
      kUavSlots, ARR_ELEMENT_COUNT(consts), &consts, 0);
  command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 0,
                                              input.desc_handle_vector);
  command_list->SetComputeRootDescriptorTable(
      kUavSlots + 1 + 1, transformed_input.desc_handle_scalar);

  int blocks = DivUp(N * C, kWinogradTransformShaderBlockSize);
  command_list->Dispatch(blocks, 1, 1);
}

void ShaderWrapper::Se(ID3D12GraphicsCommandList4* command_list, DXAlloc output,
    DXAlloc input, DXAlloc skip_connection, DXAlloc bias,
    DXAlloc se_w1, DXAlloc se_b1, DXAlloc se_w2,
    DXAlloc se_b2, int N, int K, bool relu, bool bias_add,
    bool skip_add, int se_k, bool /*fp16*/) {
  int consts[] = {N, K, relu, bias_add, skip_add, se_k};
  command_list->SetComputeRootSignature(root_sign_);
  command_list->SetComputeRootUnorderedAccessView(0, input.gpu_va);
  command_list->SetComputeRootUnorderedAccessView(1, output.gpu_va);
  if (bias_add) command_list->SetComputeRootUnorderedAccessView(2, bias.gpu_va);
  if (skip_add)
    command_list->SetComputeRootUnorderedAccessView(3, skip_connection.gpu_va);
  command_list->SetComputeRootUnorderedAccessView(4, se_w1.gpu_va);
  command_list->SetComputeRootUnorderedAccessView(5, se_b1.gpu_va);
  command_list->SetComputeRootUnorderedAccessView(6, se_w2.gpu_va);
  command_list->SetComputeRootUnorderedAccessView(7, se_b2.gpu_va);

  command_list->SetComputeRoot32BitConstants(
      kUavSlots, ARR_ELEMENT_COUNT(consts), &consts, 0);

  command_list->SetComputeRootDescriptorTable(
      kUavSlots + 1 + 0, input.desc_handle_vector);
  command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 1,
                                              output.desc_handle_vector);
  if (bias_add)
    command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 2,
                                                bias.desc_handle_scalar);
  if (skip_add)
    command_list->SetComputeRootDescriptorTable(
        kUavSlots + 1 + 3, skip_connection.desc_handle_vector);

  command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 4,
                                              se_w1.desc_handle_scalar);
  command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 5,
                                              se_b1.desc_handle_scalar);
  command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 6,
                                              se_w2.desc_handle_scalar);
  command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 7,
                                              se_b2.desc_handle_scalar);

  int blocks = N;
  if (K <= 128)
    command_list->SetPipelineState(se_128_);
  else if (K <= 256)
    command_list->SetPipelineState(se_256_);
  else if (K <= 320)
    command_list->SetPipelineState(se_320_);
  else if (K <= 384)
    command_list->SetPipelineState(se_384_);
  else if (K <= 512)
    command_list->SetPipelineState(se_512_);
  else if (K <= 640)
    command_list->SetPipelineState(se_640_);
  else if (K <= 768)
    command_list->SetPipelineState(se_768_);
  else if (K <= 1024)
    command_list->SetPipelineState(se_1024_);
  else
    throw Exception("Unsupported channel count for SE");

  command_list->Dispatch(blocks, 1, 1);
}

void ShaderWrapper::OutputTransform(ID3D12GraphicsCommandList4* command_list,
                                    DXAlloc output, DXAlloc transformed_output,
                                    DXAlloc skip_connection, DXAlloc bias,
                                    DXAlloc se_w1, DXAlloc se_b1, DXAlloc se_w2,
                                    DXAlloc se_b2, int N, int K, bool relu,
                                    bool bias_add, bool skip_add, bool fused_se,
                                    int se_k, bool /*fp16*/) {
  int consts[] = {N, K, relu, bias_add, skip_add, fused_se, se_k};
  command_list->SetComputeRootSignature(root_sign_);
  command_list->SetComputeRootUnorderedAccessView(0, transformed_output.gpu_va);
  command_list->SetComputeRootUnorderedAccessView(1, output.gpu_va);
  if (bias_add) command_list->SetComputeRootUnorderedAccessView(2, bias.gpu_va);
  if (skip_add)
    command_list->SetComputeRootUnorderedAccessView(3, skip_connection.gpu_va);
  if (fused_se) {
    command_list->SetComputeRootUnorderedAccessView(4, se_w1.gpu_va);
    command_list->SetComputeRootUnorderedAccessView(5, se_b1.gpu_va);
    command_list->SetComputeRootUnorderedAccessView(6, se_w2.gpu_va);
    command_list->SetComputeRootUnorderedAccessView(7, se_b2.gpu_va);
  }
  command_list->SetComputeRoot32BitConstants(
      kUavSlots, ARR_ELEMENT_COUNT(consts), &consts, 0);

  command_list->SetComputeRootDescriptorTable(
      kUavSlots + 1 + 0, transformed_output.desc_handle_scalar);
  command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 1,
                                              output.desc_handle_vector);
  if (bias_add)
    command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 2,
                                                bias.desc_handle_scalar);
  if (skip_add)
    command_list->SetComputeRootDescriptorTable(
        kUavSlots + 1 + 3, skip_connection.desc_handle_vector);
  if (fused_se) {
    command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 4,
                                                se_w1.desc_handle_scalar);
    command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 5,
                                                se_b1.desc_handle_scalar);
    command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 6,
                                                se_w2.desc_handle_scalar);
    command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 7,
                                                se_b2.desc_handle_scalar);
  }

  int blocks = 0;
  if (fused_se) {
    blocks = N;
    if (K <= 128)
      SET_WINOGRAD_SE_PSO(128)
    else if (K <= 256)
      SET_WINOGRAD_SE_PSO(256)
    else if (K <= 320)
      SET_WINOGRAD_SE_PSO(320)
    else if (K <= 384)
      SET_WINOGRAD_SE_PSO(384)
    else if (K <= 512)
      SET_WINOGRAD_SE_PSO(512)
    else if (K <= 640)
      SET_WINOGRAD_SE_PSO(640)
    else if (K <= 768)
      SET_WINOGRAD_SE_PSO(768)
    else if (K <= 1024)
      SET_WINOGRAD_SE_PSO(1024)
    else
      throw Exception("Unsupported channel count for SE");

  } else {
    blocks = DivUp(N * K, kWinogradTransformShaderBlockSize);
    command_list->SetPipelineState(winograd_output_transform_);
  }

  command_list->Dispatch(blocks, 1, 1);
}

void ShaderWrapper::Conv1x1(ID3D12GraphicsCommandList4* command_list,
                            DXAlloc output, DXAlloc input, DXAlloc weight,
                            DXAlloc bias, int N, int C, int K, bool relu,
                            bool useBias, bool /*fp16*/) {
  int consts[] = {N, K, C, useBias, relu};
  command_list->SetComputeRootSignature(root_sign_);
  command_list->SetPipelineState(conv_1x1_);
  command_list->SetComputeRootUnorderedAccessView(0, output.gpu_va);
  command_list->SetComputeRootUnorderedAccessView(1, input.gpu_va);
  command_list->SetComputeRootUnorderedAccessView(2, weight.gpu_va);
  command_list->SetComputeRootUnorderedAccessView(3, bias.gpu_va);
  command_list->SetComputeRoot32BitConstants(
      kUavSlots, ARR_ELEMENT_COUNT(consts), &consts, 0);

  command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 0,
                                              output.desc_handle_scalar);
  command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 1,
                                              input.desc_handle_scalar);
  command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 2,
                                              weight.desc_handle_scalar);
  command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 3,
                                              bias.desc_handle_scalar);

  command_list->Dispatch(K, N, 1);
}

void ShaderWrapper::AddVectors(ID3D12GraphicsCommandList4* command_list,
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
  command_list->SetComputeRootUnorderedAccessView(0, A.gpu_va);
  command_list->SetComputeRootUnorderedAccessView(1, B.gpu_va);
  command_list->SetComputeRootUnorderedAccessView(2, C.gpu_va);
  command_list->SetComputeRoot32BitConstants(
      kUavSlots, ARR_ELEMENT_COUNT(consts), &consts, 0);

  int blocks = DivUp(c_size, kAddVectorsBlockSize);
  command_list->Dispatch(blocks, 1, 1);
}

void ShaderWrapper::PolicyMap(ID3D12GraphicsCommandList4* command_list,
                              DXAlloc output, DXAlloc input, DXAlloc weights,
                              int N, int input_size, int output_size,
                              int used_size, bool /*fp16*/) {
  int consts[] = {N, input_size, used_size, output_size};
  command_list->SetComputeRootSignature(root_sign_);
  command_list->SetPipelineState(policy_map_);
  command_list->SetComputeRootUnorderedAccessView(0, input.gpu_va);
  command_list->SetComputeRootUnorderedAccessView(1, output.gpu_va);
  command_list->SetComputeRootUnorderedAccessView(2, weights.gpu_va);
  command_list->SetComputeRoot32BitConstants(kUavSlots, ARR_ELEMENT_COUNT(consts), &consts, 0);
  command_list->SetComputeRootDescriptorTable(kUavSlots+1, input.desc_handle_scalar);

  int blocks = DivUp(N * used_size, kPolicyMapBlockSize);
  command_list->Dispatch(blocks, 1, 1);
}

void ShaderWrapper::MatrixMultiply(ID3D12GraphicsCommandList4* command_list,
                                   DXAlloc output, DXAlloc A, DXAlloc B, int M,
                                   int N, int K, int batch, bool /*fp16*/) {
  int Consts[] = {M, N, K, batch};
  command_list->SetComputeRootSignature(root_sign_);

  command_list->SetPipelineState(gemm_);

  command_list->SetComputeRootUnorderedAccessView(0, A.gpu_va);
  command_list->SetComputeRootUnorderedAccessView(1, B.gpu_va);
  command_list->SetComputeRootUnorderedAccessView(2, output.gpu_va);
  command_list->SetComputeRoot32BitConstants(
      kUavSlots, ARR_ELEMENT_COUNT(Consts), &Consts, 0);
  command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 0,
                                              A.desc_handle_vector);
  command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 1,
                                              B.desc_handle_vector);
  command_list->SetComputeRootDescriptorTable(kUavSlots + 1 + 2,
                                              output.desc_handle_vector);


  int blocksX = DivUp(N, kGemmElPerBlockX);
  int blocksY = DivUp(M, kGemmElPerBlockY);
  int blocksZ = batch;

  command_list->Dispatch(blocksX, blocksY, blocksZ);
}

}  // namespace dx_backend
}  // namespace lczero
