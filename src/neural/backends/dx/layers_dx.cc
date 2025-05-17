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

#include "layers_dx.h"

#include <comdef.h>

#include <cassert>
#include <cstring>
#include <vector>

#include "MetaCommand.h"
#include "network_dx.h"
#include "utils/exception.h"

namespace lczero {
namespace dx_backend {

// Utility functions used only in this file.
namespace {

static void CopyFloatToHalf(dx_half* out, const float* in, size_t elements) {
  for (int i = 0; i < elements; i++) {
    out[i] = FP32toFP16(in[i]);
  }
}

static void CpuTranspose(float* op, float* ip, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; i++)
    for (size_t j = 0; j < cols; j++) op[j * rows + i] = ip[i * cols + j];
}

template <int M, int N, int K>
static void MatrixMulCPU(float* c, const float* a, const float* b) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float S = 0;
      for (int k = 0; k < K; ++k) S += a[i * K + k] * b[k * N + j];
      c[i * N + j] = S;
    }
}

static void FilterTransform4x4(float* transformed_filter, const float* filter) {
  // transform applied to filter (of size 3x3)
  float G[6 * 3] = {1.0f / 4,  0,         0,         -1.0f / 6,  -1.0f / 6,
                    -1.0f / 6, -1.0f / 6, 1.0f / 6,  -1.0f / 6,  1.0f / 24,
                    1.0f / 12, 1.0f / 6,  1.0f / 24, -1.0f / 12, 1.0f / 6,
                    0,         0,         1};

  float Gt[3 * 6] = {1.0f / 4, -1.0f / 6, -1.0f / 6, 1.0f / 24, 1.0f / 24,  0,
                     0,        -1.0f / 6, 1.0f / 6,  1.0f / 12, -1.0f / 12, 0,
                     0,        -1.0f / 6, -1.0f / 6, 1.0f / 6,  1.0f / 6,   1};

  float temp_filter[6 * 3];
  MatrixMulCPU<6, 3, 3>(temp_filter, G, filter);
  MatrixMulCPU<6, 6, 3>(transformed_filter, temp_filter, Gt);
}

#define FILTER_IDX_NCHW(k, c, h, w) ((k)*C * S * R + (c)*S * R + (h)*R + w)

// Transform filter for winograd.
// (e.g: for K C H W - 256x256x3x3, filter output is 6x6x256x256 - H W K C)
static void TransformFilterTensor_Winograd4x4(int K, int C,
                                              float* transformed_filter,
                                              const float* weight) {
  constexpr int S = 3;
  constexpr int R = 3;

  for (int k = 0; k < K; k++) {
    for (int c = 0; c < C; c++) {
      // 1. read single filter from memory
      float filter_tile[3][3];
      for (int s = 0; s < S; s++)
        for (int r = 0; r < R; r++) {
          filter_tile[s][r] = weight[FILTER_IDX_NCHW(k, c, s, r)];
        }

      // 2. transform it
      float transformed_tile[6][6];
      FilterTransform4x4(&(transformed_tile[0][0]), &(filter_tile[0][0]));

      // 3. write it back to memory (in HWCK layout)
      for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) {
          transformed_filter[i * 6 * C * K + j * C * K + c * K + k] =
              transformed_tile[i][j];
        }
    }
  }
}

static void GetGemmTensorDesc(TensorDesc* out_desc, int batch_size, int rows,
                              int cols, bool fp16) {
  memset(out_desc, 0, sizeof(TensorDesc));
  out_desc->DimensionCount = 4;
  out_desc->DataType = fp16 ? 1 : 0;

  out_desc->Size[0] = batch_size;
  out_desc->Size[1] = 1;
  out_desc->Size[2] = rows;
  out_desc->Size[3] = cols;

  // row-major by default
  out_desc->Stride[3] = 1;
  out_desc->Stride[2] = cols;
  out_desc->Stride[1] = rows * cols;
  out_desc->Stride[0] = rows * cols;

  for (int i = 0; i < 4; i++) out_desc->StrideAlignment[i] = 1;

  out_desc->BaseAlignmentInBytes = 4096;  // arbitary
  out_desc->PhysicalSizeInElements = batch_size * rows * cols;
}

static void GetConvTensorDesc(TensorDesc* out_desc, int N, int C, int H, int W,
                              bool fp16) {
  memset(out_desc, 0, sizeof(TensorDesc));
  out_desc->DimensionCount = 4;
  out_desc->DataType = fp16 ? 1 : 0;

  out_desc->Size[0] = N;
  out_desc->Size[1] = C;
  out_desc->Size[2] = H;
  out_desc->Size[3] = W;

  // NCHW layout
  out_desc->Stride[3] = 1;
  out_desc->Stride[2] = W;
  out_desc->Stride[1] = H * W;
  out_desc->Stride[0] = C * H * W;

  for (int i = 0; i < 4; i++) out_desc->StrideAlignment[i] = 1;

  out_desc->BaseAlignmentInBytes = 4096;  // arbitary
  out_desc->PhysicalSizeInElements = N * C * H * W;
}

}  // namespace

GemmMetaCommand::GemmMetaCommand(DxContext* dx_context, int rows, int cols,
                                 int K, int gemm_batch, bool fp16,
                                 bool a_transpose, bool b_transpose) {
  memset(scratch_data_persistent_, 0, sizeof(scratch_data_persistent_));
  memset(scratch_data_temporary_, 0, sizeof(scratch_data_temporary_));
  memset(meta_commands_, 0, sizeof(meta_commands_));

  // Note: the way GEMM is used, the 'rows'/M - dimension is a function of
  // batch size. gemm_batch is different and unrelated (either 36 for Winograd,
  // or 1 for other FC layers)
  int num_meta_commands = 1;
  if (rows == 0) {
    // Create metacommands for each 'rows' that is multiple of 8.
    num_meta_commands = kMaxMetacommands;
    rows_known_ = false;
  } else {
    rows_known_ = true;
  }

  for (int i = 0; i < num_meta_commands; i++) {
    int num_rows = rows ? rows : (i + 1) * kMetacommandGranulity;

    GemmCreateDesc createDesc = {};
    GetGemmTensorDesc(&createDesc.DescOut, gemm_batch, num_rows, cols, fp16);
    GetGemmTensorDesc(&createDesc.DescA, gemm_batch, a_transpose ? K : num_rows,
                      a_transpose ? num_rows : K, fp16);
    GetGemmTensorDesc(&createDesc.DescB, gemm_batch, b_transpose ? cols : K,
                      b_transpose ? K : cols, fp16);
    createDesc.cMatrixNull = 1;
    createDesc.ActivationIsNull = 1;
    createDesc.Alpha = 1.0;
    createDesc.Beta = 0.0;
    createDesc.Precision = fp16 ? 1 : 0;  // 0 - fp32, 1 - fp16
    createDesc.TransA = a_transpose;
    createDesc.TransB = b_transpose;

    ID3D12MetaCommand* pMetacommand = nullptr;
    HRESULT hr = dx_context->getDevice()->CreateMetaCommand(
        GemmGuid, 1, &createDesc, sizeof(createDesc),
        IID_PPV_ARGS(&pMetacommand));

    if (hr != S_OK) {
#ifdef DEBUG_DUMP_PER_LAYER_DATA
      printf(
          "\nCan't create GEMM Metacommand for "
          "rows: %d, cols: %d, K: %d, batch: "
          "%d\n",
          num_rows, cols, K, gemm_batch);
#endif
      create_succeeded_ = false;
      return;
    }

    meta_commands_[i] = pMetacommand;

    size_t persistent_size = pMetacommand->GetRequiredParameterResourceSize(
        D3D12_META_COMMAND_PARAMETER_STAGE_EXECUTION, 4);
    size_t temp_size = pMetacommand->GetRequiredParameterResourceSize(
        D3D12_META_COMMAND_PARAMETER_STAGE_EXECUTION, 5);

    if (persistent_size) {
      dx_context->CreateAlloc(persistent_size, D3D12_HEAP_TYPE_DEFAULT,
                              scratch_data_persistent_[i], fp16);
    }

    if (temp_size) {
      dx_context->CreateAlloc(temp_size, D3D12_HEAP_TYPE_DEFAULT,
                              scratch_data_temporary_[i], fp16);
    }

    GemmInitDesc initDesc = {};
    initDesc.PersistentResource =
        scratch_data_persistent_[i].desc_handle_scalar;
    initDesc.TemporaryResource = scratch_data_temporary_[i].desc_handle_scalar;

    dx_context->getCommandList()->InitializeMetaCommand(
        meta_commands_[i], &initDesc, sizeof(initDesc));
  }

  create_succeeded_ = true;
}

void GemmMetaCommand::PerformGemm(int rows, DXAlloc A, DXAlloc B,
                                  DXAlloc output,
                                  ID3D12GraphicsCommandList4* command_list) {
  if (!create_succeeded_) throw Exception("Metacommand not created");

  int index = 0;
  if (!rows_known_) {
    index = DivUp(rows, 8) - 1;
  }

  ID3D12MetaCommand* meta_command = meta_commands_[index];
  DXAlloc& scratch_persistent = scratch_data_persistent_[index];
  DXAlloc& scratch_temporary = scratch_data_temporary_[index];

  GemmExecuteDesc exec_desc = {};
  exec_desc.AResource = A.desc_handle_scalar;
  exec_desc.BResource = B.desc_handle_scalar;
  exec_desc.OutputResource = output.desc_handle_scalar;
  exec_desc.PersistentResource = scratch_persistent.desc_handle_scalar;
  exec_desc.TemporaryResource = scratch_temporary.desc_handle_scalar;

  command_list->ExecuteMetaCommand(meta_command, &exec_desc, sizeof(exec_desc));
}

GemmMetaCommand::~GemmMetaCommand() {
  for (int i = 0; i < kMaxMetacommands; i++) {
    if (scratch_data_temporary_[i].resource)
      scratch_data_temporary_[i].resource->Release();
    if (scratch_data_persistent_[i].resource)
      scratch_data_persistent_[i].resource->Release();
    if (meta_commands_[i]) meta_commands_[i]->Release();
  }
}

ConvMetaCommand::ConvMetaCommand(DxContext* dx_context, int C, int K, int H,
                                 int W, int F, bool relu, bool bias,
                                 bool fp16) {
  memset(scratch_data_persistent_, 0, sizeof(scratch_data_persistent_));
  memset(scratch_data_temporary_, 0, sizeof(scratch_data_temporary_));
  memset(meta_commands_, 0, sizeof(meta_commands_));

  for (int i = 0; i < kMaxMetacommands; i++) {
    int n = (i + 1) * kMetacommandGranulity;

    ConvCreateDesc createDesc = {};
    GetConvTensorDesc(&createDesc.InputDesc, n, C, H, W, fp16);
    GetConvTensorDesc(&createDesc.OutputDesc, n, K, H, W, fp16);
    GetConvTensorDesc(&createDesc.FilterDesc, K, C, F, F, fp16);
    GetConvTensorDesc(&createDesc.BiasDesc, K, 1, 1, 1, fp16);
    createDesc.BiasNull = bias ? 0 : 1;
    createDesc.Mode = 1;  // 1 is for cross-correlation (0 - conv)

    createDesc.Direction = 0;       // forward
    createDesc.DimensionCount = 2;  // 2D conv
    createDesc.Stride[0] = 1;
    createDesc.Stride[1] = 1;
    createDesc.Dilation[0] = 1;
    createDesc.Dilation[1] = 1;

    int pad = (F - 1) / 2;
    createDesc.StartPadding[0] = pad;
    createDesc.StartPadding[1] = pad;
    createDesc.EndPadding[0] = pad;
    createDesc.EndPadding[1] = pad;
    createDesc.GroupCount = 1;
    if (relu) {
      createDesc.ActivationFunction = 9;  // relu (guess?)
      createDesc.ActivationIsNull = 0;
    } else {
      createDesc.ActivationIsNull = 1;
    }
    createDesc.Precision = fp16 ? 1 : 0;  // 0 - fp32, 1 - fp16

    ID3D12MetaCommand* pMetacommand = nullptr;
    HRESULT hr = dx_context->getDevice()->CreateMetaCommand(
        ConvGuid, 1, &createDesc, sizeof(createDesc),
        IID_PPV_ARGS(&pMetacommand));

    if (hr != S_OK) {
#ifdef DEBUG_DUMP_PER_LAYER_DATA
      printf(
          "\nCan't create Conv Metacommand for "
          "N, C, K, H, W, f: %d %d %d %d %d %d\n",
          n, C, K, H, W, F);
#endif
      create_succeeded_ = false;
      return;
    }

    meta_commands_[i] = pMetacommand;

    size_t persistent_size = pMetacommand->GetRequiredParameterResourceSize(
        D3D12_META_COMMAND_PARAMETER_STAGE_EXECUTION, 4);
    size_t temp_size = pMetacommand->GetRequiredParameterResourceSize(
        D3D12_META_COMMAND_PARAMETER_STAGE_EXECUTION, 5);

    if (persistent_size) {
      dx_context->CreateAlloc(persistent_size, D3D12_HEAP_TYPE_DEFAULT,
                              scratch_data_persistent_[i], fp16);
    }

    if (temp_size) {
      dx_context->CreateAlloc(temp_size, D3D12_HEAP_TYPE_DEFAULT,
                              scratch_data_temporary_[i], fp16);
    }

    InitConvDesc initDesc = {};
    initDesc.PersistentResource =
        scratch_data_persistent_[i].desc_handle_scalar;
    initDesc.TemporaryResource = scratch_data_temporary_[i].desc_handle_scalar;

    dx_context->getCommandList()->InitializeMetaCommand(
        meta_commands_[i], &initDesc, sizeof(initDesc));
  }
  use_bias_ = bias;
  create_succeeded_ = true;
}

void ConvMetaCommand::PerformConv(int batch, DXAlloc input, DXAlloc filter,
                                  DXAlloc bias, DXAlloc output,
                                  ID3D12GraphicsCommandList4* command_list) {
  if (!create_succeeded_) throw Exception("Metacommand not created");

  int index = DivUp(batch, 8) - 1;

  ID3D12MetaCommand* meta_command = meta_commands_[index];
  DXAlloc& scratch_persistent = scratch_data_persistent_[index];
  DXAlloc& scratch_temporary = scratch_data_temporary_[index];

  ExecuteConvDesc exec_desc = {};
  exec_desc.InputResource = input.desc_handle_scalar;
  exec_desc.FilterResource = filter.desc_handle_scalar;
  if (use_bias_) exec_desc.BiasResource = bias.desc_handle_scalar;
  exec_desc.OutputResource = output.desc_handle_scalar;
  exec_desc.PersistentResource = scratch_persistent.desc_handle_scalar;
  exec_desc.TemporaryResource = scratch_temporary.desc_handle_scalar;

  command_list->ExecuteMetaCommand(meta_command, &exec_desc, sizeof(exec_desc));
}

ConvMetaCommand::~ConvMetaCommand() {
  for (int i = 0; i < kMaxMetacommands; i++) {
    if (scratch_data_temporary_[i].resource)
      scratch_data_temporary_[i].resource->Release();
    if (scratch_data_persistent_[i].resource)
      scratch_data_persistent_[i].resource->Release();
    if (meta_commands_[i]) meta_commands_[i]->Release();
  }
}

BaseLayer::BaseLayer(int c, int h, int w, BaseLayer* ip, DxContext* dx_context,
                     bool fp16)
    : input_(ip), C(c), H(h), W(w), dx_context_(dx_context), fp16_(fp16) {}

ConvLayer::ConvLayer(bool fp16, GemmMetaCommand* pMetaCommandGemm,
                     ConvMetaCommand* pMetaCommandConv, DxContext* dx_context,
                     BaseLayer* ip, int C, int H, int W, int filter, int Cin,
                     bool bias, bool relu, bool skipAdd, bool se, int se_k)
    : BaseLayer(C, H, W, ip, dx_context, fp16),
      meta_command_gemm_(pMetaCommandGemm),
      meta_command_conv_(pMetaCommandConv),
      c_input_(Cin),
      filter_size_(filter),
      use_relu_(relu),
      use_bias_(bias),
      skip_add_(skipAdd),
      has_se_(se),
      se_k_(se_k),
      weights_(),
      transformed_weights_(),
      biases_(),
      w1_(),
      w2_(),
      b1_(),
      b2_() {
  size_t element_size = fp16 ? sizeof(dx_half) : sizeof(float);
  size_t weight_size = element_size * C * Cin * filter * filter;
  size_t blas_size = element_size * C;

  dx_context->CreateAlloc(weight_size, D3D12_HEAP_TYPE_DEFAULT, weights_, fp16);

  if (filter == 3) {
    // 6x6 transformed filter size, for 3x3 convolution
    dx_context->CreateAlloc(weight_size * 4, D3D12_HEAP_TYPE_DEFAULT,
                            transformed_weights_, fp16);
  }

  if (use_bias_) {
    dx_context->CreateAlloc(blas_size, D3D12_HEAP_TYPE_DEFAULT, biases_, fp16);
  }

  if (has_se_) {
    const size_t num_weights1 = C * se_k_;
    const size_t num_weights2 = num_weights1 * 2;
    const size_t num_biases1 = se_k_;
    const size_t num_biases2 = 2 * C;

    const size_t weight_size1 = element_size * num_weights1;
    const size_t weight_size2 = element_size * num_weights2;
    const size_t biases_size1 = element_size * num_biases1;
    const size_t biases_size2 = element_size * num_biases2;

    dx_context->CreateAlloc(weight_size1, D3D12_HEAP_TYPE_DEFAULT, w1_, fp16);
    dx_context->CreateAlloc(weight_size2, D3D12_HEAP_TYPE_DEFAULT, w2_, fp16);
    dx_context->CreateAlloc(biases_size1, D3D12_HEAP_TYPE_DEFAULT, b1_, fp16);
    dx_context->CreateAlloc(biases_size2, D3D12_HEAP_TYPE_DEFAULT, b2_, fp16);
  }

  shader_wrapper_ = dx_context->getShaderWrapper();
}

void ConvLayer::LoadWeights(float* cpu_filter, float* cpu_bias,
                            DxContext* dx_context) {
  int num_weights = c_input_ * C * filter_size_ * filter_size_;
  size_t element_size = fp16_ ? sizeof(dx_half) : sizeof(float);
  size_t weight_size = element_size * num_weights;
  size_t bias_size = element_size * C;

  std::vector<dx_half> temp(num_weights);
  if (fp16_) {
    CopyFloatToHalf(temp.data(), cpu_filter, num_weights);
    dx_context->ScheduleUpload(weights_, temp.data(), weight_size);
  } else {
    dx_context->ScheduleUpload(weights_, cpu_filter, weight_size);
  }

  if (cpu_bias) {
    if (fp16_) {
      CopyFloatToHalf(temp.data(), cpu_bias, C);
      dx_context->ScheduleUpload(biases_, temp.data(), bias_size);
    } else {
      dx_context->ScheduleUpload(biases_, cpu_bias, bias_size);
    }
  }

  if (filter_size_ == 3) {
    std::vector<float> temp_transformed(num_weights * 4);
    TransformFilterTensor_Winograd4x4(C, c_input_, temp_transformed.data(),
                                      cpu_filter);
    if (fp16_) {
      std::vector<dx_half> temp_transformed_half(num_weights * 4);
      CopyFloatToHalf(temp_transformed_half.data(), temp_transformed.data(),
                      num_weights * 4);
      dx_context->ScheduleUpload(transformed_weights_,
                                 temp_transformed_half.data(), weight_size * 4);
    } else {
      dx_context->ScheduleUpload(transformed_weights_, temp_transformed.data(),
                                 weight_size * 4);
    }
  }
}

void ConvLayer::LoadSEWeights(float* w1, float* b1, float* w2, float* b2) {
  size_t element_size = fp16_ ? sizeof(dx_half) : sizeof(float);
  const size_t num_weights1 = C * se_k_;
  const size_t num_weights2 = num_weights1 * 2;
  const size_t weight_size1 = element_size * num_weights1;
  const size_t weight_size2 = element_size * num_weights2;

  const size_t num_biases1 = se_k_;
  const size_t biases_size1 = element_size * num_biases1;
  const size_t num_biases2 = 2 * C;
  const size_t biases_size2 = element_size * num_biases2;

  // The shader uses transposed weight matrices.

  std::vector<float> temp_transposed(num_weights2);
  std::vector<dx_half> temp_half(num_weights2);

  CpuTranspose(temp_transposed.data(), w1, se_k_, C);
  if (fp16_) {
    CopyFloatToHalf(temp_half.data(), temp_transposed.data(), num_weights1);
    dx_context_->ScheduleUpload(w1_, temp_half.data(), weight_size1);
  } else {
    dx_context_->ScheduleUpload(w1_, temp_transposed.data(), weight_size1);
  }

  CpuTranspose(temp_transposed.data(), w2, 2 * C, se_k_);
  if (fp16_) {
    CopyFloatToHalf(temp_half.data(), temp_transposed.data(), num_weights2);
    dx_context_->ScheduleUpload(w2_, temp_half.data(), weight_size2);
  } else {
    dx_context_->ScheduleUpload(w2_, temp_transposed.data(), weight_size2);
  }

  if (fp16_) {
    CopyFloatToHalf(temp_half.data(), b1, num_biases1);
    dx_context_->ScheduleUpload(b1_, temp_half.data(), biases_size1);
  } else {
    dx_context_->ScheduleUpload(b1_, b1, biases_size1);
  }

  if (fp16_) {
    CopyFloatToHalf(temp_half.data(), b2, num_biases2);
    dx_context_->ScheduleUpload(b2_, temp_half.data(), biases_size2);
  } else {
    dx_context_->ScheduleUpload(b2_, b2, biases_size2);
  }
}

void ConvLayer::Eval(int N, DXAlloc output, DXAlloc input, DXAlloc input2,
                     DXAlloc scratch, DXAlloc scratch2,
                     ID3D12GraphicsCommandList4* command_list) {
  // Use winograd for filter size of 3, when GEMM metacommand is available,
  // Or when GEMM metacommand isn't available but Convolution metacommand is
  // also not available (compute shader matrix multiply path).
  bool useWinograd =
      (filter_size_ == 3) &&
      ((meta_command_gemm_ && meta_command_gemm_->IsAvailable()) ||
       !meta_command_conv_ || !meta_command_conv_->IsAvailable());

  if (useWinograd) {
    // Need to pad up the input to gemm too (i.e, the transformed Input tensor)!
    // It's in HWNC layout, and 'N'/GemmN needs to be padded up (HW = 6x6)
    // to make it simple, just pad up N to multiple of 2 here (so that gemmN is
    // multiple of 8).
    // TODO: figure out why padding up by 4 is needed (instead of 2!)
    // N = ((N + 1) / 2) * 2;
    N = ((N + 3) / 4) * 4;

    // 1. Input transform (input->scratch)
    shader_wrapper_->InputTransform(command_list, scratch, input, N, c_input_,
                                    fp16_);

    dx_context_->UavBarrier(command_list);

    // 2. Gemm (scratch -> scratch2)
    if (meta_command_gemm_ && meta_command_gemm_->IsAvailable())
      meta_command_gemm_->PerformGemm(N * 4, scratch, transformed_weights_,
                                      scratch2, command_list);
    else
      shader_wrapper_->MatrixMultiply(command_list, scratch2, scratch,
                                      transformed_weights_, N * 4, C, c_input_,
                                      36, fp16_);

    dx_context_->UavBarrier(command_list);

    // 3. Output transform (scratch2 -> output)
    shader_wrapper_->OutputTransform(
        command_list, output, scratch2, input2, biases_, w1_, b1_, w2_, b2_, N,
        C, use_relu_, use_bias_, skip_add_, has_se_, se_k_, fp16_);

  } else if (meta_command_conv_ && meta_command_conv_->IsAvailable()) {
    if (skip_add_ || has_se_)
      meta_command_conv_->PerformConv(N, input, weights_, biases_, scratch,
                                      command_list);
    else
      meta_command_conv_->PerformConv(N, input, weights_, biases_, output,
                                      command_list);
    if (has_se_) {
      dx_context_->UavBarrier(command_list);
      shader_wrapper_->Se(command_list, output, scratch, input2, biases_, w1_,
                          b1_, w2_, b2_, N, C, use_relu_, false, skip_add_,
                          se_k_, fp16_);
    } else if (skip_add_) {
      // Need seperate pass for skip connection addition as Metacommand API
      // doesn't allow it to be fused with convolution.
      dx_context_->UavBarrier(command_list);
      shader_wrapper_->AddVectors(command_list, output, scratch, input2,
                                  N * C * H * W, N * C * H * W, N * C * H * W,
                                  use_relu_, false, fp16_);
    }
  } else if (filter_size_ == 1) {
    shader_wrapper_->Conv1x1(command_list, output, input, weights_, biases_, N,
                             c_input_, C, use_relu_, use_bias_, fp16_);
  } else {
    throw Exception("Unsupported filter shape for convolution! ");
  }
}

ConvLayer::~ConvLayer() {
  if (weights_.resource) weights_.resource->Release();
  if (biases_.resource) biases_.resource->Release();
  if (transformed_weights_.resource) transformed_weights_.resource->Release();

  if (w1_.resource) w1_.resource->Release();
  if (w2_.resource) w2_.resource->Release();
  if (b1_.resource) b1_.resource->Release();
  if (b2_.resource) b2_.resource->Release();
}

FCLayer::FCLayer(bool fp16, DxContext* dx_context, BaseLayer* ip, int C, int H,
                 int W, bool bias, bool relu, bool tanh)
    : BaseLayer(C, H, W, ip, dx_context, fp16),
      use_bias_(bias),
      use_relu_(relu),
      meta_command_(),
      use_tanh_(tanh) {
  size_t element_size = fp16_ ? sizeof(dx_half) : sizeof(float);
  size_t weight_size =
      element_size * C * H * W * ip->GetC() * ip->GetH() * ip->GetW();
  size_t blas_size = element_size * C * H * W;

  dx_context->CreateAlloc(weight_size, D3D12_HEAP_TYPE_DEFAULT, weights_, fp16);
  if (use_bias_)
    dx_context->CreateAlloc(blas_size, D3D12_HEAP_TYPE_DEFAULT, biases_, fp16);

  shader_wrapper_ = dx_context->getShaderWrapper();

  // Create metacommand object
  int rows = 0;                                  // batch size
  int cols = C * H * W;                          // cols of the output matrix
  int K = ip->GetC() * ip->GetH() * ip->GetW();  // cols of input matrix
  // We do Out = A * weight.
  // The weight matrix need to be transpsoed before it can be multiplied.
  // The transpose is done on CPU when loading weights
  meta_command_ = std::make_unique<GemmMetaCommand>(dx_context, rows, cols, K,
                                                    1, fp16, false, false);
}

void FCLayer::LoadWeights(float* cpuWeight, float* cpuBias,
                          DxContext* dx_context) {
  size_t rows = C * H * W;
  size_t cols = input_->GetC() * input_->GetH() * input_->GetW();
  size_t num_weights = rows * cols;

  size_t element_size = fp16_ ? sizeof(dx_half) : sizeof(float);
  size_t weight_size = element_size * num_weights;
  size_t num_biases = C * H * W;
  size_t bias_size = element_size * num_biases;

  std::vector<float> temp_transposed(num_weights);
  CpuTranspose(temp_transposed.data(), cpuWeight, rows, cols);
  std::vector<dx_half> temp(num_weights);
  if (fp16_) {
    CopyFloatToHalf(temp.data(), temp_transposed.data(), num_weights);
    dx_context->ScheduleUpload(weights_, temp.data(), weight_size);
  } else {
    dx_context->ScheduleUpload(weights_, temp_transposed.data(), weight_size);
  }

  if (cpuBias) {
    if (fp16_) {
      CopyFloatToHalf(temp.data(), cpuBias, C);
      dx_context->ScheduleUpload(biases_, temp.data(), bias_size);
    } else {
      dx_context->ScheduleUpload(biases_, cpuBias, bias_size);
    }
  }
}

void FCLayer::Eval(int N, DXAlloc output, DXAlloc input, DXAlloc /*input2*/,
                   DXAlloc /*scratch*/, DXAlloc /*scratch2*/,
                   ID3D12GraphicsCommandList4* command_list) {
  int num_outputs = C * H * W;
  int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

  if (meta_command_->IsAvailable())
    meta_command_->PerformGemm(N, input, weights_, output, command_list);
  else
    shader_wrapper_->MatrixMultiply(command_list, output, input, weights_,
                                    DivUp(N, 8) * 8, num_outputs, num_inputs, 1,
                                    fp16_);

  if (use_bias_ || use_relu_ || use_tanh_) {
    dx_context_->UavBarrier(command_list);
    shader_wrapper_->AddVectors(command_list, output, output, biases_,
                                N * num_outputs, N * num_outputs, num_outputs,
                                use_relu_, use_tanh_, fp16_);
  }
}

FCLayer::~FCLayer() {
  if (weights_.resource) weights_.resource->Release();
  if (biases_.resource) biases_.resource->Release();
}

PolicyMapLayer::PolicyMapLayer(bool fp16, DxContext* dx_context, BaseLayer* ip,
                               int C, int H, int W, int usedSize)
    : BaseLayer(C, H, W, ip, dx_context, fp16), used_size_(usedSize) {
  size_t weight_size = sizeof(int) * used_size_;
  dx_context->CreateAlloc(weight_size, D3D12_HEAP_TYPE_DEFAULT, weights_, fp16);
}

void PolicyMapLayer::LoadWeights(const short* cpuWeights) {
  // convert from short to int (as HLSL might have trouble reading short)
  std::vector<int> temp(used_size_);
  for (int i = 0; i < used_size_; i++) temp[i] = (int)cpuWeights[i];
  dx_context_->ScheduleUpload(weights_, temp.data(), sizeof(int) * used_size_);
}

void PolicyMapLayer::Eval(int N, DXAlloc output, DXAlloc input,
                          DXAlloc /*input2*/, DXAlloc /*scratch*/,
                          DXAlloc /*scratch2*/,
                          ID3D12GraphicsCommandList4* command_list) {
  int inputSize =
      this->input_->GetC() * this->input_->GetH() * this->input_->GetW();
  int outputSize = this->C * this->H * this->W;
  dx_context_->getShaderWrapper()->PolicyMap(command_list, output, input,
                                             weights_, N, inputSize, outputSize,
                                             used_size_, fp16_);
}

PolicyMapLayer::~PolicyMapLayer() {
  if (weights_.resource) weights_.resource->Release();
}

void DxError(HRESULT status, const char* file, const int& line) {
  if (FAILED(status)) {
    assert(0);
    char message[512];
    _com_error err(status);
    LPCTSTR errMsg = err.ErrorMessage();
    sprintf_s(message, "Dx error: %s (%s:%d) ", errMsg, file, line);
    throw Exception(message);
  }
}

}  // namespace dx_backend
}  // namespace lczero
