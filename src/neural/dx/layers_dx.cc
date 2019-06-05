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
#include "layers_dx.h"
#include <cassert>
#include <cstring>
#include <vector>
#include "comdef.h"
#include "utils/exception.h"

#include "network_dx.h"
#include "MetaCommand.h"

namespace lczero {
namespace dx_backend {

// for testing
size_t totalScratchSpace = 0;

void convertFp32NCHWtoFp16NHWC(dx_half* out, const float* in, int N, int C,
                               int H, int W) {
  int outIndex = 0;
  for (int n = 0; n < N; n++)
    for (int h = 0; h < H; h++)
      for (int w = 0; w < W; w++)
        for (int c = 0; c < C; c++) {
          int inIndex = n * C * H * W + c * H * W + h * W + w;
          out[outIndex++] = FP32toFP16(in[inIndex]);
        }
}

static void getTensorDesc(TensorDesc* outDesc, int n, int c,
                          int h, int w, bool fp16 = true, bool nhwc = true) {
  memset(outDesc, 0, sizeof(TensorDesc));
  outDesc->DimensionCount = 4;
  outDesc->DataType = fp16 ? 1 : 0;

  outDesc->Size[0] = n;
  outDesc->Size[1] = c;
  outDesc->Size[2] = h;
  outDesc->Size[3] = w;

  if (nhwc) {
    outDesc->Stride[1] = 1;
    outDesc->Stride[3] = c;
    outDesc->Stride[2] = c * w;
    outDesc->Stride[0] = c * w * h;
  } else {
    outDesc->Stride[3] = 1;
    outDesc->Stride[2] = w;
    outDesc->Stride[1] = w * h;
    outDesc->Stride[0] = c * w * h;
  }

  for (int i = 0; i < 4; i++)
    outDesc->StrideAlignment[i] = 1;

  outDesc->BaseAlignmentInBytes = 4096;
  outDesc->PhysicalSizeInElements = n * c * h * w;
}

ConvMetaCommand::ConvMetaCommand(DxContext* pContext, int Cin, int Cout, int H,
                                 int W, int filterSize, bool skipAdd,
                                 bool hasBias, bool hasRelu) {
  if (skipAdd) hasRelu = false;  // relu done after skip addition

  for (int i = 0; i < kMaxSupportedBatchSize; i++) {
    int n = i + 1;

    ConvCreateDesc desc = {};

    getTensorDesc(&desc.InputDesc, n, Cin, H, W);
    getTensorDesc(&desc.OutputDesc, n, Cout, H, W);    
    getTensorDesc(&desc.FilterDesc, Cout, Cin, filterSize, filterSize);    
    getTensorDesc(&desc.BiasDesc, Cout, 1, 1, 1);    
    desc.BiasNull = hasBias ? 0 : 1;
    desc.Mode = 1;  // 1 is for cross-correlation (0 - conv)

    desc.Direction = 0; // forward
    desc.DimensionCount = 2;  // 2D conv
    desc.Stride[0] = 1;
    desc.Stride[1] = 1;
    desc.Dilation[0] = 1;
    desc.Dilation[1] = 1;

    int pad = (filterSize - 1) / 2;
    desc.StartPadding[0] = pad;
    desc.StartPadding[1] = pad;
    desc.EndPadding[0] = pad;
    desc.EndPadding[1] = pad;
    desc.GroupCount = 1;
    if (hasRelu) {
      desc.ActivationFunction = 9;    // relu (guess?)
      desc.ActivationIsNull = 0;
    } else {
      desc.ActivationIsNull = 1;
    }
    desc.Precision = 1; // fp16

    int paramSize = sizeof(desc);

    HRESULT hr = pContext->getDevice()->CreateMetaCommand(
        ConvGuid, 0, &desc, sizeof(desc),
        IID_PPV_ARGS(&pMetaCommands[i]));

    if (hr != S_OK) {
      throw Exception("Error creating convolution Metacommand\n");
    }

    size_t sizeInBytes = 0;

    sizeInBytes = pMetaCommands[i]->GetRequiredParameterResourceSize(
    D3D12_META_COMMAND_PARAMETER_STAGE_INITIALIZATION,
    3 /*index of persistent resource in init desc*/);

    // TODO: Consider creating a single allocation with chunks suballocated for
    // each metacommand object
    if (sizeInBytes) {
      totalScratchSpace += sizeInBytes;
      printf(
          "allocating %llu bytes for persistent metacommand storage, total: "
          "%llu\n",
          sizeInBytes, totalScratchSpace);
      pContext->CreateAlloc(sizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
                            &scratch_data_[i]);
    } else {
      scratch_data_[i].pResource = nullptr;
      scratch_data_[i].gpuVA = 0;
    }

    InitConvDesc initDesc = {};
    if (sizeInBytes) initDesc.PersistentResource = scratch_data_[i].descHandle;

    pContext->getCommandList()->InitializeMetaCommand(
        pMetaCommands[i], &initDesc, sizeof(initDesc));
  }
}

ConvMetaCommand::~ConvMetaCommand() {
  for (int i = 0; i < kMaxSupportedBatchSize; i++) {
    scratch_data_[i].pResource->Release();
    pMetaCommands[i]->Release();
  }
}

BaseLayer::BaseLayer(int c, int h, int w, BaseLayer* ip, DxContext* pContext)
    : input_(ip), C(c), H(h), W(w), dx_context_(pContext) {}

ConvLayer::ConvLayer(ConvMetaCommand* pMetaCommand, DxContext* pContext,
                     BaseLayer* ip, int C, int H, int W, int filter, int Cin,
                     bool relu, bool bias, bool skipAdd)
    : BaseLayer(C, H, W, ip, pContext),
      meta_command_(pMetaCommand),
      c_input_(Cin),
      filter_size_(filter),
      use_relu_(relu),
      use_bias_(bias),
      skip_add_(skipAdd),
      weights_(nullptr),
      biases_(nullptr) {
  size_t weight_size =
      sizeof(dx_half) * c_input_ * C * filter_size_ * filter_size_;
  size_t blas_size = sizeof(dx_half) * C;

  weights_ = new DXAlloc;
  pContext->CreateAlloc(weight_size, D3D12_HEAP_TYPE_DEFAULT, weights_);

  if (use_bias_) {
    biases_ = new DXAlloc;
    pContext->CreateAlloc(blas_size, D3D12_HEAP_TYPE_DEFAULT, biases_);
  }
}

void cpuTranspose(dx_half* op, dx_half* ip, int rows, int cols) {
  printf("\ntranspoising to H W = %d %d\n", rows, cols);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) op[j * rows + i] = ip[i * cols + j];
}

void ConvLayer::LoadWeights(float* pfilter, float* pBias, DxContext* pContext) {
  int num_weights = c_input_ * C * filter_size_ * filter_size_;
  size_t weight_size = sizeof(dx_half) * num_weights;
  size_t bias_size = sizeof(dx_half) * C;

  std::vector<dx_half> temp(num_weights);
  convertFp32NCHWtoFp16NHWC(temp.data(), pfilter, C, c_input_, filter_size_,
                            filter_size_);
  pContext->scheduleUpload(*weights_, temp.data(), weight_size);

  if (pBias) {
    convertFp32NCHWtoFp16NHWC(temp.data(), pBias, C, 1, 1, 1);
    pContext->scheduleUpload(*biases_, temp.data(), bias_size);
  }
}

void ConvLayer::Eval(int N, dx_alloc_handle output, dx_alloc_handle input,
                     dx_alloc_handle input2, dx_command_stream cmdStream) {
  ExecuteConvDesc desc = {};

  desc.InputResource = input->descHandle;
  desc.OutputResource = output->descHandle;
  desc.FilterResource = weights_->descHandle;
  if (use_bias_) desc.BiasResource = biases_->descHandle;
  desc.PersistentResource = meta_command_->getScratchHandle(N);

  if (input2) {
    assert(skip_add_);
    // arbitary input2 not supported by dx path
    // assert(input2->gpuVA == output->gpuVA);

    desc.OutputResource = dx_context_->getDefaultScratch()->descHandle;
  }

  cmdStream->ExecuteMetaCommand(meta_command_->getMetaCommand(N), &desc, sizeof(desc));

  // Ankan - test!
  //dx_context_->dumpTensor(*output, 1024);
  //exit(0);

  if (input2) {
    dx_context_->getCommandList()->ResourceBarrier(
        1, &CD3DX12_RESOURCE_BARRIER::UAV(
               dx_context_->getDefaultScratch()->pResource));

    //dx_context_->dumpTensor(*output, 1024);
    //dx_context_->dumpTensor(*dx_context_->getDefaultScratch(), 1024);
    //exit(0);

    dx_context_->getShaderWrapper()->skipAddRelu(
        cmdStream, *dx_context_->getDefaultScratch(), *input2, *output, true,
        N * C * 64);

  }
}

ConvLayer::~ConvLayer() {
  weights_->pResource->Release();
  delete weights_;
  weights_ = nullptr;
  if (biases_) {
    biases_->pResource->Release();
    delete biases_;
    biases_ = nullptr;
  }
}

FCLayer::FCLayer(DxContext* pContext, BaseLayer* ip, int C, int H, int W,
                 bool bias, bool relu, bool tanh, bool softmax, bool fp32Out)
    : BaseLayer(C, H, W, ip, pContext),
      use_bias_(bias),
      use_relu_(relu),
      use_tanh_(tanh),
      use_softmax_(softmax) {
  size_t weight_size =
      sizeof(dx_half) * C * H * W * ip->GetC() * ip->GetH() * ip->GetW();
  size_t blas_size = sizeof(float) * C * H * W;  // biases are in fp32

  weights_ = new DXAlloc;
  pContext->CreateAlloc(weight_size, D3D12_HEAP_TYPE_DEFAULT, weights_);

  if (use_bias_) {
    biases_ = new DXAlloc;
    pContext->CreateAlloc(blas_size, D3D12_HEAP_TYPE_DEFAULT, biases_);
  } else {
    biases_ = nullptr;
  }
}

void FCLayer::LoadWeights(float* cpuWeight, float* cpuBias,
                          DxContext* pContext) {
  shader_wrapper_ = pContext->getShaderWrapper();

  size_t num_weights =
      C * H * W * input_->GetC() * input_->GetH() * input_->GetW();
  size_t weight_size = sizeof(dx_half) * num_weights;
  size_t num_biases = C * H * W;
  size_t bias_size = sizeof(float) * num_biases;

  std::vector<dx_half> scratch(num_weights);
  std::vector<dx_half> temp(num_weights);

  // Observe the way FC layer weights need to be converted.
  convertFp32NCHWtoFp16NHWC(scratch.data(), cpuWeight, num_biases,
                            input_->GetC(), input_->GetH(), input_->GetW());

  // transpose weight matrix so that matrix multiply is faster
  cpuTranspose(temp.data(), scratch.data(), C, num_weights / C);
  pContext->scheduleUpload(*weights_, temp.data(), weight_size);

  if (cpuBias) {
    // no conversion! plain copy
    pContext->scheduleUpload(*biases_, cpuBias, bias_size);
  }
}

void FCLayer::Eval(int N, dx_alloc_handle output, dx_alloc_handle input,
                   dx_alloc_handle input2, dx_command_stream cmdStream) {
  int num_outputs = C * H * W;
  int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

  if (use_softmax_) {
    // if (N == 256) return;  // Ankan - bad test!

    // The shader has these hardcoded.
    assert(num_outputs == 1858);
    assert(num_inputs == 2048);
    shader_wrapper_->policyFC(cmdStream, *output, *input, *weights_, *biases_,
                              N);

  } else if (num_outputs == 1) {
    // FC2 of value head.
    // The shader has this hardcoded.
    assert(num_inputs == 128);
    shader_wrapper_->valueFC2(cmdStream, *output, *input, *weights_, *biases_,
                              N);
  } else {
    // FC1 of value head.
    // The shader has these hardcoded.
    assert(num_outputs == 128);
    assert(num_inputs == 2048);
    shader_wrapper_->valueFC1(cmdStream, *output, *input, *weights_, *biases_,
                              N);
  }
}

FCLayer::~FCLayer() {
  weights_->pResource->Release();
  delete weights_;
  weights_ = nullptr;
  if (biases_) {
    biases_->pResource->Release();
    delete biases_;
    biases_ = nullptr;
  }
}

// misc error handling stuff
/*
inline char *GetMessageForHresult(HRESULT hr) {
  _com_error error(hr);
  return error.ErrorMessage();
}
*/

void DxError(HRESULT status, const char* file, const int& line) {
  if (FAILED(status)) {
    assert(0);
    char message[128];
    sprintf(message, "Dx error: %s (%s:%d) ", "generic dx error", file, line);
    throw Exception(message);
  }
}

}  // namespace dx_backend
}  // namespace lczero
