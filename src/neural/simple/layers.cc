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

#include "winograd_filter.h"
#include "winograd_convolution3.h"
#include "activation.h"
#include "convolution1.h"
#include "fully_connected_layer.h"
#include "se_unit.h"
#include "simple_common.h"

#include "layers.h"

#include "utils/exception.h"

#include <cstddef>
#include <cassert>
#include <cstring>
#include <vector>

namespace lczero {
namespace simple_backend {

BaseLayer::BaseLayer(int c, int h, int w, BaseLayer* ip)
    : input_(ip), C(c), H(h), W(w) {}

ConvLayer::ConvLayer(BaseLayer* ip, int C, int H, int W, int filter, int Cin,
                     bool relu, bool skip)
    : BaseLayer(C, H, W, ip),
      c_input_(Cin),
      filter_size_(filter),
      use_relu_(relu),
      use_skip_(skip) {
  biases = (float*)malloc(sizeof(float) * C);

  if (filter_size_ == 1) {
    weights = (float*)malloc(C * c_input_ * sizeof(float));
    return;
  }
  if (filter_size_ == 3) {
    constexpr auto kWinogradAlpha = 4;
    constexpr auto kWinogradTile = kWinogradAlpha * kWinogradAlpha;

    weights = (float*)malloc(kWinogradTile * C * c_input_ * sizeof(float));
    return;
  }

  throw Exception("Unsupported filter_size.");
}

void ConvLayer::LoadWeights(float* pfilter, float* pBias, void* /*scratch*/) {
  memcpy(biases, pBias, sizeof(float) * C);

  if (filter_size_ == 1) {
    memcpy(weights, pfilter, sizeof(float) * C * c_input_);
    return;
  }
  if (filter_size_ == 3) {
    WinogradFilterTransformF(weights, pfilter, C, c_input_);
    return;
  }
}

void ConvLayer::Eval(int N, float* output, const float* input, void* scratch,
                     size_t /*scratch_size*/) {
  if (filter_size_ == 1) {
    Convolution1::Forward(N, c_input_, C, input, weights,
                          use_skip_ ? (float*)scratch : output);
  } else if (filter_size_ == 3) {
    WinogradConvolution3 convolve3(N, c_input_, C);
    convolve3.Forward(N, c_input_, C, input, weights,
                      use_skip_ ? (float*)scratch : output);
  }
  BiasResidualRelu(N, C, output, biases, use_skip_ ? (float*)scratch : nullptr,
                   use_relu_);
}

ConvLayer::~ConvLayer() {
  free(biases);
  free(weights);
}

SELayer::SELayer(BaseLayer* ip, int fc1Outputs)
    : BaseLayer(ip->GetC(), ip->GetH(), ip->GetW(), ip),
      numFc1Out_(fc1Outputs) {
  w1_ = (float*)malloc(C * numFc1Out_ * sizeof(float));
  w2_ = (float*)malloc(2 * C * numFc1Out_ * sizeof(float));

  b1_ = (float*)malloc(numFc1Out_ * sizeof(float));
  b2_ = (float*)malloc(2 * C * sizeof(float));
}

SELayer::~SELayer() {
  free(w1_);
  free(w2_);
  free(b1_);
  free(b2_);
}

void SELayer::LoadWeights(float* w1, float* b1, float* w2, float* b2,
                          void* /*scratch*/) {
  const size_t num_weights1 = C * numFc1Out_;
  const size_t weight_size1 = sizeof(float) * num_weights1;

  const size_t weight_size2 = 2 * weight_size1;

  // Weight for the first FC layer.
  memcpy(w1_, w1, weight_size1);

  // Weight for the second FC layer.
  memcpy(w2_, w2, weight_size2);

  // Bias for the first FC layer.
  memcpy(b1_, b1, numFc1Out_ * sizeof(float));

  // Bias for the second FC layer.
  memcpy(b2_, b2, 2 * C * sizeof(float));
}

void SELayer::Eval(int N, float* output, const float* input, void* /*scratch*/,
                   size_t /*scratch_size*/) {
  ApplySEUnit(N, C, numFc1Out_, input, output, w1_, b1_, w2_, b2_, output);
}

FCLayer::FCLayer(BaseLayer* ip, int C, int H, int W, bool relu, bool bias,
                 bool tanh)
    : BaseLayer(C, H, W, ip),
      use_bias_(bias),
      use_relu_(relu),
      use_tanh_(tanh) {
  const size_t weight_size =
      sizeof(float) * C * H * W * ip->GetC() * ip->GetH() * ip->GetW();
  const size_t bias_size = sizeof(float) * C * H * W;

  weights_ = (float*)malloc(weight_size);
  if (use_bias_) {
    biases_ = (float*)malloc(bias_size);
  } else {
    biases_ = nullptr;
  }
}

void FCLayer::LoadWeights(float* cpuWeight, float* cpuBias, void* /*scratch*/) {
  const size_t num_weights =
      C * H * W * input_->GetC() * input_->GetH() * input_->GetW();
  const size_t weight_size = sizeof(float) * num_weights;
  const size_t num_biases = C * H * W;
  const size_t bias_size = sizeof(float) * num_biases;

  memcpy(weights_, cpuWeight, weight_size);
  if (use_bias_) {
    memcpy(biases_, cpuBias, bias_size);
  }
}

void FCLayer::Eval(int N, float* output_tensor, const float* input_tensor,
                   void* /*scratch*/, size_t /*scratch_size*/) {
  const int num_outputs = C * H * W;
  const int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();
  FullyConnectedLayer::Forward1D(N, num_inputs, num_outputs, input_tensor,
                                 weights_, biases_, use_relu_, use_tanh_,
                                 output_tensor);
}

FCLayer::~FCLayer() {
  free(weights_);
  free(biases_);
}

PolicyMapLayer::PolicyMapLayer(BaseLayer* ip, int C, int H, int W, int usedSize)
    : BaseLayer(C, H, W, ip), used_size_(usedSize) {
  size_t weight_size = sizeof(short) * used_size_ * 64;

  weights_ = (short*)malloc(weight_size);
}

void PolicyMapLayer::LoadWeights(const short* cpuWeight, void* /*scratch*/) {
  size_t weight_size = sizeof(short) * used_size_;
  memcpy(weights_, cpuWeight, weight_size);
}

void PolicyMapLayer::Eval(int N, float* output_tensor,
                          const float* input_tensor, void* /*scratch*/,
                          size_t /*scratch_size*/) {
  int inputSize =
      this->input_->GetC() * this->input_->GetH() * this->input_->GetW();
  int outputSize = this->C * this->H * this->W;

  for (int batch = 0; batch < N; batch++) {
    for (int i = 0; i < used_size_; i++) {
      auto j = weights_[i];
      if (j >= 0) {
        output_tensor[batch * outputSize + j] =
            input_tensor[batch * inputSize + i];
      }
    }
  }
}

PolicyMapLayer::~PolicyMapLayer() { free(weights_); }

}  // namespace simple_backend
}  // namespace lczero
