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

#include <cstddef>

#include "utils/exception.h"

#include "dnnl.hpp"

namespace lczero {
namespace dnnl_backend {

// The Layer objects only hold memory for weights, biases, etc
// memory for input and output tensors is provided by caller of Eval.

class BaseLayer {
 public:
  int GetC() const { return C; }
  int GetH() const { return H; }
  int GetW() const { return W; }

  BaseLayer(int c, int h, int w, BaseLayer* ip);
  virtual ~BaseLayer() = default;
  size_t GetOutputSize(int N) const { return sizeof(float) * N * C * H * W; }

  virtual void Eval(int N, dnnl::memory& output, dnnl::memory& input,
                    dnnl::engine& eng, dnnl::stream& stream) = 0;

 protected:
  BaseLayer* input_;

  int C;  // Output tensor dimensions.
  int H;
  int W;
};

class ConvLayer : public BaseLayer {
  using BaseLayer::C;
  using BaseLayer::H;
  using BaseLayer::W;
  using BaseLayer::GetC;
  using BaseLayer::GetH;
  using BaseLayer::GetW;

 public:
  ConvLayer(BaseLayer* ip, int C, int H, int W, int size, int Cin,
            bool relu = false, bool skip = false);

  void LoadWeights(float* pfilter, float* pBias, dnnl::engine& eng);
  void Eval(int N, dnnl::memory& output, dnnl::memory& input, dnnl::engine& eng,
            dnnl::stream& stream) override;

 private:
  const int c_input_;
  const int filter_size_;
  const bool use_relu_;
  const bool use_skip_;

  dnnl::memory filter_mem;
  dnnl::memory bias_mem;

  // Cache previous convolution primitive in case the batch size is the same.
  int last_batch_ = 0;
  dnnl::convolution_forward conv_;
  dnnl::memory scratchpad_mem;
  // Cached values to change in/out tensors for best performance.
  dnnl::memory::desc in_md;
  dnnl::memory::desc out_md;
};

class FCLayer : public BaseLayer {
 public:
  FCLayer(BaseLayer* ip, int C, int H, int W, bool relu, bool tanh = false);

  void LoadWeights(float* cpuWeight, float* cpuBias, dnnl::engine& eng);
  void Eval(int N, dnnl::memory& output, dnnl::memory& input, dnnl::engine& eng,
            dnnl::stream& stream) override;

 private:
  const bool use_relu_;
  const bool use_tanh_;

  dnnl::memory filter_mem;
  dnnl::memory bias_mem;

  // Cache previous primitive in case the batch size is the same.
  int last_batch_ = 0;
  dnnl::inner_product_forward fc_;
  // Cached values to change in/out tensors for best performance.
  dnnl::memory::desc in_md;
  dnnl::memory::desc out_md;
};

// Fused SE layer:
// global avg -> FC1 -> FC2 -> global scale -> add skip connection -> RELU.
class SELayer : public BaseLayer {
  using BaseLayer::C;

 public:
  SELayer(BaseLayer* ip, int numFc1Out);

  void LoadWeights(float* w1, float* b1, float* w2, float* b2,
                   dnnl::engine& eng);

  void Eval(int N, dnnl::memory& output, dnnl::memory& input, dnnl::engine& eng,
            dnnl::stream& stream) override;

 private:
  dnnl::memory filter_mem;
  dnnl::memory bias_mem;
  dnnl::memory filter2a_mem;
  dnnl::memory bias2a_mem;
  dnnl::memory filter2b_mem;
  dnnl::memory bias2b_mem;

  int numFc1Out_;

  // Cache previous primitives in case the batch size is the same.
  int last_batch_ = 0;
  dnnl::pooling_forward pooling_;
  dnnl::inner_product_forward fc_;
  dnnl::inner_product_forward fc2a_;
  dnnl::inner_product_forward fc2b_;
  dnnl::binary mul_;
  dnnl::binary add_;
  // Cached values to change tensors for best performance.
  dnnl::memory::desc pool_out_md;
  dnnl::memory::desc fc1_in_md;
  dnnl::memory::desc fc1_out_md;
  dnnl::memory::desc fc2a_out_md;
  dnnl::memory::desc fc2b_out_md;
};

}  // namespace dnnl_backend
}  // namespace lczero
