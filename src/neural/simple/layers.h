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

namespace lczero {
namespace simple_backend {

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

  virtual void Eval(int N, float* output, const float* input, void* scratch,
                    size_t scratch_size) = 0;

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

  ConvLayer(int C, int H, int W, int size, int Cin, bool relu = false,
            bool bias = false);

  ~ConvLayer();
  void LoadWeights(float* pfilter, float* pBias, void* scratch);
  void Eval(int N, float* output, const float* input, void* scratch,
            size_t scratch_size) override;

 private:
  const int c_input_;
  const int filter_size_;
  const bool use_relu_;
  const bool use_skip_;

  float* biases = nullptr;
  float* weights = nullptr;
};

class FCLayer : public BaseLayer {
 public:
  FCLayer(BaseLayer* ip, int C, int H, int W, bool relu, bool bias,
          bool tanh = false);
  ~FCLayer();

  void LoadWeights(float* cpuWeight, float* cpuBias, void* scratch);
  void Eval(int N, float* output, const float* input, void* scratch,
            size_t scratch_size) override;

 private:
  const bool use_bias_;
  const bool use_relu_;
  const bool use_tanh_;
  float* weights_ = nullptr;
  float* biases_ = nullptr;
};

class PolicyMapLayer : public BaseLayer {
 public:
  PolicyMapLayer(BaseLayer* ip, int C, int H, int W, int usedSize);
  ~PolicyMapLayer();

  void LoadWeights(const short* cpuWeight, void* scratch);
  void Eval(int N, float* output, const float* input, void* scratch,
            size_t scratch_size) override;

 private:
  int used_size_;
  short* weights_ = nullptr;
};

// Fused SE layer:
// (optional bias add +) global avg -> FC1 -> FC2 -> global scale -> add skip
// connection -> RELU.
class SELayer : public BaseLayer {
  using BaseLayer::C;

 public:
  SELayer(BaseLayer* ip, int numFc1Out);
  ~SELayer();

  void LoadWeights(float* w1, float* b1, float* w2, float* b2, void* scratch);

  void Eval(int N, float* output, const float* input, void* scratch,
            size_t scratch_size) override;

 private:
  float* w1_ = nullptr;
  float* b1_ = nullptr;
  float* w2_ = nullptr;
  float* b2_ = nullptr;
  int numFc1Out_;
};

}  // namespace simple_backend
}  // namespace lczero
