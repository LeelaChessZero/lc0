/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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

#include <string>
#include <vector>

namespace lczero {
namespace metal_backend {

struct Activations {
  std::string default_activation = "relu";
  std::string smolgen_activation = "swish";
  std::string ffn_activation = "relu_2";
};

class MetalNetworkBuilder {
 public:
  MetalNetworkBuilder(void);
  ~MetalNetworkBuilder(void);

  std::string init(int gpu_id);

  void build(int kInputPlanes, MultiHeadWeights& weights,
             InputEmbedding embedding, bool attn_body, bool attn_policy,
             bool conv_policy, bool wdl, bool moves_left,
             Activations& activations, std::string& policy_head,
             std::string& value_head);

  void forwardEval(float* inputs, int batchSize,
                   std::vector<float*> output_mems);

 private:
  int gpu_id;
};

}  // namespace metal_backend
}  // namespace lczero
