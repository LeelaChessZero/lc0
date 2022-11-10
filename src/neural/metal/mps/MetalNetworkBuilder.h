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

class MetalNetworkBuilder {
public:
    MetalNetworkBuilder(void);
    ~MetalNetworkBuilder(void);

    std::string init(int gpu_id);

    void build(int kInputPlanes, int channelSize, int kernelSize, LegacyWeights& weights, bool attn_policy, bool conv_policy, bool wdl, bool moves_left, std::string default_activation);

    void forwardEval(float * inputs, int batchSize, std::vector<float *> output_mems);

    void saveVariables(std::vector<std::string> names);

    void dumpVariables(std::vector<std::string> names, int batches);

private:
    int gpu_id;
};

}  // namespace metal_backend
}  // namespace lczero
