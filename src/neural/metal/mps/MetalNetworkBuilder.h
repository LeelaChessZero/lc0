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

    //void init(void* weights, void* options);
    void init();

    void* makeConvolutionBlock(void * previousLayer, int inputSize, int channelSize, int kernelSize,
                               float * weights, float * biases, bool withRelu, std::string label);

    void* makeResidualBlock(void * previousLayer, int inputSize, int channelSize, int kernelSize,
                            float * weights1, float * biases1, float * weights2, float * biases2,
                            bool withRelu, std::string label, bool withSe, int seFcOutputs,
                            float * seWeights1, float * seBiases1, float * seWeights2, float * seBiases2);

    void* makeFullyConnectedLayer(void * previousLayer, int inputSize, int outputSize,
                                  float * weights, float * biases, std::string activation, std::string label);
    
    void* makeFlattenLayer(void * previousLayer);

    void* makeReshapeLayer(void * previousLayer, int resultWidth, int resultHeight, int resultChannels);
    
    void* makePolicyMapLayer(void * previousLayer, short * policyMap);
    
    void* buildGraph(std::vector<void *> * outputs);

    std::vector<float*> forwardEval(uint64_t * masks, float * values, std::vector<float *> * outputs, int batchSize, int inputChannels);
    
private:
    void* self;
};

}  // namespace metal_backend
}  // namespace lczero
