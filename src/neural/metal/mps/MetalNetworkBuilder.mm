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

#import "MetalNetworkBuilder.h"
#import "NetworkGraph.h"

#import "Utilities.h"

namespace lczero {
namespace metal_backend {

MetalNetworkBuilder::MetalNetworkBuilder(void) : self(NULL) {}

MetalNetworkBuilder::~MetalNetworkBuilder(void)
{
    [(id)self dealloc];
}

//void MetalNetworkBuilder::init(void* weights, void* options)
std::string MetalNetworkBuilder::init()
{
    // All metal devices.
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    
    if ([devices count] < 1) {
        // No GPU device.
        [NSException raise:@"Could not find device" format:@"Could not find a GPU or CPU compute device"];
        return "";
    }
    
    // Get the metal device and commandQueue to be used.
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == NULL) {
        // Fallback to first device if there is none.
        // @todo allow GPU to be specified via options.
        device = devices[0];
    }
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    // Initialize the metal MPS Graph executor with the device.
    self = [[Lc0NetworkGraph alloc] initWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                      commandQueue:commandQueue];
    
    return std::string([device.name UTF8String]);
}

void* MetalNetworkBuilder::getInputPlaceholder(int maxBatchSize, int width, int height, int channels, std::string label) {
    return [(id)self inputPlaceholderWithMaxBatch:maxBatchSize
                                    inputChannels:channels
                                           height:height
                                            width:width
                                         label:[NSString stringWithUTF8String:label.c_str()]];
}

void* MetalNetworkBuilder::makeConvolutionBlock(void * previousLayer, int inputSize, int channelSize, int kernelSize,
                                                float * weights, float * biases, bool withRelu, std::string label) {
    return [(id)self addConvolutionBlockWithParent:(MPSGraphTensor *)previousLayer
                                     inputChannels:inputSize
                                    outputChannels:channelSize
                                        kernelSize:kernelSize
                                           weights:weights
                                            biases:biases
                                           hasRelu:(BOOL)withRelu
                                             label:[NSString stringWithUTF8String:label.c_str()]];
}

void* MetalNetworkBuilder::makeResidualBlock(void * previousLayer, int inputSize, int channelSize, int kernelSize,
                                             float * weights1, float * biases1, float * weights2, float * biases2,
                                             bool withRelu, std::string label, bool withSe, int seFcOutputs,
                                             float * seWeights1, float * seBiases1, float * seWeights2, float * seBiases2) {

    return [(id)self addResidualBlockWithParent:(MPSGraphTensor *)previousLayer
                                  inputChannels:inputSize
                                 outputChannels:channelSize
                                     kernelSize:kernelSize
                                       weights1:weights1
                                        biases1:biases1
                                       weights2:weights2
                                        biases2:biases2
                                          label:[NSString stringWithUTF8String:label.c_str()]
                                          hasSe:withSe ? YES : NO
                                     seWeights1:seWeights1
                                      seBiases1:seBiases1
                                     seWeights2:seWeights2
                                      seBiases2:seBiases2
                                    seFcOutputs:seFcOutputs];
}

void* MetalNetworkBuilder::makeFullyConnectedLayer(void * previousLayer, int inputSize, int outputSize,
                                                   float * weights, float * biases,
                                                   std::string activation, std::string label) {

    return [(id)self addFullyConnectedLayerWithParent:(MPSGraphTensor *)previousLayer
                                        inputChannels:inputSize
                                       outputChannels:outputSize
                                              weights:weights
                                               biases:biases
                                           activation:[NSString stringWithUTF8String:activation.c_str()]
                                                label:[NSString stringWithUTF8String:label.c_str()]];
}

void* MetalNetworkBuilder::makePolicyMapLayer(void * previousLayer, uint32_t * policyMap, std::string label) {
    return [(id)self addPolicyMapLayerWithParent:(MPSGraphTensor *)previousLayer
                                       policyMap:policyMap
                                           label:[NSString stringWithUTF8String:label.c_str()]];
}

void* MetalNetworkBuilder::setSelectedOutputs(std::vector<void *> * outputs) {
    NSArray<MPSGraphTensor *> * resultTensors = @[];

    for (const auto& output : *outputs) {
        resultTensors = [resultTensors arrayByAddingObject:(MPSGraphTensor *)output];
    }
    [(id)self setResultTensors:resultTensors];

    return (void*) self;
}

void MetalNetworkBuilder::forwardEval(float * inputs, int batchSize, int inputChannels, std::vector<float *> output_mems)
{
    NSArray<MPSGraphTensor *> * resultTensors = [(id)self runInferenceWithBatchSize:batchSize
                                                                             inputs:inputs
                                                                      inputChannels:inputChannels
                                                                      outputBuffers:&output_mems[0]];
}

}  // namespace metal_backend
}  // namespace lczero
