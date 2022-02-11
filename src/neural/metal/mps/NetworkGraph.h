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

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

@interface MPSGraphTensor(Lc0Extensions)

-(NSUInteger) size;

-(NSUInteger) sizeOfDimensions:(NSArray<NSNumber *> *)dimensions;

@end

static MPSImageFeatureChannelFormat fcFormat = MPSImageFeatureChannelFormatFloat16;

@interface Lc0NetworkGraph : NSObject {
  @public
    // Keep the device and command queue objects around for ease of use.
    MPSGraphDevice * _device;
    id<MTLCommandQueue> _queue;
    
    // All the nodes in the graph.
    MPSGraph * _graph;
    
    // Input tensor placeholder.
    MPSGraphTensor * _inputTensor;
    
    // Array to keep output tensors.
    NSArray<MPSGraphTensor *> * _resultTensors;
    
    // Size of inference volume.
    NSUInteger _batchesPerSplit;
    
    // Variables for triple buffering
    dispatch_semaphore_t _doubleBufferingSemaphore;
//    NSUInteger currentFrameIndex;
//    NSArray<id <MTLBuffer>> dynamicDataBuffers;
}

-(nonnull instancetype) initWithDevice:(id<MTLDevice> __nonnull)device
                       batchesPerSplit:(NSUInteger)batchesPerSplit;

-(nonnull MPSGraphTensor *) inputPlaceholderWithMaxBatch:(NSUInteger)maxBatchSize
                                           inputChannels:(NSUInteger)channels
                                                  height:(NSUInteger)height
                                                   width:(NSUInteger)width
                                                   label:(NSString * __nullable)label;

-(nonnull MPSGraphTensor *) addConvolutionBlockWithParent:(MPSGraphTensor * __nonnull)parent
                                          inputChannels:(NSUInteger)inputChannels
                                         outputChannels:(NSUInteger)outputChannels
                                             kernelSize:(NSUInteger)kernelSize
                                                weights:(float * __nonnull)weights
                                                 biases:(float * __nonnull)biases
                                                hasRelu:(BOOL)hasRelu
                                                  label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) addResidualBlockWithParent:(MPSGraphTensor * __nonnull)parent
                                       inputChannels:(NSUInteger)inputChannels
                                      outputChannels:(NSUInteger)outputChannels
                                          kernelSize:(NSUInteger)kernelSize
                                            weights1:(float * __nonnull)weights1
                                             biases1:(float * __nonnull)biases1
                                            weights2:(float * __nonnull)weights2
                                             biases2:(float * __nonnull)biases2
                                               label:(NSString * __nonnull)label
                                               hasSe:(BOOL)hasSe
                                          seWeights1:(float * __nullable)seWeights1
                                           seBiases1:(float * __nullable)seBiases1
                                          seWeights2:(float * __nullable)seWeights2
                                           seBiases2:(float * __nullable)seBiases2
                                         seFcOutputs:(NSUInteger)seFcOutputs;

-(nonnull MPSGraphTensor *) addFullyConnectedLayerWithParent:(MPSGraphTensor * __nonnull)parent
                                             inputChannels:(NSUInteger)inputChannels
                                            outputChannels:(NSUInteger)outputChannels
                                                   weights:(float * __nonnull)weights
                                                    biases:(float * __nonnull)biases
                                                activation:(NSString * __nullable)activation
                                                     label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) addSEUnitWithParent:(MPSGraphTensor * __nonnull)parent
                                     skipNode:(MPSGraphTensor * __nonnull)skipNode
                                inputChannels:(NSUInteger)inputChannels
                               outputChannels:(NSUInteger)outputChannels
                                  seFcOutputs:(NSUInteger)seFcOutputs
                                     weights1:(float * __nonnull)weights1
                                      biases1:(float * __nonnull)biases1
                                     weights2:(float * __nonnull)weights2
                                      biases2:(float * __nonnull)biases2
                                        label:(NSString * __nonnull)label
                                      hasRelu:(BOOL)hasRelu;

-(nonnull MPSGraphTensor *) addPolicyMapLayerWithParent:(MPSGraphTensor * __nonnull)parent
                                              policyMap:(uint32_t * __nonnull)policyMap
                                                  label:(NSString *)label;

-(void) setResultTensors:(NSArray<MPSGraphTensor *> * __nonnull)results;

-(nonnull NSArray<MPSGraphTensor *> *) runInferenceWithBatchSize:(NSUInteger)batchSize
                                                          inputs:(float * __nonnull)inputs
                                                   inputChannels:(NSUInteger)inputPlanes
                                                   outputBuffers:(float * * __nonnull)outputBuffers;

@end
