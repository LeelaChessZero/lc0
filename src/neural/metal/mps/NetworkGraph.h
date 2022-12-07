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

-(NSUInteger) sizeOfDimensions:(NSArray<NSNumber *> * __nonnull)dimensions;

@end

static MPSImageFeatureChannelFormat fcFormat = MPSImageFeatureChannelFormatFloat16;

@interface Lc0NetworkGraph : NSObject {
  @public
    // Keep the device and command queue objects around for ease of use.
    MPSGraphDevice * _device;
    id<MTLCommandQueue> _queue;

    // MPSGraph implementation.
    MPSGraph * _graph;

    // Input tensor and tensor data placeholders.
    MPSGraphTensor * _inputTensor;

    // Variables to track results of graph inference.
    NSArray<MPSGraphTensor *> * _resultTensors;
    NSArray<MPSGraphTensor *> * _targetTensors;
    NSMutableDictionary<NSNumber *, MPSGraphTensorDataDictionary *> * _resultDataDicts;
    NSMutableDictionary<NSString *, NSObject *> * _readVariables;

    // Variables for triple buffering
    dispatch_semaphore_t _doubleBufferingSemaphore;
}

+(Lc0NetworkGraph * _Nonnull) getGraphAt:(NSNumber * _Nonnull)index;

+(void) graphWithDevice:(id<MTLDevice> __nonnull)device
                index:(NSNumber * _Nonnull)index;

-(nonnull instancetype) initWithDevice:(id<MTLDevice> __nonnull)device;

-(nonnull MPSGraphTensor *) inputPlaceholderWithInputChannels:(NSUInteger)channels
                                                       height:(NSUInteger)height
                                                        width:(NSUInteger)width
                                                        label:(NSString * __nullable)label;

-(nonnull MPSGraphTensor *) addConvolutionBlockWithParent:(MPSGraphTensor * __nonnull)parent
                                            inputChannels:(NSUInteger)inputChannels
                                           outputChannels:(NSUInteger)outputChannels
                                               kernelSize:(NSUInteger)kernelSize
                                                  weights:(float * __nonnull)weights
                                                   biases:(float * __nonnull)biases
                                               activation:(NSString * __nullable)activation
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
                                           seFcOutputs:(NSUInteger)seFcOutputs
                                            activation:(NSString * __nullable)activation;

-(nonnull MPSGraphTensor *) addFullyConnectedLayerWithParent:(MPSGraphTensor * __nonnull)parent
                                               inputChannels:(NSUInteger)inputChannels
                                              outputChannels:(NSUInteger)outputChannels
                                                     weights:(float * __nonnull)weights
                                                      biases:(float * __nonnull)biases
                                                  activation:(NSString * __nullable)activation
                                                       label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) addLayerNormalizationWithSkipParent:(MPSGraphTensor * __nonnull)parent
                                                 secondaryInput:(MPSGraphTensor * __nonnull)secondary
                                                         gammas:(float * __nonnull)gammas
                                                          betas:(float * __nonnull)betas
                                                    channelSize:(NSUInteger)channelSize
                                                        epsilon:(float)epsilon
                                                          label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) scaledMHAMatmulWithQueries:(MPSGraphTensor * __nonnull)queries
                                              withKeys:(MPSGraphTensor * __nonnull)keys
                                            withValues:(MPSGraphTensor * __nonnull)values
                                                 heads:(NSUInteger)heads
                                                 label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) scaledQKMatmulWithQueries:(MPSGraphTensor * __nonnull)queries
                                             withKeys:(MPSGraphTensor * __nonnull)keys
                                                scale:(float)scale
                                                label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) attentionPolicyPromoMatmulConcatWithParent:(MPSGraphTensor * __nonnull)parent
                                                              withKeys:(MPSGraphTensor * __nonnull)keys
                                                               weights:(float * __nonnull)weights
                                                             inputSize:(NSUInteger)inputSize
                                                            outputSize:(NSUInteger)outputSize
                                                             sliceFrom:(NSUInteger)sliceFrom
                                                           channelSize:(NSUInteger)channelSize
                                                                 label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) transposeChannelsWithTensor:(MPSGraphTensor * __nonnull)tensor
                                                  label:(NSString * __nonnull)label;

-(void) setResultTensors:(NSArray<MPSGraphTensor *> * __nonnull)results;

-(nonnull NSArray<MPSGraphTensor *> *) runInferenceWithBatchSize:(NSUInteger)batchSize
                                                          inputs:(float * __nonnull)inputs
                                                         outputs:(float * __nonnull * __nonnull)outputBuffers;

-(nonnull MPSCommandBuffer *) runCommandSubBatchWithInputs:(float * __nonnull)inputs
                                                  subBatch:(NSUInteger)subBatch
                                              subBatchSize:(NSUInteger)subBatchSize;

-(void) copyResultsToBuffers:(float * __nonnull * __nonnull)outputBuffers
                subBatchSize:(NSUInteger)subBatchSize;

-(void) setVariable:(NSString * __nonnull)name
             tensor:(MPSGraphTensor * __nonnull)tensor;

-(void) trackVariable:(NSString * __nonnull)name;

-(void) dumpVariable:(NSString * __nonnull)name
             batches:(NSUInteger)batches;

@end
