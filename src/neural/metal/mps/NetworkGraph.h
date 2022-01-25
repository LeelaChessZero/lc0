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
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import "ConvWeights.h"

static MPSImageFeatureChannelFormat fcFormat = MPSImageFeatureChannelFormatFloat16;

@interface Lc0GraphNode : NSObject

@property(readwrite, nonatomic, retain) NSArray<Lc0GraphNode *> * __nullable parents;

@property(readwrite, nonatomic, retain) MPSKernel * __nonnull kernel;

@property(readwrite, nonatomic, retain) MPSImageBatch * __nullable result;

@property(readwrite, nonatomic, retain) NSArray * __nullable params;

@property(readwrite) NSUInteger numChildren;

+(nonnull instancetype) graphNodeWithCnnKernel:(MPSKernel * __nonnull)kernel
                                       parents:(NSArray<Lc0GraphNode *> * __nullable)parents
                                        params:(NSArray * __nullable)params;

-(nonnull MPSImageBatch *) encodeBatchToCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                                                input:(MPSImageBatch * __nullable)input
                                         retainResult:(BOOL)retainResult;

@end

@interface Lc0NetworkGraph : NSObject {
  @public
    // Keep the MTLDevice and MTLCommandQueue objects around for ease of use.
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    
    // All the nodes in the graph.
    NSArray<Lc0GraphNode *> *graphNodes;
    
    // Nodes to keep images for output.
    NSArray<Lc0GraphNode *> *resultNodes;
}

-(nonnull instancetype) initWithDevice:(id <MTLDevice> __nonnull)inputDevice
                          commandQueue:(id <MTLCommandQueue> __nonnull)commandQueue;

-(nonnull Lc0GraphNode *) addConvolutionBlockWithParent:(Lc0GraphNode * __nullable)parent
                                          inputChannels:(NSUInteger)inputChannels
                                         outputChannels:(NSUInteger)outputChannels
                                             kernelSize:(NSUInteger)kernelSize
                                                weights:(float * __nonnull)weights
                                                 biases:(float * __nonnull)biases
                                                hasRelu:(BOOL)hasRelu
                                                  label:(NSString * __nonnull)label;

-(nonnull Lc0GraphNode *) addResidualBlockWithParent:(Lc0GraphNode * __nullable)parent
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

-(nonnull Lc0GraphNode *) addFullyConnectedLayerWithParent:(Lc0GraphNode * __nonnull)parent
                                             inputChannels:(NSUInteger)inputChannels
                                            outputChannels:(NSUInteger)outputChannels
                                                   weights:(float * __nonnull)weights
                                                    biases:(float * __nonnull)biases
                                                activation:(NSString * __nullable)activation
                                                     label:(NSString * __nonnull)label;

-(nonnull Lc0GraphNode *) addSEUnitWithParent:(Lc0GraphNode * __nonnull)parent
                                     skipNode:(Lc0GraphNode * __nonnull)skipNode
                                inputChannels:(NSUInteger)inputChannels
                               outputChannels:(NSUInteger)outputChannels
                                  seFcOutputs:(NSUInteger)seFcOutputs
                                     weights1:(float * __nonnull)weights1
                                      biases1:(float * __nonnull)biases1
                                     weights2:(float * __nonnull)weights2
                                      biases2:(float * __nonnull)biases2
                                        label:(NSString * __nonnull)label
                                      hasRelu:(BOOL)hasRelu;

-(nonnull Lc0GraphNode *) addFlattenLayerWithParent:(Lc0GraphNode * __nonnull)parent;

-(nonnull Lc0GraphNode *) addReshapeLayerWithParent:(Lc0GraphNode * __nonnull)parent
                                       reshapeWidth:(NSUInteger)width
                                      reshapeHeight:(NSUInteger)height
                             reshapeFeatureChannels:(NSUInteger)channels;

-(nonnull Lc0GraphNode *) addPolicyMapLayerWithParent:(Lc0GraphNode * __nonnull)parent
                                            policyMap:(short * __nonnull)policyMap;

-(void) buildGraphWithResultNodes:(NSArray<Lc0GraphNode *> * __nonnull)results;

-(nonnull MPSImageBatch *) createInputImageBatchWithBatchSize:(NSUInteger)batchSize
                                                        masks:(uint64_t * __nonnull)masks
                                                       values:(float * __nonnull)values
                                                inputChannels:(NSUInteger)inputPlanes
                                                 subBatchSize:(NSUInteger)subBatchSize;

-(nonnull NSArray<Lc0GraphNode *> *) runInferenceWithImageBatch:(MPSImageBatch * __nonnull)inputBatch;

-(nonnull NSArray<Lc0GraphNode *> *) runInferenceWithBatchSize:(NSUInteger)batchSize
                                                         masks:(uint64_t * __nonnull)masks
                                                        values:(float * __nonnull)values
                                                 inputChannels:(NSUInteger)inputChannels
                                                  subBatchSize:(NSUInteger)subBatchSize;

-(nonnull id<MTLDevice>) getDevice;

@end
