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

#import "NetworkGraph.h"
#import "ConvWeights.h"
#import "Shaders.h"

#import <vector>

#ifndef ADVANCE_PTR
#   define ADVANCE_PTR(_a, _size) (__typeof__(_a))((uintptr_t) (_a) + (size_t)(_size))
#endif

@implementation Lc0GraphNode

+(nonnull instancetype) graphNodeWithCnnKernel:(MPSKernel * __nonnull)kernel
                                       parents:(NSArray<Lc0GraphNode *> * __nullable)parents
                                        params:(NSArray * __nullable)params
{
    Lc0GraphNode * node = [[Lc0GraphNode alloc] init];
    node.parents = parents;
    node.kernel = kernel;
    node.params = params;
    
    return node;
}

-(nonnull MPSImageBatch *) encodeBatchToCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                                                input:(MPSImageBatch * __nullable)input
                                         retainResult:(BOOL)retainResult
{
    assert(input != nil || [self.parents count] > 0);
    
    if ([self.parents count] > 0) input = self.parents[0].result;
    
    if (retainResult) {
        // Graph nodes specified as outputs shouldn't be temporary images so we can read it to CPU.
        [self.kernel setDestinationImageAllocator:[MPSImage defaultAllocator]];
    }

    if ([self.kernel isKindOfClass:[Lc0SeMultiplyAdd class]]) {
        // SE nodes accept more parameters.
        assert([self.parents count] >= 3);
        self.result = [(Lc0SeMultiplyAdd *)self.kernel encodeBatchToCommandBuffer:commandBuffer
                                                        seSourceImages:input
                                                      convSourceImages:self.parents[1].result
                                                      skipSourceImages:self.parents[2].result];
    }
    else if ([self.kernel isKindOfClass:[MPSCNNBinaryKernel class]]) {
        // Binary kernel nodes accept a secondary image.
        assert([self.parents count] >= 2);
        self.result = [(MPSCNNBinaryKernel *)self.kernel encodeBatchToCommandBuffer:commandBuffer
                                                                      primaryImages:input
                                                                    secondaryImages:self.parents[1].result];
    }
    else if ([self.kernel isKindOfClass:[MPSNNReshape class]]) {
        // Reshape nodes accept more parameters.
        assert([self.params count] >= 3);
        self.result = [(MPSNNReshape *)self.kernel encodeBatchToCommandBuffer:commandBuffer
                                                                 sourceImages:input
                                                                reshapedWidth:[self.params[0] intValue]
                                                               reshapedHeight:[self.params[1] intValue]
                                                      reshapedFeatureChannels:[self.params[2] intValue]];
    }
    else if ([self.kernel isKindOfClass:[MPSImageTranspose class]]) {
        MPSImage * image = input[0];
        MPSImageDescriptor * descriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:image.featureChannelFormat
                                                                                         width:image.width
                                                                                        height:image.height
                                                                               featureChannels:image.featureChannels
                                                                                numberOfImages:image.numberOfImages
                                                                                         usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];
        descriptor.storageMode = MTLStorageModeShared;
        id<MPSImageAllocator> allocator = retainResult ? [MPSImage defaultAllocator] : [MPSTemporaryImage defaultAllocator];
        self.result = [allocator imageBatchForCommandBuffer:commandBuffer
                                            imageDescriptor:descriptor
                                                     kernel:self.kernel
                                                      count:[input count]];
        for (int i=0; i < [self.result count]; i++) {
            [(MPSImageTranspose *)self.kernel encodeToCommandBuffer:commandBuffer
                                                        sourceImage:input[i]
                                                   destinationImage:self.result[i]];
        }
    }
    else if ([self.kernel isKindOfClass:[MPSCNNKernel class]]) {
        self.result = [(MPSCNNKernel *)self.kernel encodeBatchToCommandBuffer:commandBuffer
                                                                 sourceImages:input];
    }

    // Ensure temporary images have readCount incremented to match number of children nodes that will be read later.
    if ([self.result[0] isKindOfClass:[MPSTemporaryImage class]]) {
        MPSImageBatchIncrementReadCount(self.result, self.numChildren - 1);
    }
    return self.result;
}

@end

@implementation Lc0NetworkGraph

-(nonnull instancetype) initWithDevice:(id<MTLDevice> __nonnull)inputDevice
                          commandQueue:(id<MTLCommandQueue> __nonnull)commandQueue {
    
    self = [super init];
    device = inputDevice;
    queue = commandQueue;
    graphNodes = @[];
    resultNodes = @[];
    
    return self;
}

-(nonnull id<MTLDevice>) getDevice {
    return device;
}

-(nonnull MPSImageBatch *) createInputImageBatchWithBatchSize:(NSUInteger)batchSize
                                                        masks:(uint64_t * __nonnull)masks
                                                       values:(float * __nonnull)values
                                                inputChannels:(NSUInteger)inputPlanes
                                                 subBatchSize:(NSUInteger)subBatchSize
{
    
    const uint8_t boardWidth = 8;
    const uint8_t boardHeight = 8;
    const uint8_t bytesPerRow = boardWidth * sizeof(float);

    MPSImageDescriptor *inputDesc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                                                                   width:boardWidth
                                                                                  height:boardHeight
                                                                         featureChannels:inputPlanes
                                                                          numberOfImages:subBatchSize
                                                                                   usage:MTLTextureUsageShaderRead];
    // Buffer for expanding packed planes.
    float * buffer = (float *)malloc(inputPlanes * boardWidth * boardHeight * sizeof(float));
    
    // Create an input MPSImageBatch.
    MPSImageBatch *inputBatch = @[];
    for (int subBatch = 0; subBatch < batchSize; subBatch += subBatchSize) {
        
        MPSImage *inputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:inputDesc];
        
        // Expand packed planes to full planes, one batch at a time.
        int subBatchEnd = MIN(subBatch + subBatchSize, batchSize);
        for (int i = subBatch; i < subBatchEnd; i++) {
            float * dptr = buffer;
            for (int j = 0; j < inputPlanes; j++) {
                const float value = values[j + i * inputPlanes];
                const uint64_t mask = masks[j + i * inputPlanes];
                for (auto k = 0; k < 64; k++) {
                    *(dptr++) = (mask & (((uint64_t)1) << k)) != 0 ? value : 0;
                }
            }
            [inputImage writeBytes:buffer
                        dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth
                        imageIndex:i - subBatch];
        }
        // Add image to input batch.
        inputBatch = [inputBatch arrayByAddingObject:inputImage];
    }
    
    return inputBatch;
}

    
-(nonnull NSArray<Lc0GraphNode *> *) runInferenceWithImageBatch:(MPSImageBatch * __nonnull)inputBatch {
    // Make an MPSCommandBuffer, when passed to the encode of MPSNNGraph, commitAndContinue will be automatically used.
    MPSCommandBuffer *commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
    
    // Use double buffering to keep the GPU completely busy.
    //        dispatch_semaphore_t doubleBufferingSemaphore = dispatch_semaphore_create(2);
    //        dispatch_semaphore_wait(doubleBufferingSemaphore, DISPATCH_TIME_FOREVER);
    
    // Encode inference network
    MPSImageBatch *output;
    int i = 0;
    for (Lc0GraphNode * node in graphNodes) {
        @autoreleasepool {
            output = [node encodeBatchToCommandBuffer:commandBuffer
                                                          input:inputBatch
                                                   retainResult:[resultNodes containsObject:node]];
        }
    }

    // Transfer data from GPU to CPU.
    for (Lc0GraphNode * node in resultNodes) {
        MPSImageBatchSynchronize(node.result, commandBuffer);
    }

    // Commit the command buffer. Wait for the last batch to be processed.
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    return resultNodes;
}

-(nonnull NSArray<Lc0GraphNode *> *) runInferenceWithBatchSize:(NSUInteger)batchSize
                                                         masks:(uint64_t * __nonnull)masks
                                                        values:(float * __nonnull)values
                                                 inputChannels:(NSUInteger)inputChannels
                                                  subBatchSize:(NSUInteger)subBatchSize
{
    @autoreleasepool {
        // Create an input MPSImageBatch.
        MPSImageBatch *inputBatch = [self createInputImageBatchWithBatchSize:batchSize
                                                                       masks:masks
                                                                      values:values
                                                               inputChannels:inputChannels
                                                                subBatchSize:subBatchSize];
        
        return [self runInferenceWithImageBatch:inputBatch];
    }
}

-(void) buildGraphWithResultNodes:(NSArray<Lc0GraphNode *> * __nonnull)results
{
    resultNodes = results;
}

-(nonnull Lc0GraphNode *) addConvolutionBlockWithParent:(Lc0GraphNode * __nullable)parent
                                          inputChannels:(NSUInteger)inputChannels
                                         outputChannels:(NSUInteger)outputChannels
                                             kernelSize:(NSUInteger)kernelSize
                                                weights:(float * __nonnull)weights
                                                 biases:(float * __nonnull)biases
                                                hasRelu:(BOOL)hasRelu
                                                  label:(NSString * __nonnull)label
{
    ConvWeights *convWeights = [[ConvWeights alloc] initWithDevice:device
                                                     inputChannels:inputChannels
                                                    outputChannels:outputChannels
                                                       kernelWidth:kernelSize
                                                      kernelHeight:kernelSize
                                                            stride:1
                                                           weights:weights
                                                            biases:biases
                                                             label:[NSString stringWithFormat:@"%@/weights", label]
                                                   fusedActivation:hasRelu ? @"relu" : nil];
    
    graphNodes = [graphNodes arrayByAddingObject:[Lc0GraphNode graphNodeWithCnnKernel:[[MPSCNNConvolution alloc] initWithDevice:device weights:convWeights] parents:(parent != nil ? @[parent] : nil) params:nil]];
    parent.numChildren++;

    return graphNodes[[graphNodes count] - 1];
}

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
                                         seFcOutputs:(NSUInteger)seFcOutputs
{
    // Conv1
    ConvWeights *convWeights1 = [[ConvWeights alloc] initWithDevice:device
                                                      inputChannels:inputChannels
                                                     outputChannels:outputChannels
                                                        kernelWidth:kernelSize
                                                       kernelHeight:kernelSize
                                                             stride:1
                                                            weights:weights1
                                                             biases:biases1
                                                              label:[NSString stringWithFormat:@"%@/conv1/weights", label]
                                                   fusedActivation:@"relu"];
    

    Lc0GraphNode *conv1 = [Lc0GraphNode graphNodeWithCnnKernel:[[MPSCNNConvolution alloc] initWithDevice:device weights:convWeights1] parents:(parent != nil ? @[parent] : nil) params:nil];
    graphNodes = [graphNodes arrayByAddingObject:conv1];
    parent.numChildren++;
    
    // Conv2
    ConvWeights *convWeights2 = [[ConvWeights alloc] initWithDevice:device
                                                      inputChannels:inputChannels
                                                     outputChannels:outputChannels
                                                        kernelWidth:kernelSize
                                                       kernelHeight:kernelSize
                                                             stride:1
                                                            weights:weights2
                                                             biases:biases2
                                                              label:[NSString stringWithFormat:@"%@/conv2/weights", label]
                                                    fusedActivation:nil];
    
    Lc0GraphNode *conv2 = [Lc0GraphNode graphNodeWithCnnKernel:[[MPSCNNConvolution alloc] initWithDevice:device weights:convWeights2] parents:(conv1 != nil ? @[conv1] : nil) params:nil];
    graphNodes = [graphNodes arrayByAddingObject:conv2];
    conv1.numChildren++;
    

    if (hasSe) {
        // SE Unit.
        Lc0GraphNode *seUnit = [self addSEUnitWithParent:conv2
                                                skipNode:parent
                                           inputChannels:inputChannels
                                          outputChannels:outputChannels
                                             seFcOutputs:seFcOutputs
                                                weights1:seWeights1
                                                 biases1:seBiases1
                                                weights2:seWeights2
                                                 biases2:seBiases2
                                                   label:[NSString stringWithFormat:@"%@/se", label]
                                                 hasRelu:YES];
        return seUnit;
    }
    else {
        Lc0GraphNode *add = [Lc0GraphNode graphNodeWithCnnKernel:[[MPSCNNAdd alloc] initWithDevice:device]
                                                         parents:@[parent, conv2]
                                                          params:nil];
        graphNodes = [graphNodes arrayByAddingObject:add];
        parent.numChildren++;
        conv2.numChildren++;

        Lc0GraphNode *relu = [Lc0GraphNode graphNodeWithCnnKernel:[[MPSCNNNeuronReLU alloc] initWithDevice:device a:0.0]
                                                          parents:@[add]
                                                           params:nil];
        graphNodes = [graphNodes arrayByAddingObject:relu];
        add.numChildren++;
        
        return relu;
    }
}

-(nonnull Lc0GraphNode *) addFullyConnectedLayerWithParent:(Lc0GraphNode * __nonnull)parent
                                             inputChannels:(NSUInteger)inputChannels
                                            outputChannels:(NSUInteger)outputChannels
                                                   weights:(float * __nonnull)weights
                                                    biases:(float * __nonnull)biases
                                                activation:(NSString * __nullable)activation
                                                     label:(NSString * __nonnull)label
{
    ConvWeights *convWeights = [[ConvWeights alloc] initWithDevice:device
                                                      inputChannels:inputChannels
                                                     outputChannels:outputChannels
                                                        kernelWidth:1
                                                       kernelHeight:1
                                                             stride:1
                                                            weights:weights
                                                             biases:biases
                                                              label:[NSString stringWithFormat:@"%@/weights", label]
                                                    fusedActivation:activation];
    
    graphNodes = [graphNodes arrayByAddingObject:[Lc0GraphNode graphNodeWithCnnKernel:[[MPSCNNFullyConnected alloc] initWithDevice:device weights:convWeights] parents:@[parent] params:nil]];
    parent.numChildren++;
    
    return graphNodes[[graphNodes count] - 1];;
}

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
                                      hasRelu:(BOOL)hasRelu
{
    // 1. Global Average Pooling 2D
    MPSCNNPoolingAverage *pool = [[MPSCNNPoolingAverage alloc] initWithDevice:device kernelWidth:8 kernelHeight:8 strideInPixelsX:8 strideInPixelsY:8];

    Lc0GraphNode * poolNode = [Lc0GraphNode graphNodeWithCnnKernel:pool parents:@[parent] params:nil];
    graphNodes = [graphNodes arrayByAddingObject:poolNode];
    parent.numChildren++;
    
    // 2. FC Layer 1.
    Lc0GraphNode *fcNode1 = [self addFullyConnectedLayerWithParent:poolNode
                                                     inputChannels:inputChannels
                                                    outputChannels:seFcOutputs
                                                           weights:weights1
                                                            biases:biases1
                                                        activation:@"relu"
                                                             label:[NSString stringWithFormat:@"%@/fc1", label]];

    // 3. FC Layer 2.
    Lc0GraphNode *fcNode2 = [self addFullyConnectedLayerWithParent:fcNode1
                                                  inputChannels:seFcOutputs
                                                 outputChannels:2 * inputChannels
                                                        weights:weights2
                                                         biases:biases2
                                                     activation:nil
                                                          label:[NSString stringWithFormat:@"%@/fc2", label]];
    
    // 4. Multiply and add.
    Lc0SeMultiplyAdd *multiply = [[Lc0SeMultiplyAdd alloc] initWithDevice:device gammaChannels:inputChannels betaChannels:inputChannels hasRelu:hasRelu];
    
    Lc0GraphNode * multiplyNode = [Lc0GraphNode graphNodeWithCnnKernel:multiply parents:@[fcNode2, parent, skipNode] params:nil];
    graphNodes = [graphNodes arrayByAddingObject:multiplyNode];
    fcNode2.numChildren++;
    parent.numChildren++;
    skipNode.numChildren++;
    
    return multiplyNode;
    
}

-(nonnull Lc0GraphNode *) addFlattenLayerWithParent:(Lc0GraphNode * __nonnull)parent
{
    // Flatten kernel.
    Lc0Flatten *flatten = [[Lc0Flatten alloc] initWithDevice:device];
    
    Lc0GraphNode * flattenNode = [Lc0GraphNode graphNodeWithCnnKernel:flatten parents:@[parent] params:nil];
    graphNodes = [graphNodes arrayByAddingObject:flattenNode];
    parent.numChildren++;
    
    return flattenNode;
}

-(nonnull Lc0GraphNode *) addReshapeLayerWithParent:(Lc0GraphNode * __nonnull)parent
                                       reshapeWidth:(NSUInteger)width
                                      reshapeHeight:(NSUInteger)height
                             reshapeFeatureChannels:(NSUInteger)channels
{
    // Reshape kernel.
    MPSNNReshape *reshape = [[MPSNNReshape alloc] initWithDevice:device];
    
    Lc0GraphNode * reshapeNode = [Lc0GraphNode graphNodeWithCnnKernel:reshape parents:@[parent] params:@[@(width), @(height), @(channels)]];
    graphNodes = [graphNodes arrayByAddingObject:reshapeNode];
    parent.numChildren++;
    
    return reshapeNode;
}

-(nonnull Lc0GraphNode *) addPolicyMapLayerWithParent:(Lc0GraphNode * __nonnull)parent
                                            policyMap:(short * __nonnull)policyMap
{
    // Policy map kernel.
    Lc0PolicyMap *map = [[Lc0PolicyMap alloc] initWithDevice:device policyMap:policyMap];
    
    Lc0GraphNode * mapNode = [Lc0GraphNode graphNodeWithCnnKernel:map parents:@[parent] params:nil];
    graphNodes = [graphNodes arrayByAddingObject:mapNode];
    parent.numChildren++;
    
    return mapNode;
}

@end

