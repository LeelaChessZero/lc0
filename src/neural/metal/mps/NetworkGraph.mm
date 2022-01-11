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

NSString * listOfFloats(float * floats, int count) {
    NSMutableString * buf = [[NSMutableString alloc] init];
    [buf appendString:@"|"];
    for (int i=0; i<count; i++) {
        [buf appendFormat:@"%i|", (int)floats[i]];
        //[buf appendFormat:float[i]];
       // [buf appendString:@"|"];
    }
    return buf;
}

void logBeforeAfter(float * before, float * after, int inputPlanes) {
    for (int i=0; i < inputPlanes; i++)
        for (int j=0; j<64; j++)
            NSLog(@"%i: plane[%i] position[%i] %i >> %i", i*64+j, i, j, (int)before[i*64+j], (int)after[i*64+j]);
}


@implementation Lc0GraphNode

+(nonnull instancetype) graphNodeWithCnnKernel:(MPSKernel * __nonnull)kernel
                                       parents:(NSArray<Lc0GraphNode *> * __nullable)parents {
    Lc0GraphNode * node = [[Lc0GraphNode alloc] init];
    node.parents = parents;
    node.kernel = kernel;
    
    return node;
}

@end

@implementation Lc0NetworkGraph

-(nonnull instancetype) initWithDevice:(id<MTLDevice> __nonnull)inputDevice
                          commandQueue:(id<MTLCommandQueue> __nonnull)commandQueue {
    
    self = [super init];
    device = inputDevice;
    queue = commandQueue;
    graphNodes = @[];
    
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
    float buffer[inputPlanes * boardWidth * boardHeight];
    
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
                //NSLog(@"mask[%d]: %lu, value[%d]: %f", j + i * inputPlanes, mask, j + i * inputPlanes, value);
                for (auto k = 0; k < 64; k++) {
                    *(dptr++) = (mask & (((uint64_t)1) << k)) != 0 ? value : 0;
//                    *(dptr++) = j == 1 && k == 9 ? 1 : 0;
                }
            }
//            for (int k = 0; k < inputPlanes * boardWidth * boardHeight; k++) {
//                NSLog(@"Input at %i: %f", k, buffer[k]);
//            }
            [inputImage writeBytes:buffer
                        dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth
                        imageIndex:i - subBatch];
        }
        // Add image to input batch.
        inputBatch = [inputBatch arrayByAddingObject:inputImage];
    }
    
    return inputBatch;
}

    
-(nonnull NSMutableArray<MPSImageBatch*> *) runInferenceWithImageBatch:(MPSImageBatch * __nonnull)inputBatch {
    // Make an MPSCommandBuffer, when passed to the encode of MPSNNGraph, commitAndContinue will be automatically used.
    NSLog(@"Initializing command buffer");
    MPSCommandBuffer *commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
    
    // Encode inference network
    NSLog(@"Encoding command buffer");
//    NSLog(@"convkernel: %@", graphNodes[0]);
//    ((MPSCNNKernel *)graphNodes[0][@"kernel"]).destinationImageAllocator = [MPSImage defaultAllocator];
//    NSMutableArray<MPSImageBatch*> *results = [[NSMutableArray alloc] init];
//    MPSImageBatch *output = [graphNodes[0][@"kernel"] encodeBatchToCommandBuffer:commandBuffer
//                                                     sourceImages:inputBatch];
    NSLog(@"Processing %i graph nodes %@", [graphNodes count], graphNodes);
    NSMutableArray<MPSImageBatch*> *results = [[NSMutableArray alloc] init];
    MPSImageBatch *output;
    MPSImageBatch *input;
    int i = 0;
    for (Lc0GraphNode * node in graphNodes) {
        NSLog(@"Started node %i %@", i, node);
        NSLog(@"Node info: parents - %i, children - %i, kernel %@", [node.parents count], node.numChildren, node.kernel);
       if (node /* is in result graph nodes list */) {
            //((MPSCNNKernel *)node.kernel).destinationImageAllocator = [MPSImage defaultAllocator];
        }
        if (node.parents == nil || [node.parents count] == 0) {
            input = inputBatch;
        }
        else {
            input = node.parents[0].result;
        }
        if ([node.kernel isKindOfClass:[MPSCNNBinaryKernel class]]) {
            // Arithmetic kernels need result image location allocated. They also have 2 inputs.
            MPSCNNArithmeticGradientStateBatch *destStates = @[];
            id<MPSImageAllocator> allocator;
            if (i == [graphNodes count] - 1) {
                allocator = [MPSImage defaultAllocator];
            }
            else {
                allocator = [MPSTemporaryImage defaultAllocator];
            }
            MPSImage * image = input[0];
            MPSImageDescriptor *descriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:image.featureChannelFormat
                                                                                            width:image.width
                                                                                           height:image.height
                                                                                  featureChannels:image.featureChannels
                                                                                   numberOfImages:image.numberOfImages
                                                                                            usage:image.usage];
            MPSImageBatch *tempBatch = [allocator imageBatchForCommandBuffer:commandBuffer
                                                             imageDescriptor:descriptor
                                                                      kernel:node.kernel
                                                                       count:[input count]];
            for (MPSImage * image in input) {
                destStates = [destStates arrayByAddingObject:[MPSCNNArithmeticGradientState alloc]];
            }
            NSLog(@"Parents: %@ %@", node.parents[0].result, node.parents[1].result);
            NSLog(@"Temp image readcounts: %i %i", ((MPSTemporaryImage *)node.parents[0].result[0]).readCount, ((MPSTemporaryImage *)node.parents[1].result[0]).readCount);
            [(MPSCNNBinaryKernel *)node.kernel encodeBatchToCommandBuffer:commandBuffer
                                                             primaryImages:node.parents[0].result
                                                           secondaryImages:node.parents[1].result
                                                         destinationStates:destStates
                                                         destinationImages:tempBatch];
            output = tempBatch;
            node.result = output;
        }
        else if ([node.kernel isKindOfClass:[MPSCNNKernel class]]) {
            if (i == [graphNodes count] - 1) {
                // Last graph node shouldn't be temporary image so we can read it to CPU.
                // @todo Later on, the list of outputs will be a graph property.
                ((MPSCNNKernel *)node.kernel).destinationImageAllocator = [MPSImage defaultAllocator];
            }
            output = [(MPSCNNKernel *)node.kernel encodeBatchToCommandBuffer:commandBuffer sourceImages:input];
            node.result = output;
            if ([output[0] isKindOfClass:[MPSTemporaryImage class]]) {
                MPSImageBatchIncrementReadCount(output, node.numChildren - 1);
                NSLog(@"Temp image readcount: %i", ((MPSTemporaryImage *)output[0]).readCount);
            }
            NSLog(@"Result: %@", output);
        }
        NSLog(@"Finished node %i %@", i++, node);
    }
    
    // Transfer data from GPU to CPU.
    NSLog(@"Synchronizing GPU to CPU");
    MPSImageBatchSynchronize(output, commandBuffer);

    // Commit the command buffer. Wait for the last batch to be processed.
    NSLog(@"Committing command buffer");
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    NSLog(@"Buffer completed");

    // Add the first result at the beginning of the list of result images.
    [results insertObject:output atIndex:0];
    NSLog(@"Got results: %@", results);
    return results;
}

-(nonnull NSMutableArray<MPSImageBatch*> *) runInferenceWithBatchSize:(NSUInteger)batchSize
                                                           masks:(uint64_t * __nonnull)masks
                                                          values:(float * __nonnull)values
                                                   inputChannels:(NSUInteger)inputChannels
                                                    subBatchSize:(NSUInteger)subBatchSize
{
    @autoreleasepool {
        // Use double buffering to keep the GPU completely busy.
//        dispatch_semaphore_t doubleBufferingSemaphore = dispatch_semaphore_create(2);
//        dispatch_semaphore_wait(doubleBufferingSemaphore, DISPATCH_TIME_FOREVER);
        
        // Create batches of MPSImages that each contain multiple images (i.e. multiple board positions
        // in sub-batches) in order to optimize GPU usage.

        // Create an input MPSImageBatch.
        MPSImageBatch *inputBatch = [self createInputImageBatchWithBatchSize:batchSize
                                                                       masks:masks
                                                                      values:values
                                                               inputChannels:inputChannels
                                                                subBatchSize:subBatchSize];
        
        return [self runInferenceWithImageBatch:inputBatch];

        // Make an MPSCommandBuffer, when passed to the encode of MPSNNGraph, commitAndContinue will be automatically used.
        //NSLog(@"Initializing command buffer");
        MPSCommandBuffer *commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
        
        // Encode inference network
        NSLog(@"Encoding command buffer");
        NSMutableArray<MPSImageBatch*> *results = [[NSMutableArray alloc] init];
        MPSImageBatch *output = [graph encodeBatchToCommandBuffer:commandBuffer
                                                     sourceImages:@[inputBatch]
                                                     sourceStates:nil
                                               intermediateImages:results
                                                destinationStates:nil];

        // Transfer data from GPU to CPU.
        MPSImageBatchSynchronize(output, commandBuffer);
    
//        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> __nonnull) {
//            // Release double buffering semaphore for the iteration to be encoded.
//            dispatch_semaphore_signal(doubleBufferingSemaphore);

//            // Check the output of inference network to calculate accuracy.
//            [outputBatch enumerateObjectsUsingBlock:^(MPSImage * __nonnull outputImage, NSUInteger idx, BOOL * _Nonnull stop) {
//                /*uint8_t *start = ADVANCE_PTR(outputs, inputOffset + (boardWidth * boardHeight * idx));
//                [inputImage writeBytes:start
//                            dataLayout:(MPSDataLayoutFeatureChannelsxHeightxWidth)
//                            imageIndex:0];*/
//            }];
//
//        }];
        
        // Commit the command buffer. Wait for the last batch to be processed.
        //NSLog(@"Committing command buffer");
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Add the first result at the beginning of the list of result images.
        [results insertObject:output atIndex:0];
        NSLog(@"Got results: batchSize: %i, count: %i, %@", batchSize, [results count], results);
        return results;
    }
}

-(nonnull MPSNNGraph *) buildGraphWithResultImages:(NSArray <MPSNNImageNode *> * __nonnull)resultImages
                                  resultsAreNeeded:(BOOL * __nullable)areResultsNeeded
{
    
//    graph = [MPSNNGraph graphWithDevice:device resultImages:resultImages resultsAreNeeded:areResultsNeeded];
//    graph.format = fcFormat;
//
//    return graph;
    return nil;
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
                                                             label:label
                                                   fusedActivation:hasRelu ? @"relu" : nil];
    
    graphNodes = [graphNodes arrayByAddingObject:[Lc0GraphNode graphNodeWithCnnKernel:[[MPSCNNConvolution alloc] initWithDevice:device weights:convWeights] parents:parent != nil ? @[parent] : nil]];
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
                                                              label:label
                                                   fusedActivation:@"relu"];
    

    Lc0GraphNode *conv1 = [Lc0GraphNode graphNodeWithCnnKernel:[[MPSCNNConvolution alloc] initWithDevice:device weights:convWeights1] parents:parent != nil ? @[parent] : nil];
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
                                                              label:label
                                                    fusedActivation:nil];
    
    Lc0GraphNode *conv2 = [Lc0GraphNode graphNodeWithCnnKernel:[[MPSCNNConvolution alloc] initWithDevice:device weights:convWeights2] parents:conv1 != nil ? @[conv1] : nil];
    graphNodes = [graphNodes arrayByAddingObject:conv2];
    conv1.numChildren++;
    
    if (hasSe) {
        // SE Unit.
        Lc0GraphNode *seUnit = [self addSEUnitWithParent:conv2
                                           inputChannels:inputChannels
                                          outputChannels:outputChannels
                                             seFcOutputs:seFcOutputs
                                                weights1:seWeights1
                                                 biases1:seBiases1
                                                weights2:seWeights2
                                                 biases2:seBiases2
                                                   label:label];
    }
    
//    Lc0GraphNode *add = [Lc0GraphNode graphNodeWithCnnKernel:[[MPSCNNAdd alloc] initWithDevice:device] parents:@[conv1, conv2]];
//    graphNodes = [graphNodes arrayByAddingObject:add];
//    conv1.numChildren++;
//    conv2.numChildren++;
    
    return graphNodes[[graphNodes count] - 1];
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
                                                              label:label
                                                    fusedActivation:activation];
    
    NSLog(@"Got here: inputs %i, outputs %i, weights %f, biases %f", inputChannels, outputChannels, weights[0], biases[0]);
    graphNodes = [graphNodes arrayByAddingObject:[Lc0GraphNode graphNodeWithCnnKernel:[[MPSCNNFullyConnected alloc] initWithDevice:device weights:convWeights] parents:@[parent]]];
    parent.numChildren++;
    NSLog(@"Got here too");
    return graphNodes[[graphNodes count] - 1];;
}

-(nonnull Lc0GraphNode *) addSEUnitWithParent:(Lc0GraphNode * __nonnull)parent
                                inputChannels:(NSUInteger)inputChannels
                               outputChannels:(NSUInteger)outputChannels
                                  seFcOutputs:(NSUInteger)seFcOutputs
                                     weights1:(float * __nonnull)weights1
                                      biases1:(float * __nonnull)biases1
                                     weights2:(float * __nonnull)weights2
                                      biases2:(float * __nonnull)biases2
                                        label:(NSString * __nonnull)label
{
    // 1. Global Average Pooling 2D
    MPSCNNPoolingAverage *pool = [[MPSCNNPoolingAverage alloc] initWithDevice:device kernelWidth:8 kernelHeight:8 strideInPixelsX:8 strideInPixelsY:8];
    //pool.paddingPolicy = validPoolingPadding;

    Lc0GraphNode * poolNode = [Lc0GraphNode graphNodeWithCnnKernel:pool parents:@[parent]];
    graphNodes = [graphNodes arrayByAddingObject:poolNode];
    parent.numChildren++;
    
    NSLog(@"Making SE FC node1: inputChannels %i, seFcOutputs %i", inputChannels, seFcOutputs);

    // 2. FC Layer 1.
    Lc0GraphNode *fcNode1 = [self addFullyConnectedLayerWithParent:poolNode
                                                     inputChannels:inputChannels
                                                    outputChannels:seFcOutputs
                                                           weights:weights1
                                                            biases:biases1
                                                        activation:@"relu"
                                                             label:label];

    // 3. FC Layer 2.
    Lc0GraphNode *fcNode2 = [self addFullyConnectedLayerWithParent:fcNode1
                                                  inputChannels:seFcOutputs
                                                 outputChannels:2 * inputChannels
                                                        weights:weights2
                                                         biases:biases2
                                                     activation:nil
                                                          label:label];
    // 4. Multiply and add.
    SEMultiply *multiply = [[SEMultiply alloc] initWithDevice:device gammaChannels:inputChannels betaChannels:inputChannels];
    
    NSLog(@"SE multiply kernel - %@", multiply);
    
    Lc0GraphNode * multiplyNode = [Lc0GraphNode graphNodeWithCnnKernel:multiply parents:@[fcNode2]];
    graphNodes = [graphNodes arrayByAddingObject:multiplyNode];
    fcNode2.numChildren++;
    
    return multiplyNode;
    
}

-(nonnull const char *) getTestData:(char * __nullable)data {
    return [[NSString stringWithFormat:@"Test data: %s", data] UTF8String];
}

@end

