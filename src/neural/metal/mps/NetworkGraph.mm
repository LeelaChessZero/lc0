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


@implementation Lc0NetworkGraph

-(nonnull instancetype) initWithDevice:(id<MTLDevice> __nonnull)inputDevice
                          commandQueue:(id<MTLCommandQueue> __nonnull)commandQueue {
    
    self = [super init];
    device = inputDevice;
    queue = commandQueue;
    
    return self;
}

-(nonnull id<MTLDevice>) getDevice {
    return device;
}

-(nonnull NSMutableArray<MPSImageBatch*> *) runInferenceWithBatchSize:(NSUInteger)batchSize
                                                           masks:(uint64_t * __nonnull)masks
                                                          values:(float * __nonnull)values
                                                   inputChannels:(NSUInteger)inputPlanes
                                                    subBatchSize:(NSUInteger)subBatchSize
{
    const uint8_t boardWidth = 8;
    const uint8_t boardHeight = 8;
    
    @autoreleasepool {
        MPSImageDescriptor *inputDesc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                                                                       width:boardWidth
                                                                                      height:boardHeight
                                                                             featureChannels:inputPlanes
                                                                              numberOfImages:subBatchSize
                                                                                       usage:MTLTextureUsageShaderRead];
        
        //NSLog(@"Inputplanes: %i, batchSize: %i, subBatchSize: %i", inputPlanes, batchSize, subBatchSize);
        // Use double buffering to keep the GPU completely busy.
        dispatch_semaphore_t doubleBufferingSemaphore = dispatch_semaphore_create(2);
        dispatch_semaphore_wait(doubleBufferingSemaphore, DISPATCH_TIME_FOREVER);
        
        // Create batches of MPSImages that each contain multiple images (i.e. multiple board positions
        // in sub-batches) in order to optimize GPU usage.
        
        //NSLog(@"sizeof(float): %i", sizeof(float));
        int bytesPerRow = 8 * sizeof(float);
        
        // Buffer for expanding packed planes.
        float buffer[inputPlanes * boardWidth * boardHeight];
        
        // Create an input MPSImageBatch.
        MPSImageBatch *inputBatch = @[];
        for (int subBatch = 0; subBatch < batchSize; subBatch += subBatchSize) {

            MPSImage *inputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:inputDesc];
                    
            //float after[inputPlanes * boardWidth * boardHeight];
            
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
                        //NSLog(@"Expanded image data: batch: %d, plane: %d, square: %d; value: %f", i, j, k, *(dptr++));
                    }
                }
                //NSLog(@"Texture copy before assignment");
                //logBeforeAfter(buffer, after, inputPlanes);

                [inputImage writeBytes:buffer
                            dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth
                            imageIndex:i - subBatch];
                
                //[inputImage.texture getBytes:after bytesPerRow:bytesPerRow fromRegion:MTLRegionMake2D(0,0,8,8) mipmapLevel:0];
                
                //[inputImage readBytes:after
                //           dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth
                //           imageIndex:i];
                
                //NSLog(@"Completed subbatch %i of %i of %i batches", i, subBatchEnd, batchSize);
    //            NSLog(@"Texture copy before assignment");
    //            NSLog(listOfFloats(buffer, 8*8*inputPlanes));
                
                //NSLog(@"Texture copy before >> after assignment");
                //logBeforeAfter(buffer, after, inputPlanes);
                    
            }
            // Add image to input batch.
            inputBatch = [inputBatch arrayByAddingObject:inputImage];
        }
        
        /*// Write the image data from the CPU image area to the GPU image batch.
        [inputBatch enumerateObjectsUsingBlock:^(MPSImage * __nonnull inputImage, NSUInteger idx, BOOL * __nonnull stop) {
            // Advance image pointer in the supplied inputs and copy into the GPU.
            uint8_t *start = (uint8_t *) ADVANCE_PTR(inputs, inputOffset + (boardWidth * boardHeight * idx));
            [inputImage writeBytes:start
                        dataLayout:(MPSDataLayoutFeatureChannelsxHeightxWidth)
                        imageIndex:0];
        }];*/
        
        // Make an MPSCommandBuffer, when passed to the encode of MPSNNGraph, commitAndContinue will be automatically used.
        //NSLog(@"Initializing command buffer");
        MPSCommandBuffer *commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
        
        // Encode inference network
        //NSLog(@"Encoding command buffer");
        NSMutableArray<MPSImageBatch*> *results = [[NSMutableArray alloc] init];
        MPSImageBatch *output = [graph encodeBatchToCommandBuffer:commandBuffer
                                                     sourceImages:@[inputBatch]
                                                     sourceStates:nil
                                               intermediateImages:results
                                                destinationStates:nil];

        // Transfer data from GPU to CPU.
        MPSImageBatchSynchronize(output, commandBuffer);
    
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> __nonnull) {
            // Release double buffering semaphore for the iteration to be encoded.
            dispatch_semaphore_signal(doubleBufferingSemaphore);

//            // Check the output of inference network to calculate accuracy.
//            [outputBatch enumerateObjectsUsingBlock:^(MPSImage * __nonnull outputImage, NSUInteger idx, BOOL * _Nonnull stop) {
//                /*uint8_t *start = ADVANCE_PTR(outputs, inputOffset + (boardWidth * boardHeight * idx));
//                [inputImage writeBytes:start
//                            dataLayout:(MPSDataLayoutFeatureChannelsxHeightxWidth)
//                            imageIndex:0];*/
//            }];
//
        }];
        
        // Commit the command buffer. Wait for the last batch to be processed.
        //NSLog(@"Committing command buffer");
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Add the first result at the beginning of the list of result images.
        [results insertObject:output atIndex:0];
        NSLog(@"Got results: batchSize: %i, count: %i", batchSize, [results count]);
        return results;
    }
}

-(nonnull MPSNNGraph *) buildGraphWithResultImages:(NSArray <MPSNNImageNode *> * __nonnull)resultImages
                                  resultsAreNeeded:(BOOL * __nullable)areResultsNeeded
{
    
    graph = [MPSNNGraph graphWithDevice:device resultImages:resultImages resultsAreNeeded:areResultsNeeded];
    graph.format = fcFormat;
        
    return graph;
}

-(nonnull MPSNNFilterNode *) convolutionBlockWithSource:(MPSNNImageNode * __nonnull)input
                                          inputChannels:(NSUInteger)inputChannels
                                         outputChannels:(NSUInteger)outputChannels
                                            kernelWidth:(NSUInteger)kernelWidth
                                           kernelHeight:(NSUInteger)kernelHeight
                                                weights:(ConvWeights * __nonnull)weights
                                                hasRelu:(BOOL)hasRelu
{
    MPSCNNConvolutionNode *convNode = [MPSCNNConvolutionNode nodeWithSource:input weights:weights];
    convNode.paddingPolicy = sameConvPadding;
    
    if (hasRelu) {
        MPSCNNNeuronReLUNode *relu = [MPSCNNNeuronReLUNode nodeWithSource:convNode.resultImage a:0.f];
        //MPSCNNPoolingMaxNode *pool1 = [MPSCNNPoolingMaxNode nodeWithSource:relu1.resultImage filterSize:2 stride:2];
        //pool1.paddingPolicy = samePoolingPadding;
        
        // @todo Batch norm.
        
        return relu;
    }
    return convNode;
}

-(nonnull MPSNNFilterNode *) residualBlockWithSource:(MPSNNImageNode * __nonnull)input
                                       inputChannels:(NSUInteger)inputChannels
                                      outputChannels:(NSUInteger)outputChannels
                                         kernelWidth:(NSUInteger)kernelWidth
                                        kernelHeight:(NSUInteger)kernelHeight
                                            weights1:(ConvWeights * __nonnull)weights1
                                            weights2:(ConvWeights * __nonnull)weights2
                                           seWeights:(ConvWeights * __nullable)seWeights
{
    // Conv1
    MPSCNNConvolutionNode *conv1Node = [MPSCNNConvolutionNode nodeWithSource:input weights:weights1];
    conv1Node.paddingPolicy = sameConvPadding;
    MPSCNNNeuronReLUNode *relu1 = [MPSCNNNeuronReLUNode nodeWithSource:conv1Node.resultImage a:0.f];
    
    // Conv2
    MPSCNNConvolutionNode *conv2Node = [MPSCNNConvolutionNode nodeWithSource:relu1.resultImage weights:weights2];
    conv2Node.paddingPolicy = sameConvPadding;
    
    if (seWeights) {
        // @todo add SE unit here.
    }
    
    MPSNNAdditionNode *add = [MPSNNAdditionNode nodeWithLeftSource:relu1.resultImage
                                                       rightSource:conv2Node.resultImage];
    
    MPSCNNNeuronReLUNode *relu2 = [MPSCNNNeuronReLUNode nodeWithSource:add.resultImage a:0.f];
    
    return relu2;
    
}

-(nonnull MPSNNFilterNode *) fullyConnectedLayerWithSource:(MPSNNImageNode * __nonnull)input
                                                   weights:(ConvWeights * __nonnull)weights
                                                activation:(NSString * __nullable)activation
{
    MPSCNNFullyConnectedNode *fcNode = [MPSCNNFullyConnectedNode nodeWithSource:input
                                                                         weights:weights];
    if ([activation isEqual:@"softmax"]) {
        MPSCNNSoftMaxNode *softmax = [MPSCNNSoftMaxNode nodeWithSource:fcNode.resultImage];
        return softmax;
    }
    
    if ([activation isEqual:@"relu"]) {
        MPSCNNNeuronReLUNode *relu = [MPSCNNNeuronReLUNode nodeWithSource:fcNode.resultImage a:0.f];
        return relu;
    }
    
    if ([activation isEqual:@"tanh"]) {
        MPSCNNNeuronTanHNode *tanh = [MPSCNNNeuronTanHNode nodeWithSource:fcNode.resultImage];
        return tanh;
    }
    
    return fcNode;
}

-(nonnull const char *) getTestData:(char * __nullable)data {
    return [[NSString stringWithFormat:@"Test data: %s", data] UTF8String];
}

@end

