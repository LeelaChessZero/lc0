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

#import "NetworkExecutor.h"
#import "ConvWeights.h"

@interface Lc0NetworkExecutor()



@end


@implementation Lc0NetworkExecutor

-(nonnull instancetype) initWithDevice:(id<MTLDevice> __nonnull)inputDevice
                          commandQueue:(id<MTLCommandQueue> __nonnull)commandQueue {
    
    self = [super init];
    device = inputDevice;
    queue = commandQueue;
    
    //MPSNNFilterNode *finalNode = [self buildInferenceGraph];
    
    //inferenceGraph = [[MPSNNGraph alloc] initWithDevice:inputDevice resultImage:finalNode.resultImage resultImageIsNeeded:YES];
    //inferenceGraph.format = fcFormat;
    
    return self;
}

-(nonnull id<MTLDevice>) getDevice {
    return device;
}

-(nonnull MPSImageBatch *) runInferenceWithInputs:(void * __nonnull)inputs batchSize:(int)batchSize {
    MPSImageDescriptor *inputDesc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatUnorm8
                                   width:8
                                  height:8
                         featureChannels:1
                          numberOfImages:1 /* BATCH_SIZE */
                                   usage:MTLTextureUsageShaderRead];
    
    // Create an input MPSImageBatch.
//    MPSImageBatch *inputBatch = @[];
//    for(NSUInteger i = 0; i < 1000 /* BATCH_SIZE */; i++){
//        MPSImage *inputImage = [[MPSImage alloc] initWithDevice:device
//                                                imageDescriptor:inputDesc];
//        inputBatch = [inputBatch arrayByAddingObject: inputImage];
//    }
//
//    [inferenceGraph executeAsyncWithSourceImages: [inputImage]
//                                        callback:^(outputImage, error) {
//        if let image = outputImage {
//            self.computeBoundingBoxes(image)
//        }
//    }
}

-(nonnull MPSNNFilterNode *) buildInferenceGraph {
    
    BOOL isConvolutionPolicyHead = YES;
    BOOL isWdl = NO;
    BOOL hasMlh = NO;
    NSUInteger inputSize = 112;
    NSUInteger channelSize = 128;
    NSUInteger kernelSize = 3;
    
    // @todo initialize array of weight objects.
    allWeights = @[];
    ConvWeights *weights;

    // 1. Input layer
    allWeights = [allWeights arrayByAddingObject:[ConvWeights alloc]];
    MPSNNFilterNode *layer = [self convolutionBlockWithSource:[MPSNNImageNode nodeWithHandle:nil]
                                                 inputChannels:inputSize
                                                outputChannels:channelSize
                                                   kernelWidth:kernelSize
                                                  kernelHeight:kernelSize
                                                       weights:allWeights[0]
                                                      hasRelu:YES
                                                         label:@"input"];
    
    // 2. Residual blocks
    for (int i = 1; i < 21; i+=2) {
        allWeights = [allWeights arrayByAddingObjectsFromArray:@[[ConvWeights alloc], [ConvWeights alloc]]];
        layer = [self residualBlockWithSource:layer.resultImage
                                                    inputChannels:channelSize
                                                   outputChannels:channelSize
                                                      kernelWidth:kernelSize
                                                     kernelHeight:kernelSize
                                                         weights1:allWeights[i]
                                                         weights2:allWeights[i+1]
                                                        seWeights:NULL
                                                            label:[NSString stringWithFormat:@"resblock_%d", i]];
    }
    
    // 3. Policy head.
    MPSNNFilterNode *policy;
    if (isConvolutionPolicyHead) {
        allWeights = [allWeights arrayByAddingObject:[ConvWeights alloc]];
        policy = [self convolutionBlockWithSource:layer.resultImage
                                                    inputChannels:channelSize
                                                   outputChannels:channelSize
                                                      kernelWidth:kernelSize
                                                     kernelHeight:kernelSize
                                                          weights:allWeights[21]
                                                           hasRelu:YES
                                                            label:@"policy/conv1"];
        
        
        allWeights = [allWeights arrayByAddingObject:[ConvWeights alloc]];
        policy = [self convolutionBlockWithSource:policy.resultImage
                                                     inputChannels:channelSize
                                                    outputChannels:80
                                                       kernelWidth:kernelSize
                                                      kernelHeight:kernelSize
                                                           weights:allWeights[22]
                                                           hasRelu:NO
                                                             label:@"policy/conv2"];
    
    }
    else {
        NSUInteger policySize = 1024;
        allWeights = [allWeights arrayByAddingObject:[ConvWeights alloc]];
        policy = [self convolutionBlockWithSource:layer.resultImage
                                                     inputChannels:channelSize
                                                    outputChannels:policySize
                                                       kernelWidth:1
                                                      kernelHeight:1
                                                           weights:allWeights[21]
                                                           hasRelu:YES
                                                             label:@"policy/conv"];
        
        // Flatten and dense;
        policy = [MPSNNReshapeNode nodeWithSource:policy.resultImage
                                                       resultWidth:1
                                                      resultHeight:1
                                             resultFeatureChannels:policySize * 8 * 8];
        
        allWeights = [allWeights arrayByAddingObject:[ConvWeights alloc]];
        [allWeights[22] initWithDevice:device
                                     inputChannels:1024
                                     outputChannels:10
                                        kernelWidth:1
                                       kernelHeight:1
                                             stride:1
                                              label:@"conv2"];
        policy = [MPSCNNFullyConnectedNode nodeWithSource:policy.resultImage weights:allWeights[22]];
    }

    
    // 4. Value head.
    if (isWdl) {
        
    }
    else {
        allWeights = [allWeights arrayByAddingObject:[ConvWeights alloc]];
        [allWeights[23] initWithDevice:device
                                     inputChannels:1024
                                    outputChannels:10
                                       kernelWidth:1
                                      kernelHeight:1
                                            stride:1
                                             label:@"conv2"];
        MPSCNNFullyConnectedNode *fc2Node = [MPSCNNFullyConnectedNode nodeWithSource:layer.resultImage
                                                                             weights:allWeights[23]];
        MPSCNNNeuronReLUNode *relu3 = [MPSCNNNeuronReLUNode nodeWithSource:fc2Node.resultImage a:0.f];
        MPSNNFilterNode *f2InputNode = relu3;
    }
    
    // 5. Moves left head.
    if (hasMlh) {
        allWeights = [allWeights arrayByAddingObject:[ConvWeights alloc]];
        [allWeights[24] initWithDevice:device
                                     inputChannels:1024
                                    outputChannels:10
                                       kernelWidth:1
                                      kernelHeight:1
                                            stride:1
                                             label:@"conv2"];
        MPSCNNFullyConnectedNode *fc1Node = [MPSCNNFullyConnectedNode nodeWithSource:layer.resultImage
                                                                             weights:allWeights[24]];
        MPSCNNNeuronReLUNode *relu3 = [MPSCNNNeuronReLUNode nodeWithSource:fc1Node.resultImage a:0.f];
        MPSNNFilterNode *f2InputNode = relu3;
    }
    
    return policy;
}

-(MPSImageBatch * __nullable) encodeInferenceBatchToCommandBuffer:(id <MTLCommandBuffer> __nonnull) commandBuffer
                                                     sourceImages:(MPSImageBatch * __nonnull) sourceImage{
    
    MPSImageBatch *returnImage = [inferenceGraph encodeBatchToCommandBuffer:commandBuffer
                                                               sourceImages:@[sourceImage]
                                                               sourceStates:nil
                                                         intermediateImages:nil
                                                          destinationStates:nil];
    
    return returnImage;
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
    if (activation == @"softmax") {
        MPSCNNSoftMaxNode *softmax = [MPSCNNSoftMaxNode nodeWithSource:fcNode.resultImage];
        return softmax;
    }
    
    if (activation == @"relu") {
        MPSCNNNeuronReLUNode *relu = [MPSCNNNeuronReLUNode nodeWithSource:fcNode.resultImage a:0.f];
        return relu;
    }
    
    if (activation == @"tanh") {
        MPSCNNNeuronTanHNode *tanh = [MPSCNNNeuronTanHNode nodeWithSource:fcNode.resultImage];
        return tanh;
    }
    
    return fcNode;
}

-(nonnull const char *) getTestData:(char * __nullable)data {
    return [[NSString stringWithFormat:@"Test data: %s", data] UTF8String];
}

@end

