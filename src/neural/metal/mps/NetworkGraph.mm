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

#import <vector>

#import "Utilities.h"

static MPSGraphConvolution2DOpDescriptor * __nonnull convolution2DDescriptor = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:1
                                                                                                             strideInY:1
                                                                                                       dilationRateInX:1
                                                                                                       dilationRateInY:1
                                                                                                                groups:1
                                                                                                          paddingStyle:MPSGraphPaddingStyleTF_SAME
                                                                                                            dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                                                                         weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

static MPSGraphPooling2DOpDescriptor * __nonnull averagePoolingDescriptor = [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:8
                                                                                                                        kernelHeight:8
                                                                                                                           strideInX:8
                                                                                                                           strideInY:8
                                                                                                                        paddingStyle:MPSGraphPaddingStyleTF_VALID
                                                                                                                          dataLayout:MPSGraphTensorNamedDataLayoutNCHW];

static const NSUInteger kNumPolicyOutputs = 1858;

static const NSUInteger kBatchesPerSplit = 10;

static const NSUInteger kMaxInflightBuffers = 2;

@implementation MPSGraphTensor(Lc0Extensions)

-(NSUInteger) size {
    NSUInteger size = 1;
    for (NSNumber * dim in self.shape) {
        size *= [dim intValue];
    }
    return size;
}

-(NSUInteger) sizeOfDimensions:(NSArray<NSNumber *> *)dimensions {
    NSUInteger size = 1;
    for (NSNumber * dim in dimensions) {
        if ([dim intValue] < [self.shape count])
            size *= [self.shape[[dim intValue]] intValue];
    }
    return size;
}

@end

@implementation Lc0NetworkGraph

-(nonnull instancetype) initWithDevice:(id<MTLDevice> __nonnull)inputDevice
                          commandQueue:(id<MTLCommandQueue> __nonnull)commandQueue
{
    
    self = [super init];
    device = inputDevice;
    queue = commandQueue;
    graph = [[MPSGraph alloc] init];
    resultTensors = @[];
    
    doubleBufferingSemaphore = dispatch_semaphore_create(kMaxInflightBuffers);
//    dynamicDataBuffers = @[];
//    currentFrameIndex = 0;
    
    // MTLResourceOptions bufferOptions = /* ... */;
//    NSMutableArray *mutableDynamicDataBuffers = [NSMutableArray arrayWithCapacity:kMaxInflightBuffers];
//    for(int i = 0; i < kMaxInflightBuffers; i++)
//    {
//        // Create a new buffer with enough capacity to store one instance of the dynamic buffer data
//        id <MTLBuffer> buffer = [inputDevice newBufferWithLength:sizeof(DynamicBufferData) options:bufferOptions];
//        [mutableDynamicDataBuffers addObject:dynamicDataBuffer];
//    }
//    _dynamicDataBuffers = [mutableDynamicDataBuffers copy];
    
    return self;
}

-(nonnull id<MTLDevice>) getDevice {
    return device;
}

-(nonnull NSArray<MPSGraphTensor *> *) runInferenceWithBatchSize:(NSUInteger)batchSize
                                                          inputs:(float * __nonnull)inputs
                                                   inputChannels:(NSUInteger)inputPlanes
                                                   outputBuffers:(float * * __nonnull)outputBuffers
{
    NSLog(@"Batchsize: %u", batchSize);
    //NSLog(@"Size of input dimensions: %lu", [inputTensor sizeOfDimensions:@[@1,@2,@3]] * batchSize);
    NSUInteger splits = (batchSize + kBatchesPerSplit - 1) / kBatchesPerSplit;
    NSUInteger inputDataLength = [inputTensor size];
    
    // Keeping track of latest command buffer.
    MPSCommandBuffer * latestCommandBuffer = nil;
    
    // Split batchSize into smaller sub-batches and run using double-buffering.
    for (NSUInteger step = 0; step < splits; step++) {
        
        // Double buffering semaphore to correctly double buffer iterations.
        dispatch_semaphore_wait(doubleBufferingSemaphore, DISPATCH_TIME_FOREVER);
        
        // Create command buffer for this sub-batch.
        MPSCommandBuffer *commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
        
        NSData * inputData = [NSData dataWithBytesNoCopy:inputs + step * inputDataLength
                                                  length:inputDataLength * sizeof(float)
                                            freeWhenDone:NO];
        
        MPSGraphTensorData * inputTensorData = [[MPSGraphTensorData alloc] initWithDevice:device
                                                                                     data:inputData
                                                                                    shape:inputTensor.shape
                                                                                 dataType:inputTensor.dataType];
        
        // Create execution descriptor with block to update results for each iteration.
        MPSGraphExecutionDescriptor * executionDescriptor = [[MPSGraphExecutionDescriptor alloc] init];
        executionDescriptor.completionHandler = ^(MPSGraphTensorDataDictionary * resultsDictionary, NSError * error) {
            MPSNDArray * array;
            for (NSUInteger rsIdx = 0; rsIdx < [resultTensors count]; rsIdx++) {
                NSUInteger outputDataLength = [resultTensors[rsIdx] size];
                // @todo: read directly from the DataTensor.
                array = [resultsDictionary[resultTensors[rsIdx]] mpsndarray];
                NSLog(@"mpsndarray: %@ %i dimensions; output data length %i; step %i", array, [array numberOfDimensions], outputDataLength, step);
                for (int k=0;k< [array numberOfDimensions]; k++) {
                    NSLog(@"dimension %i => %i", k, [array lengthOfDimension:k]);
                }
                [[resultsDictionary[resultTensors[rsIdx]] mpsndarray] readBytes:outputBuffers[rsIdx] + step * outputDataLength strideBytes:nil];
                
                NSLog(@"Output mems, %f, %f, %f, %f", outputBuffers[rsIdx][0], outputBuffers[rsIdx][1], outputBuffers[rsIdx][2], outputBuffers[rsIdx][3]);
            }
            
            // Release double buffering semaphore for the next training iteration to be encoded.
            dispatch_semaphore_signal(doubleBufferingSemaphore);
            
        };

        [graph encodeToCommandBuffer:commandBuffer
                               feeds:@{inputTensor : inputTensorData}
                       targetTensors:resultTensors
                    targetOperations:nil
                 executionDescriptor:executionDescriptor];

        //[graph runWithFeeds:@{inputTensor : inputTensorData} targetTensors:resultTensors targetOperations:nil];

        // Commit the command buffer
        [commandBuffer commit];
        latestCommandBuffer = commandBuffer;


    }
    // Wait for the last batch to be processed.
    [latestCommandBuffer waitUntilCompleted];
    
    NSLog(@"Finished inference");
    
    return resultTensors;
}

-(void) setResultTensors:(NSArray<MPSGraphTensor *> * __nonnull)results
{
    resultTensors = results;
}

-(nonnull MPSGraphTensor *) inputPlaceholderWithMaxBatch:(NSUInteger)maxBatchSize
                                           inputChannels:(NSUInteger)channels
                                                  height:(NSUInteger)height
                                                   width:(NSUInteger)width
                                                   label:(NSString * __nullable)label
{
    // Create a placeholder tensor that can hold the specified number of sub-batches.
    inputTensor = [graph placeholderWithShape:@[@(kBatchesPerSplit), @(channels), @(height), @(width)] name:label];
    
    return inputTensor;
}

-(nonnull MPSGraphTensor *) addConvolutionBlockWithParent:(MPSGraphTensor * __nonnull)parent
                                            inputChannels:(NSUInteger)inputChannels
                                           outputChannels:(NSUInteger)outputChannels
                                               kernelSize:(NSUInteger)kernelSize
                                                  weights:(float * __nonnull)weights
                                                   biases:(float * __nonnull)biases
                                                  hasRelu:(BOOL)hasRelu
                                                    label:(NSString * __nonnull)label
{
    NSData * weightsData = [NSData dataWithBytesNoCopy:weights
                                                length:outputChannels * inputChannels * kernelSize * kernelSize * sizeof(float)
                                          freeWhenDone:NO];

    MPSGraphTensor * weightsTensor = [graph variableWithData:weightsData
                                                       shape:@[@(outputChannels), @(inputChannels), @(kernelSize), @(kernelSize)]
                                                    dataType:MPSDataTypeFloat32
                                                        name:[NSString stringWithFormat:@"%@/weights", label]];
    
    NSData * biasData = [NSData dataWithBytesNoCopy:biases
                                             length:outputChannels * sizeof(float)
                                       freeWhenDone:NO];

    MPSGraphTensor * biasTensor = [graph variableWithData:biasData
                                                    shape:@[@(outputChannels), @1, @1]
                                                 dataType:MPSDataTypeFloat32
                                                     name:[NSString stringWithFormat:@"%@/biases", label]];
    
    MPSGraphTensor * convTensor = [graph convolution2DWithSourceTensor:parent
                                                         weightsTensor:weightsTensor
                                                            descriptor:convolution2DDescriptor
                                                                  name:[NSString stringWithFormat:@"%@/conv", label]];
    
    MPSGraphTensor * convBiasTensor = [graph additionWithPrimaryTensor:convTensor
                                                       secondaryTensor:biasTensor
                                                                  name:[NSString stringWithFormat:@"%@/bias_add", label]];

    if (hasRelu) {
        MPSGraphTensor * reluTensor = [graph reLUWithTensor:convBiasTensor name:[NSString stringWithFormat:@"%@/relu", label]];
        return reluTensor;
    }

    return convBiasTensor;
}

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
{
    
    MPSGraphTensor * conv1Tensor = [self addConvolutionBlockWithParent:parent
                                                         inputChannels:inputChannels
                                                        outputChannels:outputChannels
                                                            kernelSize:kernelSize
                                                               weights:weights1
                                                                biases:biases1
                                                               hasRelu:YES
                                                                 label:[NSString stringWithFormat:@"%@/conv1", label]];
    
    MPSGraphTensor * conv2Tensor = [self addConvolutionBlockWithParent:conv1Tensor
                                                         inputChannels:inputChannels
                                                        outputChannels:outputChannels
                                                            kernelSize:kernelSize
                                                               weights:weights2
                                                                biases:biases2
                                                               hasRelu:NO
                                                                 label:[NSString stringWithFormat:@"%@/conv2", label]];
    
    if (hasSe) {
        // SE Unit.
        MPSGraphTensor * seUnit = [self addSEUnitWithParent:conv2Tensor
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
        MPSGraphTensor * residualTensor = [graph additionWithPrimaryTensor:parent
                                                           secondaryTensor:conv2Tensor
                                                                      name:[NSString stringWithFormat:@"%@/add", label]];
        
        MPSGraphTensor * reluTensor = [graph reLUWithTensor:residualTensor
                                                       name:[NSString stringWithFormat:@"%@/relu", label]];
        return reluTensor;
    }
}

-(nonnull MPSGraphTensor *) addFullyConnectedLayerWithParent:(MPSGraphTensor * __nonnull)parent
                                             inputChannels:(NSUInteger)inputChannels
                                            outputChannels:(NSUInteger)outputChannels
                                                   weights:(float * __nonnull)weights
                                                    biases:(float * __nonnull)biases
                                                activation:(NSString * __nullable)activation
                                                     label:(NSString * __nonnull)label
{
    NSData * weightData = [NSData dataWithBytesNoCopy:weights
                                               length:outputChannels * inputChannels * sizeof(float)
                                         freeWhenDone:NO];
    
    MPSGraphTensor * weightTensor = [graph variableWithData:weightData
                                                      shape:@[@(outputChannels), @(inputChannels)]
                                                   dataType:MPSDataTypeFloat32
                                                       name:[NSString stringWithFormat:@"%@/weights", label]];
    
    // Leela weights are OIHW, need to be transposed to IO** to allow matmul.
    MPSGraphTensor * transposeTensor = [graph transposeTensor:weightTensor
                                                    dimension:0
                                                withDimension:1
                                                         name:[NSString stringWithFormat:@"%@/weights_transpose", label]];
    
    MPSGraphTensor * reshaped = [graph reshapeTensor:parent
                                           withShape:@[parent.shape[0], @([parent sizeOfDimensions:@[@1, @2, @3]])]
                                                name:[NSString stringWithFormat:@"%@/reshape", label]];

    MPSGraphTensor * fcTensor = [graph matrixMultiplicationWithPrimaryTensor:reshaped
                                                             secondaryTensor:transposeTensor
                                                                        name:[NSString stringWithFormat:@"%@/matmul", label]];
    
    NSData * biasData = [NSData dataWithBytesNoCopy:biases
                                             length:outputChannels * sizeof(float)
                                       freeWhenDone:NO];
    
    MPSGraphTensor * biasTensor = [graph variableWithData:biasData
                                                    shape:@[@(outputChannels)]
                                                 dataType:MPSDataTypeFloat32
                                                     name:[NSString stringWithFormat:@"%@/biases", label]];
    
    MPSGraphTensor * addTensor = [graph additionWithPrimaryTensor:fcTensor
                                                  secondaryTensor:biasTensor
                                                             name:[NSString stringWithFormat:@"%@/bias_add", label]];
    
    
    if ([activation isEqual:@"relu"]) {
        return [graph reLUWithTensor:addTensor name:[NSString stringWithFormat:@"%@/relu", label]];
    }
    else if ([activation isEqual:@"tanh"]) {
        return [graph tanhWithTensor:addTensor name:[NSString stringWithFormat:@"%@/tanh", label]];
    }
    else if ([activation isEqual:@"sigmoid"]) {
        return [graph sigmoidWithTensor:addTensor name:[NSString stringWithFormat:@"%@/sigmoid", label]];
    }
    else if ([activation isEqual:@"softmax"]) {
        NSLog(@"Doing softmax");
        return [graph softMaxWithTensor:addTensor axis:1 name:[NSString stringWithFormat:@"%@/softmax", label]];
    }

    return addTensor;
}

-(nonnull MPSGraphTensor *) addSEUnitWithParent:(MPSGraphTensor * __nonnull)parent
                                       skipNode:(MPSGraphTensor * __nonnull)skipTensor
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
    MPSGraphTensor * poolTensor = [graph avgPooling2DWithSourceTensor:parent
                                                         descriptor:averagePoolingDescriptor
                                                               name:[NSString stringWithFormat:@"%@/pool", label]];
    
    // 2. FC Layer 1.
    MPSGraphTensor * fc1Tensor = [self addFullyConnectedLayerWithParent:poolTensor
                                                     inputChannels:inputChannels
                                                    outputChannels:seFcOutputs
                                                           weights:weights1
                                                            biases:biases1
                                                        activation:@"relu"
                                                             label:[NSString stringWithFormat:@"%@/fc1", label]];

    // 3. FC Layer 2.
    MPSGraphTensor * fc2Tensor = [self addFullyConnectedLayerWithParent:fc1Tensor
                                                  inputChannels:seFcOutputs
                                                 outputChannels:2 * inputChannels
                                                        weights:weights2
                                                         biases:biases2
                                                     activation:nil
                                                          label:[NSString stringWithFormat:@"%@/fc2", label]];
    
    // 4. Slice 1 and gamma.
    MPSGraphTensor * slice1Tensor = [graph sliceTensor:fc2Tensor
                                             dimension:1
                                                 start:0
                                                length:inputChannels
                                                  name:[NSString stringWithFormat:@"%@/slice1", label]];
    
    MPSGraphTensor * gammaTensor = [graph sigmoidWithTensor:slice1Tensor
                                                       name:[NSString stringWithFormat:@"%@/sigmoid", label]];
    
    // 5. Slice 2
    MPSGraphTensor * slice2Tensor = [graph sliceTensor:fc2Tensor
                                             dimension:1
                                                 start:inputChannels
                                                length:inputChannels
                                                  name:[NSString stringWithFormat:@"%@/slice2", label]];
    
    // 5. Multiply and add.
    MPSGraphTensor * reshape1Tensor = [graph reshapeTensor:gammaTensor
                                           withShape:@[gammaTensor.shape[0], gammaTensor.shape[1], @1, @1]
                                                name:[NSString stringWithFormat:@"%@/reshape1", label]];
    
    
    MPSGraphTensor * multiplyTensor = [graph multiplicationWithPrimaryTensor:parent
                                                             secondaryTensor:reshape1Tensor
                                                                        name:[NSString stringWithFormat:@"%@/multiply", label]];
    
    MPSGraphTensor * reshape2Tensor = [graph reshapeTensor:slice2Tensor
                                                 withShape:@[slice2Tensor.shape[0], slice2Tensor.shape[1], @1, @1]
                                                      name:[NSString stringWithFormat:@"%@/reshape2", label]];

    
    // Two addition operations in series causes crashes on tensors with channels > 64.
    // Don't understand why. Might be due to failed attempts to fuse both add operations.
    // Using this multiply by 1 to interpose and prevent attempt to fuse.
//    float * ones = (float *)malloc(sizeof(float));
//    *ones = 1.0;
//    NSData * onesData = [NSData dataWithBytesNoCopy:ones length:sizeof(float)];
//
//    MPSGraphTensor * onesTensor = [graph constantWithData:onesData
//                                                    shape:@[@1, @1, @1]
//                                                 dataType:MPSDataTypeFloat32];
//
//    MPSGraphTensor * dummyTensor = [graph multiplicationWithPrimaryTensor:onesTensor
//                                                          secondaryTensor:skipTensor
//                                                     name:[NSString stringWithFormat:@"%@/dummy", label]];

    MPSGraphTensor * add1Tensor = [graph additionWithPrimaryTensor:multiplyTensor
                                                   secondaryTensor:reshape2Tensor
                                                              name:[NSString stringWithFormat:@"%@/add1", label]];
    
    MPSGraphTensor * add2Tensor = [graph additionWithPrimaryTensor:add1Tensor
                                                   secondaryTensor:skipTensor
                                                              name:[NSString stringWithFormat:@"%@/add2", label]];

    
    // 6. ReLU
    MPSGraphTensor * reluTensor = [graph reLUWithTensor:add2Tensor
                                                   name:[NSString stringWithFormat:@"%@/relu", label]];

    NSLog(@"Shapes: multiply %@, reshape2 %@, add1 %@, skip %@, add2 %@, relu %@, slice1 %@, slice2 %@", multiplyTensor.shape, reshape2Tensor.shape, add2Tensor.shape, skipTensor.shape, add2Tensor.shape, reluTensor.shape, slice1Tensor.shape, slice2Tensor.shape);
          
    return reluTensor;
}

-(nonnull MPSGraphTensor *) addPolicyMapLayerWithParent:(MPSGraphTensor * __nonnull)parent
                                            policyMap:(uint32_t * __nonnull)policyMap
                                                  label:(NSString *)label
{
    
    NSData * policyMapData = [NSData dataWithBytesNoCopy:policyMap
                                                  length:kNumPolicyOutputs * sizeof(uint32_t)
                                            freeWhenDone:NO];

    MPSGraphTensor * mappingTensor = [graph constantWithData:policyMapData
                                                       shape:@[@(kNumPolicyOutputs)]
                                                    dataType:MPSDataTypeUInt32];

    MPSGraphTensor * flatConvTensor = [graph reshapeTensor:parent
                                                 withShape:@[parent.shape[0], @([parent sizeOfDimensions:@[@1, @2, @3]])]
                                                      name:[NSString stringWithFormat:@"%@/flatten", label]];

    return [graph gatherWithUpdatesTensor:flatConvTensor
                            indicesTensor:mappingTensor
                                     axis:1
                          batchDimensions:0
                                     name:[NSString stringWithFormat:@"%@/gather", label]];
}

@end

