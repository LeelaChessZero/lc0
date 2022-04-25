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

static const NSUInteger kMaxInflightBuffers = 4;

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

-(nonnull instancetype) initWithDevice:(id<MTLDevice> __nonnull)device
                       batchesPerSplit:(NSUInteger)batchesPerSplit
{
    
    self = [super init];
    _device = [MPSGraphDevice deviceWithMTLDevice:device];
    _queue = [device newCommandQueue];
    _graph = [[MPSGraph alloc] init];
    _resultTensors = @[];
    _batchesPerSplit = batchesPerSplit;
    
    _doubleBufferingSemaphore = dispatch_semaphore_create(kMaxInflightBuffers);
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
    
    _readVariables = [[NSMutableDictionary alloc] init];
    
    return self;
}

-(nonnull NSArray<MPSGraphTensor *> *) runInferenceWithBatchSize:(NSUInteger)batchSize
                                                          inputs:(float * __nonnull)inputs
                                                   inputChannels:(NSUInteger)inputPlanes
{
//    NSLog(@"Batchsize: %u", batchSize);
    //NSUInteger splits = (batchSize + _batchesPerSplit - 1) / _batchesPerSplit;
    NSUInteger inputDataLength = [_inputTensor sizeOfDimensions:@[@1, @2, @3]] * batchSize;
    
    // Keeping track of latest command buffer.
    //MPSCommandBuffer * latestCommandBuffer = nil;
    
    // Split batchSize into smaller sub-batches and run using double-buffering.
    //for (NSUInteger step = 0; step < splits; step++) {
        
        // Double buffering semaphore to correctly double buffer iterations.
        dispatch_semaphore_wait(_doubleBufferingSemaphore, DISPATCH_TIME_FOREVER);
        
        // Create command buffer for this sub-batch.
        MPSCommandBuffer * commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:_queue];
        
        NSData * inputData = [NSData dataWithBytesNoCopy:inputs //+ step * inputDataLength
                                                  length:inputDataLength * sizeof(float)
                                            freeWhenDone:NO];
    
        MPSShape * inputShape = @[@(batchSize), _inputTensor.shape[1], _inputTensor.shape[2], _inputTensor.shape[3]];
        
        MPSGraphTensorData * inputTensorData = [[MPSGraphTensorData alloc] initWithDevice:_device
                                                                                     data:inputData
                                                                                    shape:inputShape
                                                                                 dataType:_inputTensor.dataType];
        
        // Create execution descriptor with block to update results for each iteration.
        MPSGraphExecutionDescriptor * executionDescriptor = [[MPSGraphExecutionDescriptor alloc] init];
        executionDescriptor.completionHandler = ^(MPSGraphTensorDataDictionary * resultsDictionary, NSError * error) {
            
            //NSLog(@"Done one sub batch %i", step);
            // Copy results for sub-batch back into the output buffers.
//            float * array = (float *)malloc(10240 * sizeof(float));
//            for (NSUInteger rsIdx = 0; rsIdx < [_resultTensors count]; rsIdx++) {
//                NSUInteger outputDataLength = [_resultTensors[rsIdx] size];
//                [[resultsDictionary[_resultTensors[rsIdx]] mpsndarray] readBytes:outputBuffers[rsIdx] //+ step * outputDataLength
//                                                                     strideBytes:nil];
//                [[resultsDictionary[_resultTensors[rsIdx]] mpsndarray] readBytes:array //+ step * outputDataLength
//                                                                     strideBytes:nil];
////                NSLog(@"Completed: _resultTensors[%i]", rsIdx);
////                for (int i=0; i<100; i++) {
////                    NSLog(@";%i;%f", i, array);
////                }
//            }
            
            // Release double buffering semaphore for the next training iteration to be encoded.
            dispatch_semaphore_signal(_doubleBufferingSemaphore);

        };

        _resultDataDictionary = [_graph encodeToCommandBuffer:commandBuffer
                                                        feeds:@{_inputTensor : inputTensorData}
                                                targetTensors:_targetTensors
                                             targetOperations:nil
                                          executionDescriptor:executionDescriptor];

        // Commit the command buffer
        [commandBuffer commit];
        //latestCommandBuffer = commandBuffer;
    //}

    // Wait for the last batch to be processed.
    //[latestCommandBuffer waitUntilCompleted];
    [commandBuffer waitUntilCompleted];
    
    float * array = (float *)malloc(10240 * sizeof(float));
    for (NSUInteger rsIdx = 0; rsIdx < [_resultTensors count]; rsIdx++) {
//        NSUInteger outputDataLength = [_resultTensors[rsIdx] size];
//        [[resultsDictionary[_resultTensors[rsIdx]] mpsndarray] readBytes:outputBuffers[rsIdx] //+ step * outputDataLength
//                                                             strideBytes:nil];
        [[_resultDataDictionary[_resultTensors[rsIdx]] mpsndarray] readBytes:array //+ step * outputDataLength
                                                             strideBytes:nil];
        NSLog(@"Completed(immediate): _resultTensors[%i]", rsIdx);
        for (int i=0; i<100; i++) {
            NSLog(@";%i;%f", i, array);
        }
    }
    
    NSLog(@"_targets: %@", _targetTensors);
    
    return _resultTensors;
}


-(void) copyResultsWithBatchSize:(NSUInteger)batchSize
                   outputBuffers:(float * __nonnull * __nonnull)outputBuffers
{
    // Copy results for sub-batch back into the output buffers.
    float * array = (float *)malloc(10240 * sizeof(float));
    for (NSUInteger rsIdx = 0; rsIdx < [_resultTensors count]; rsIdx++) {
        NSUInteger outputDataLength = [_resultTensors[rsIdx] size];
        [[_resultDataDictionary[_resultTensors[rsIdx]] mpsndarray] readBytes:outputBuffers[rsIdx] //+ step * outputDataLength
                                                                 strideBytes:nil];
        [[_resultDataDictionary[_resultTensors[rsIdx]] mpsndarray] readBytes:array //+ step * outputDataLength
                                                                 strideBytes:nil];
        NSLog(@"Completed(after): _resultTensors[%i]", rsIdx);
        for (int i=0; i<100; i++) {
            NSLog(@";%i;%f", i, array);
        }
    }
}


-(void) setResultTensors:(NSArray<MPSGraphTensor *> * __nonnull)results
{
    // Okay to remove nulls from the read variables.
    [_readVariables removeObjectsForKeys:[_readVariables allKeysForObject:[NSNull null]]];
    
    // Set the results we're interested in.
    _resultTensors = results;
    
    // Target tensor for graph is combination of both.
    _targetTensors = [NSArray arrayWithArray:_resultTensors];
    _targetTensors = [_targetTensors arrayByAddingObjectsFromArray:[_readVariables allValues]];
}

-(nonnull MPSGraphTensor *) inputPlaceholderWithInputChannels:(NSUInteger)channels
                                                       height:(NSUInteger)height
                                                        width:(NSUInteger)width
                                                        label:(NSString * __nullable)label
{
    // Create a placeholder tensor that can hold the specified number of sub-batches.
    _inputTensor = [_graph placeholderWithShape:@[@(-1), @(channels), @(height), @(width)] name:label];
    
    return _inputTensor;
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
    
    MPSGraphTensor * weightsTensor = [_graph variableWithData:weightsData
                                                        shape:@[@(outputChannels), @(inputChannels), @(kernelSize), @(kernelSize)]
                                                     dataType:MPSDataTypeFloat32
                                                         name:[NSString stringWithFormat:@"%@/weights", label]];
    
    NSData * biasData = [NSData dataWithBytesNoCopy:biases
                                             length:outputChannels * sizeof(float)
                                       freeWhenDone:NO];
    
    MPSGraphTensor * biasTensor = [_graph variableWithData:biasData
                                                     shape:@[@(outputChannels), @1, @1]
                                                  dataType:MPSDataTypeFloat32
                                                      name:[NSString stringWithFormat:@"%@/biases", label]];
    
    MPSGraphTensor * convTensor = [_graph convolution2DWithSourceTensor:parent
                                                          weightsTensor:weightsTensor
                                                             descriptor:convolution2DDescriptor
                                                                   name:[NSString stringWithFormat:@"%@/conv", label]];
    
    MPSGraphTensor * convBiasTensor = [_graph additionWithPrimaryTensor:convTensor
                                                        secondaryTensor:biasTensor
                                                                   name:[NSString stringWithFormat:@"%@/bias_add", label]];
    
    [self setVariable:[NSString stringWithFormat:@"%@/weights", label] tensor:weightsTensor];
    [self setVariable:[NSString stringWithFormat:@"%@/biases", label] tensor:biasTensor];
    [self setVariable:[NSString stringWithFormat:@"%@/conv", label] tensor:convTensor];
    [self setVariable:[NSString stringWithFormat:@"%@/bias_add", label] tensor:convBiasTensor];

    if (hasRelu) {
        MPSGraphTensor * reluTensor = [_graph reLUWithTensor:convBiasTensor name:[NSString stringWithFormat:@"%@/relu", label]];
        [self setVariable:[NSString stringWithFormat:@"%@/relu", label] tensor:reluTensor];
        [self setVariable:label tensor:reluTensor];
        return reluTensor;
    }
    
    [self setVariable:label tensor:convBiasTensor];
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
    
    [self setVariable:[NSString stringWithFormat:@"%@/conv1", label] tensor:conv1Tensor];
    [self setVariable:[NSString stringWithFormat:@"%@/conv2", label] tensor:conv2Tensor];
    
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
    
        [self setVariable:label tensor:seUnit];
        NSLog(@"SE Unit");
        return seUnit;
    }
    else {
        MPSGraphTensor * residualTensor = [_graph additionWithPrimaryTensor:parent
                                                            secondaryTensor:conv2Tensor
                                                                       name:[NSString stringWithFormat:@"%@/add", label]];

        MPSGraphTensor * reluTensor = [_graph reLUWithTensor:residualTensor
                                                        name:[NSString stringWithFormat:@"%@/relu", label]];
        NSLog(@"No SE Unit");
        [self setVariable:[NSString stringWithFormat:@"%@/add", label] tensor:residualTensor];
        [self setVariable:[NSString stringWithFormat:@"%@/relu", label] tensor:reluTensor];
        [self setVariable:label tensor:reluTensor];
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
    
    MPSGraphTensor * weightTensor = [_graph variableWithData:weightData
                                                       shape:@[@(outputChannels), @(inputChannels)]
                                                    dataType:MPSDataTypeFloat32
                                                        name:[NSString stringWithFormat:@"%@/weights", label]];
    
    // Leela weights are OIHW, need to be transposed to IO** to allow matmul.
    MPSGraphTensor * transposeTensor = [_graph transposeTensor:weightTensor
                                                     dimension:0
                                                 withDimension:1
                                                          name:[NSString stringWithFormat:@"%@/weights_transpose", label]];
    
    MPSGraphTensor * reshaped = [_graph reshapeTensor:parent
                                            withShape:@[@(-1), @([parent sizeOfDimensions:@[@1, @2, @3]])]
                                                 name:[NSString stringWithFormat:@"%@/reshape", label]];
    
    MPSGraphTensor * fcTensor = [_graph matrixMultiplicationWithPrimaryTensor:reshaped
                                                              secondaryTensor:transposeTensor
                                                                         name:[NSString stringWithFormat:@"%@/matmul", label]];
    
    NSData * biasData = [NSData dataWithBytesNoCopy:biases
                                             length:outputChannels * sizeof(float)
                                       freeWhenDone:NO];
    
    MPSGraphTensor * biasTensor = [_graph variableWithData:biasData
                                                     shape:@[@(outputChannels)]
                                                  dataType:MPSDataTypeFloat32
                                                      name:[NSString stringWithFormat:@"%@/biases", label]];
    
    MPSGraphTensor * addTensor = [_graph additionWithPrimaryTensor:fcTensor
                                                   secondaryTensor:biasTensor
                                                              name:[NSString stringWithFormat:@"%@/bias_add", label]];
    
    [self setVariable:[NSString stringWithFormat:@"%@/weights", label] tensor:weightTensor];
    [self setVariable:[NSString stringWithFormat:@"%@/weights_transpose", label] tensor:transposeTensor];
    [self setVariable:[NSString stringWithFormat:@"%@/reshape", label] tensor:reshaped];
    [self setVariable:[NSString stringWithFormat:@"%@/matmul", label] tensor:fcTensor];
    [self setVariable:[NSString stringWithFormat:@"%@/biases", label] tensor:biasTensor];
    [self setVariable:[NSString stringWithFormat:@"%@/bias_add", label] tensor:addTensor];
    
    if ([activation isEqual:@"relu"]) {
        MPSGraphTensor * reluTensor = [_graph reLUWithTensor:addTensor name:[NSString stringWithFormat:@"%@/relu", label]];
        [self setVariable:[NSString stringWithFormat:@"%@/relu", label] tensor:reluTensor];
        [self setVariable:label tensor:reluTensor];
        return reluTensor;
    }
    else if ([activation isEqual:@"tanh"]) {
        MPSGraphTensor * tanhTensor = [_graph tanhWithTensor:addTensor name:[NSString stringWithFormat:@"%@/tanh", label]];
        [self setVariable:[NSString stringWithFormat:@"%@/tanh", label] tensor:tanhTensor];
        [self setVariable:label tensor:tanhTensor];
        return tanhTensor;
    }
    else if ([activation isEqual:@"sigmoid"]) {
        MPSGraphTensor * sigmoidTensor = [_graph sigmoidWithTensor:addTensor name:[NSString stringWithFormat:@"%@/sigmoid", label]];
        [self setVariable:[NSString stringWithFormat:@"%@/sigmoid", label] tensor:sigmoidTensor];
        [self setVariable:label tensor:sigmoidTensor];
        return sigmoidTensor;
    }
    else if ([activation isEqual:@"softmax"]) {
        NSLog(@"Doing softmax");
        MPSGraphTensor * softmaxTensor = [_graph softMaxWithTensor:addTensor axis:1 name:[NSString stringWithFormat:@"%@/softmax", label]];
        [self setVariable:[NSString stringWithFormat:@"%@/softmax", label] tensor:softmaxTensor];
        [self setVariable:label tensor:softmaxTensor];
        return softmaxTensor;
    }
    
    [self setVariable:label tensor:addTensor];
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
    MPSGraphTensor * poolTensor = [_graph avgPooling2DWithSourceTensor:parent
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
    MPSGraphTensor * slice1Tensor = [_graph sliceTensor:fc2Tensor
                                              dimension:1
                                                  start:0
                                                 length:inputChannels
                                                   name:[NSString stringWithFormat:@"%@/slice1", label]];
    
    MPSGraphTensor * gammaTensor = [_graph sigmoidWithTensor:slice1Tensor
                                                        name:[NSString stringWithFormat:@"%@/sigmoid", label]];
    
    // 5. Slice 2
    MPSGraphTensor * slice2Tensor = [_graph sliceTensor:fc2Tensor
                                              dimension:1
                                                  start:inputChannels
                                                 length:inputChannels
                                                   name:[NSString stringWithFormat:@"%@/slice2", label]];
    
    // 5. Multiply and add.
    MPSGraphTensor * reshape1Tensor = [_graph reshapeTensor:gammaTensor
                                                  withShape:@[@(-1), gammaTensor.shape[1], @1, @1]
                                                       name:[NSString stringWithFormat:@"%@/reshape1", label]];
    
    
    MPSGraphTensor * multiplyTensor = [_graph multiplicationWithPrimaryTensor:parent
                                                              secondaryTensor:reshape1Tensor
                                                                         name:[NSString stringWithFormat:@"%@/multiply", label]];
    
    MPSGraphTensor * reshape2Tensor = [_graph reshapeTensor:slice2Tensor
                                                  withShape:@[@(-1), slice2Tensor.shape[1], @1, @1]
                                                       name:[NSString stringWithFormat:@"%@/reshape2", label]];
    
    
    // Two addition operations in series causes crashes on tensors with channels > 64.
    // Don't understand why. Might be due to failed attempts to fuse both add operations.
    // Using this multiply by 1 to interpose and prevent attempt to fuse.
//    float * ones = (float *)malloc(sizeof(float));
//    *ones = 1.0;
//    NSData * onesData = [NSData dataWithBytesNoCopy:ones length:sizeof(float)];
//
//    MPSGraphTensor * onesTensor = [_graph constantWithData:onesData
//                                                    shape:@[@1, @1, @1]
//                                                 dataType:MPSDataTypeFloat32];
//
//    MPSGraphTensor * dummyTensor = [_graph multiplicationWithPrimaryTensor:onesTensor
//                                                          secondaryTensor:skipTensor
//                                                     name:[NSString stringWithFormat:@"%@/dummy", label]];
    
    MPSGraphTensor * add1Tensor = [_graph additionWithPrimaryTensor:multiplyTensor
                                                    secondaryTensor:reshape2Tensor
                                                               name:[NSString stringWithFormat:@"%@/add1", label]];
    
    MPSGraphTensor * add2Tensor = [_graph additionWithPrimaryTensor:add1Tensor
                                                    secondaryTensor:skipTensor
                                                               name:[NSString stringWithFormat:@"%@/add2", label]];
    
    
    // 6. ReLU
    MPSGraphTensor * reluTensor = [_graph reLUWithTensor:add2Tensor
                                                    name:[NSString stringWithFormat:@"%@/relu", label]];

    //NSLog(@"Shapes: multiply %@, reshape2 %@, add1 %@, skip %@, add2 %@, relu %@, slice1 %@, slice2 %@", multiplyTensor.shape, reshape2Tensor.shape, add2Tensor.shape, skipTensor.shape, add2Tensor.shape, reluTensor.shape, slice1Tensor.shape, slice2Tensor.shape);
    
    // Add all the variables if specified.
    [self setVariable:[NSString stringWithFormat:@"%@/pool", label] tensor:poolTensor];
    [self setVariable:[NSString stringWithFormat:@"%@/fc1", label] tensor:fc1Tensor];
    [self setVariable:[NSString stringWithFormat:@"%@/fc2", label] tensor:fc2Tensor];
    [self setVariable:[NSString stringWithFormat:@"%@/slice1", label] tensor:slice1Tensor];
    [self setVariable:[NSString stringWithFormat:@"%@/slice2", label] tensor:slice2Tensor];
    [self setVariable:[NSString stringWithFormat:@"%@/sigmoid", label] tensor:gammaTensor];
    [self setVariable:[NSString stringWithFormat:@"%@/reshape1", label] tensor:reshape1Tensor];
    [self setVariable:[NSString stringWithFormat:@"%@/reshape2", label] tensor:reshape2Tensor];
    [self setVariable:[NSString stringWithFormat:@"%@/multiply", label] tensor:multiplyTensor];
    [self setVariable:[NSString stringWithFormat:@"%@/add1", label] tensor:add1Tensor];
    [self setVariable:[NSString stringWithFormat:@"%@/add2", label] tensor:add2Tensor];
    [self setVariable:[NSString stringWithFormat:@"%@/relu", label] tensor:reluTensor];
    
    [self setVariable:label tensor:reluTensor];

    return reluTensor;
}

-(nonnull MPSGraphTensor *) addPolicyMapLayerWithParent:(MPSGraphTensor * __nonnull)parent
                                              policyMap:(short * __nonnull)policyMap
                                        policyMapLength:(NSUInteger)policyMapLength
                                                  label:(NSString *)label
{
    // [1858 -> HWC or CHW]
    const bool HWC = false;
    // @todo: free this later.
    _reducedPolicyMap = (uint32_t *) malloc(kNumPolicyOutputs * sizeof(uint32_t));
    for (NSUInteger index = 0; index < policyMapLength; index++) {
        if (*(policyMap + index) == -1) continue;
//        const auto index = &mapping - kConvPolicyMap;
        const size_t displacement = index / 64;
        const size_t square = index % 64;
        const size_t row = square / 8;
        const size_t col = square % 8;
        if (HWC) {
            _reducedPolicyMap[*(policyMap + index)] = ((row * 8) + col) * 80 + displacement;
        } else {
            _reducedPolicyMap[*(policyMap + index)] = ((displacement * 8) + row) * 8 + col;
        }
    }

    NSData * policyMapData = [NSData dataWithBytesNoCopy:_reducedPolicyMap
                                                  length:kNumPolicyOutputs * sizeof(uint32_t)
                                            freeWhenDone:NO];
    
    MPSGraphTensor * mappingTensor = [_graph constantWithData:policyMapData
                                                        shape:@[@(kNumPolicyOutputs)]
                                                     dataType:MPSDataTypeUInt32];
    
    MPSGraphTensor * flatConvTensor = [_graph reshapeTensor:parent
                                                  withShape:@[@(-1), @([parent sizeOfDimensions:@[@1, @2, @3]])]
                                                       name:[NSString stringWithFormat:@"%@/flatten", label]];
    
    MPSGraphTensor * policyTensor = [_graph gatherWithUpdatesTensor:flatConvTensor
                                                      indicesTensor:mappingTensor
                                                               axis:1
                                                    batchDimensions:0
                                                               name:[NSString stringWithFormat:@"%@/gather", label]];

    [self setVariable:[NSString stringWithFormat:@"%@/constant", label] tensor:mappingTensor];
    [self setVariable:[NSString stringWithFormat:@"%@/flatten", label] tensor:flatConvTensor];
    [self setVariable:[NSString stringWithFormat:@"%@/gather", label] tensor:policyTensor];
    [self setVariable:label tensor:policyTensor];
    
    return policyTensor;
}


-(void) setVariable:(NSString * __nonnull)name
             tensor:(MPSGraphTensor *)tensor
{
    if (![[_readVariables allKeys] containsObject:name]) return;
    
    _readVariables[name] = tensor;
}

-(void) addVariable:(NSString * __nonnull)name
{
    _readVariables[name] = [NSNull null];
}

-(void) dumpVariable:(NSString * __nonnull)name
             batches:(NSUInteger)batches
{
    if (!_readVariables[name] || _readVariables[name] == [NSNull null]) {
        NSLog(@"No variable '%@' found", name);
        return;
    }

    NSUInteger size = batches * [_readVariables[name] sizeOfDimensions:@[@1, @2, @3]];
    float * dumpArray = (float *)malloc(size * sizeof(float));
    [[_resultDataDictionary[_readVariables[name]] mpsndarray] readBytes:dumpArray strideBytes:nil];
    NSLog(@"Dumping: '%@', size: %i", name, size);
    for (NSUInteger i = 0; i < (size > 100 ? 100 : size); i++) {
        NSLog(@";%i;%f", i, dumpArray[i]);
    }
}

@end

