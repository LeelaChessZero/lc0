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

// Maximum number of metal command buffers that can run simultaneously.
static const NSUInteger kMaxInflightBuffers = 2;

// Minimum batch size below which parallel command buffers will not be used.
static const NSInteger kMinSubBatchSize = 20;

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

// This is the Lc0NetworkGraph dictionary getter method.
// It is a singleton object that is used to store the Lc0NetworkGraph.
+(NSMutableDictionary * _Nonnull) getGraphs {
    // This is the Lc0NetworkGraph dictionary.
    static NSMutableDictionary * graphs = nil;

    @synchronized (self) {
        if (graphs == nil) {
            graphs = [NSMutableDictionary dictionaryWithCapacity:1];
        }
    }

    return graphs;
}

// This is the Lc0NetworkGraph getter method.
+(Lc0NetworkGraph * _Nonnull) getGraphAt:(NSNumber * _Nonnull)index {
  NSMutableDictionary * graphs = [Lc0NetworkGraph getGraphs];

  return graphs[index];
}

// This is the Lc0NetworkGraph factory method.
// It is used to create a Lc0NetworkGraph object.
// The Lc0NetworkGraph object is stored in the dictionary.
// The Lc0NetworkGraph object is initialized with the Metal device.
+(void) graphWithDevice:(id<MTLDevice> __nonnull)device
                index:(NSNumber * _Nonnull)index {
    NSMutableDictionary * graphs = [Lc0NetworkGraph getGraphs];

    @synchronized (self) {
        if (graphs[index] == nil) {
            graphs[index] = [[Lc0NetworkGraph alloc] initWithDevice:device];
        }
    }
}

-(nonnull instancetype) initWithDevice:(id<MTLDevice> __nonnull)device
{
    self = [super init];
    _device = [MPSGraphDevice deviceWithMTLDevice:device];
    _queue = [device newCommandQueue];
    _graph = [[MPSGraph alloc] init];
    _resultTensors = @[];
    _readVariables = [[NSMutableDictionary alloc] init];
    _doubleBufferingSemaphore = dispatch_semaphore_create(kMaxInflightBuffers);
    _resultDataDicts = [NSMutableDictionary dictionaryWithCapacity:kMaxInflightBuffers];

    return self;
}

-(nonnull NSArray<MPSGraphTensor *> *) runInferenceWithBatchSize:(NSUInteger)batchSize
                                                          inputs:(float * __nonnull)inputs
                                                         outputs:(float * __nonnull * __nonnull)outputBuffers
{
    // Calculate number of sub-batches to split across GPU command buffers for parallel execution.
    // Shouldn't be more than kMaxInflightBuffers and each sub-batch shouldn't be smaller than kMinSubBatchSize.
    NSUInteger splits = (batchSize + kMinSubBatchSize + 1) / kMinSubBatchSize;
    if (splits > kMaxInflightBuffers) splits = kMaxInflightBuffers;
    NSUInteger subBatchSize = batchSize / splits;
    NSUInteger inputDataLength = subBatchSize * [_inputTensor sizeOfDimensions:@[@1, @2, @3]];


    // Split batchSize into smaller sub-batches and run using double-buffering.
    NSUInteger subBatch = 0;
    MPSCommandBuffer * commandBuffer;
    for (subBatch = 0; subBatch < splits - 1; subBatch++) {
        commandBuffer = [self runCommandSubBatchWithInputs:inputs + subBatch * inputDataLength
                                  subBatch:subBatch
                              subBatchSize:subBatchSize];
    }
    // Last sub-batch may be smaller or larger than others.
    MPSCommandBuffer * latestCommandBuffer = [self runCommandSubBatchWithInputs:inputs + subBatch * inputDataLength
                                                                       subBatch:subBatch
                                                                   subBatchSize:batchSize - subBatch * subBatchSize];

    // Wait for the last batch to be processed.
    [latestCommandBuffer waitUntilCompleted];
    [commandBuffer waitUntilCompleted];

    [self copyResultsToBuffers:outputBuffers subBatchSize:subBatchSize];

    return _resultTensors;
}

-(nonnull MPSCommandBuffer *) runCommandSubBatchWithInputs:(float * __nonnull)inputs
                                                  subBatch:(NSUInteger)subBatch
                                              subBatchSize:(NSUInteger)subBatchSize
{
    // Double buffering semaphore to correctly double buffer iterations.
    dispatch_semaphore_wait(_doubleBufferingSemaphore, DISPATCH_TIME_FOREVER);

    // Create command buffer for this sub-batch.
    MPSCommandBuffer * commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:_queue];

    MPSShape * shape = @[@(subBatchSize), _inputTensor.shape[1], _inputTensor.shape[2], _inputTensor.shape[3]];

    NSData * inputData = [NSData dataWithBytesNoCopy:inputs
                                              length:subBatchSize * sizeof(float)
                                        freeWhenDone:NO];

    MPSGraphTensorData * inputTensorData = [[MPSGraphTensorData alloc] initWithDevice:_device
                                                                                 data:inputData
                                                                                shape:shape
                                                                             dataType:_inputTensor.dataType];

    // Create execution descriptor with block to update results for each iteration.
    MPSGraphExecutionDescriptor * executionDescriptor = [[MPSGraphExecutionDescriptor alloc] init];
    executionDescriptor.completionHandler = ^(MPSGraphTensorDataDictionary * resultDictionary, NSError * error) {
        _resultDataDicts[@(subBatch)] = resultDictionary;

        // Release double buffering semaphore for the next training iteration to be encoded.
        dispatch_semaphore_signal(_doubleBufferingSemaphore);
    };

    [_graph encodeToCommandBuffer:commandBuffer
                            feeds:@{_inputTensor : inputTensorData}
                    targetTensors:_targetTensors
                 targetOperations:nil
              executionDescriptor:executionDescriptor];

    // Commit the command buffer
    [commandBuffer commit];
    return commandBuffer;
}


-(void) copyResultsToBuffers:(float * __nonnull * __nonnull)outputBuffers
                subBatchSize:(NSUInteger)subBatchSize
{
    // Copy results for batch back into the output buffers.
    for (NSUInteger rsIdx = 0; rsIdx < [_resultTensors count]; rsIdx++) {
        NSUInteger outputDataLength = [_resultTensors[rsIdx] sizeOfDimensions:@[@1, @2, @3]] * subBatchSize;
        for (NSUInteger subBatch = 0; subBatch < [_resultDataDicts count]; subBatch++) {
            [[_resultDataDicts[@(subBatch)][_resultTensors[rsIdx]] mpsndarray] readBytes:outputBuffers[rsIdx] + subBatch * outputDataLength
                                                                     strideBytes:nil];
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
        return seUnit;
    }
    else {
        MPSGraphTensor * residualTensor = [_graph additionWithPrimaryTensor:parent
                                                            secondaryTensor:conv2Tensor
                                                                       name:[NSString stringWithFormat:@"%@/add", label]];

        MPSGraphTensor * reluTensor = [_graph reLUWithTensor:residualTensor
                                                        name:[NSString stringWithFormat:@"%@/relu", label]];
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

    MPSGraphTensor * add1Tensor = [_graph additionWithPrimaryTensor:multiplyTensor
                                                    secondaryTensor:reshape2Tensor
                                                               name:[NSString stringWithFormat:@"%@/add1", label]];

    MPSGraphTensor * add2Tensor = [_graph additionWithPrimaryTensor:add1Tensor
                                                    secondaryTensor:skipTensor
                                                               name:[NSString stringWithFormat:@"%@/add2", label]];

    // 6. ReLU
    MPSGraphTensor * reluTensor = [_graph reLUWithTensor:add2Tensor
                                                    name:[NSString stringWithFormat:@"%@/relu", label]];

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
                                              policyMap:(uint32_t * __nonnull)policyMap
                                                  label:(NSString *)label
{
    NSData * policyMapData = [NSData dataWithBytesNoCopy:policyMap
                                                  length:kNumPolicyOutputs * sizeof(uint32_t)
                                            freeWhenDone:NO];

    MPSGraphTensor * mappingTensor = [_graph constantWithData:policyMapData
                                                        shape:@[@(kNumPolicyOutputs)]
                                                     dataType:MPSDataTypeUInt32];

    MPSGraphTensor * flatConvTensor = [_graph reshapeTensor:parent
                                                  withShape:@[parent.shape[0], @([parent sizeOfDimensions:@[@1, @2, @3]])]
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

-(void) trackVariable:(NSString * __nonnull)name
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

    MPSGraphTensor * variable = (MPSGraphTensor *) _readVariables[name];
    NSUInteger size = [variable.shape[0] intValue] > 0 ? [variable size] : batches * [variable sizeOfDimensions:@[@1, @2, @3]];

    if (variable.dataType == MPSDataTypeUInt32) {
        uint32_t * dumpArray = (uint32_t *)malloc(size * sizeof(uint32_t));
        [[_resultDataDicts[@0][_readVariables[name]] mpsndarray] readBytes:dumpArray strideBytes:nil];
        NSLog(@"Dumping: '%@', size: %i, type: %i", name, size, variable.dataType);
        for (NSUInteger i = 0; i < (size > 100 ? 100 : size); i++) {
            NSLog(@";%i;%i", i, dumpArray[i]);
        }
    } else {
        float * dumpArray = (float *)malloc(size * sizeof(float));
        [[_resultDataDicts[@0][_readVariables[name]] mpsndarray] readBytes:dumpArray strideBytes:nil];
        NSLog(@"Dumping: '%@', size: %i, type: %i", name, size, variable.dataType);
        for (NSUInteger i = 0; i < (size > 100 ? 100 : size); i++) {
            NSLog(@";%i;%f", i, dumpArray[i]);
        }
    }
}

@end
