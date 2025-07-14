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

#import "neural/network_legacy.h"
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


-(NSUInteger) sizeOfDimensionsFrom:(NSNumber *)dimension {
    NSUInteger size = 1;
    for (NSUInteger dim = [dimension intValue]; dim < [self.shape count]; dim++) {
        size *= [self.shape[dim] intValue];
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

    [self encodeToCommandBuffer:commandBuffer
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
    _inputTensor = [self placeholderWithShape:@[@(-1), @(channels), @(height), @(width)] name:label];

    return _inputTensor;
}

-(nonnull MPSGraphTensor *) addConvolutionBlockWithParent:(MPSGraphTensor * __nonnull)parent
                                           outputChannels:(NSUInteger)outputChannels
                                               kernelSize:(NSUInteger)kernelSize
                                                  weights:(float * __nonnull)weights
                                                   biases:(float * __nonnull)biases
                                               activation:(NSString * __nullable)activation
                                                    label:(NSString * __nonnull)label
{
    NSUInteger inputChannels = [parent.shape[1] intValue];

    NSData * weightsData = [NSData dataWithBytesNoCopy:weights
                                                length:outputChannels * inputChannels * kernelSize * kernelSize * sizeof(float)
                                          freeWhenDone:NO];

    MPSGraphTensor * weightsTensor = [self variableWithData:weightsData
                                                      shape:@[@(outputChannels), @(inputChannels), @(kernelSize), @(kernelSize)]
                                                   dataType:MPSDataTypeFloat32
                                                       name:[NSString stringWithFormat:@"%@/weights", label]];

    NSData * biasData = [NSData dataWithBytesNoCopy:biases
                                             length:outputChannels * sizeof(float)
                                       freeWhenDone:NO];

    MPSGraphTensor * biasTensor = [self variableWithData:biasData
                                                   shape:@[@(outputChannels), @1, @1]
                                                dataType:MPSDataTypeFloat32
                                                    name:[NSString stringWithFormat:@"%@/biases", label]];

    MPSGraphTensor * convTensor = [self convolution2DWithSourceTensor:parent
                                                        weightsTensor:weightsTensor
                                                           descriptor:convolution2DDescriptor
                                                                 name:[NSString stringWithFormat:@"%@/conv", label]];

    MPSGraphTensor * convBiasTensor = [self additionWithPrimaryTensor:convTensor
                                                      secondaryTensor:biasTensor
                                                                 name:[NSString stringWithFormat:@"%@/bias_add", label]];

    return [self applyActivationWithTensor:convBiasTensor activation:activation label:label];
}

-(nonnull MPSGraphTensor *) addResidualBlockWithParent:(MPSGraphTensor * __nonnull)parent
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
                                            activation:(NSString * __nullable)activation
{
    MPSGraphTensor * conv1Tensor = [self addConvolutionBlockWithParent:parent
                                                        outputChannels:outputChannels
                                                            kernelSize:kernelSize
                                                               weights:weights1
                                                                biases:biases1
                                                            activation:activation
                                                                 label:[NSString stringWithFormat:@"%@/conv1", label]];

    MPSGraphTensor * conv2Tensor = [self addConvolutionBlockWithParent:conv1Tensor
                                                        outputChannels:outputChannels
                                                            kernelSize:kernelSize
                                                               weights:weights2
                                                                biases:biases2
                                                            activation:nil
                                                                 label:[NSString stringWithFormat:@"%@/conv2", label]];

    if (hasSe) {
        // SE Unit.
        return [self addSEUnitWithParent:conv2Tensor
                                skipNode:parent
                          outputChannels:outputChannels
                             seFcOutputs:seFcOutputs
                                weights1:seWeights1
                                 biases1:seBiases1
                                weights2:seWeights2
                                 biases2:seBiases2
                              activation:activation
                                   label:[NSString stringWithFormat:@"%@/se", label]];
    }
    else {
        MPSGraphTensor * residualTensor = [self additionWithPrimaryTensor:parent
                                                          secondaryTensor:conv2Tensor
                                                                     name:[NSString stringWithFormat:@"%@/add", label]];

        return [self applyActivationWithTensor:residualTensor
                                    activation:activation
                                         label:label];
    }
}

-(nonnull MPSGraphTensor *) addFullyConnectedLayerWithParent:(MPSGraphTensor * __nonnull)parent
                                              outputChannels:(NSUInteger)outputChannels
                                                     weights:(float * __nonnull)weights
                                                      biases:(float * __nullable)biases
                                                  activation:(NSString * __nullable)activation
                                                       label:(NSString * __nonnull)label
{
    NSUInteger inputChannels = [[parent.shape lastObject] intValue];

    NSData * weightData = [NSData dataWithBytesNoCopy:weights
                                               length:outputChannels * inputChannels * sizeof(float)
                                         freeWhenDone:NO];

    MPSGraphTensor * weightTensor = [self variableWithData:weightData
                                                     shape:@[@(outputChannels), @(inputChannels)]
                                                  dataType:MPSDataTypeFloat32
                                                      name:[NSString stringWithFormat:@"%@/weights", label]];

    // Leela weights are OIHW, need to be transposed to IO** to allow matmul.
    weightTensor = [self transposeTensor:weightTensor
                               dimension:0
                           withDimension:1
                                    name:[NSString stringWithFormat:@"%@/weights_transpose", label]];

    parent = [self matrixMultiplicationWithPrimaryTensor:parent
                                         secondaryTensor:weightTensor
                                                    name:[NSString stringWithFormat:@"%@/matmul", label]];

    if (biases != nil) {
        NSData * biasData = [NSData dataWithBytesNoCopy:biases
                                                 length:outputChannels * sizeof(float)
                                           freeWhenDone:NO];

        MPSGraphTensor * biasTensor = [self variableWithData:biasData
                                                       shape:@[@(outputChannels)]
                                                    dataType:MPSDataTypeFloat32
                                                        name:[NSString stringWithFormat:@"%@/biases", label]];

        parent = [self additionWithPrimaryTensor:parent
                                 secondaryTensor:biasTensor
                                            name:[NSString stringWithFormat:@"%@/bias_add", label]];
    }
    return [self applyActivationWithTensor:parent activation:activation label:label];
}

-(nonnull MPSGraphTensor *) addSEUnitWithParent:(MPSGraphTensor * __nonnull)parent
                                       skipNode:(MPSGraphTensor * __nonnull)skipTensor
                                 outputChannels:(NSUInteger)outputChannels
                                    seFcOutputs:(NSUInteger)seFcOutputs
                                       weights1:(float * __nonnull)weights1
                                        biases1:(float * __nonnull)biases1
                                       weights2:(float * __nonnull)weights2
                                        biases2:(float * __nonnull)biases2
                                     activation:(NSString * __nullable) activation
                                          label:(NSString * __nonnull)label
{

    // 1. Global Average Pooling 2D
    MPSGraphTensor * seunit = [self avgPooling2DWithSourceTensor:parent
                                                      descriptor:averagePoolingDescriptor
                                                            name:[NSString stringWithFormat:@"%@/pool", label]];

    // 2. FC Layer 1.
    seunit = [self flatten2DTensor:seunit
                              axis:1
                              name:[NSString stringWithFormat:@"%@/flatten", label]];

    seunit = [self addFullyConnectedLayerWithParent:seunit
                                     outputChannels:seFcOutputs
                                            weights:weights1
                                             biases:biases1
                                         activation:activation
                                              label:[NSString stringWithFormat:@"%@/fc1", label]];

    // 3. FC Layer 2.
    NSUInteger inputChannels = [parent.shape[1] intValue];
    seunit = [self addFullyConnectedLayerWithParent:seunit
                                     outputChannels:2 * inputChannels
                                            weights:weights2
                                             biases:biases2
                                         activation:nil
                                              label:[NSString stringWithFormat:@"%@/fc2", label]];

    // 4. Slice 1, gamma and multiply.
    MPSGraphTensor * gamma = [self sliceTensor:seunit
                                     dimension:1
                                         start:0
                                        length:inputChannels
                                          name:[NSString stringWithFormat:@"%@/slice1", label]];

    gamma = [self sigmoidWithTensor:gamma
                               name:[NSString stringWithFormat:@"%@/sigmoid", label]];

    gamma = [self reshapeTensor:gamma
                      withShape:@[@(-1), gamma.shape[1], @1, @1]
                           name:[NSString stringWithFormat:@"%@/reshape1", label]];

    gamma = [self multiplicationWithPrimaryTensor:parent
                                  secondaryTensor:gamma
                                             name:[NSString stringWithFormat:@"%@/multiply", label]];

    // 5. Slice 2 and add.
    seunit = [self sliceTensor:seunit
                     dimension:1
                         start:inputChannels
                        length:inputChannels
                          name:[NSString stringWithFormat:@"%@/slice2", label]];

    seunit = [self reshapeTensor:seunit
                       withShape:@[@(-1), seunit.shape[1], @1, @1]
                            name:[NSString stringWithFormat:@"%@/reshape2", label]];

    seunit = [self additionWithPrimaryTensor:gamma
                             secondaryTensor:seunit
                                        name:[NSString stringWithFormat:@"%@/add1", label]];

    seunit = [self additionWithPrimaryTensor:seunit
                             secondaryTensor:skipTensor
                                        name:[NSString stringWithFormat:@"%@/add2", label]];

    // 6. Default activation.
    return [self applyActivationWithTensor:seunit
                                activation:activation
                                     label:label];
}

-(nonnull MPSGraphTensor *) addPolicyMapLayerWithParent:(MPSGraphTensor * __nonnull)parent
                                              policyMap:(uint32_t * __nonnull)policyMap
                                                  label:(NSString * __nonnull)label
{
    NSData * policyMapData = [NSData dataWithBytesNoCopy:policyMap
                                                  length:kNumPolicyOutputs * sizeof(uint32_t)
                                            freeWhenDone:NO];

    MPSGraphTensor * mappingTensor = [self constantWithData:policyMapData
                                                      shape:@[@(kNumPolicyOutputs)]
                                                   dataType:MPSDataTypeUInt32];

    MPSGraphTensor * flatConvTensor = [self flatten2DTensor:parent
                                                       axis:1
                                                       name:[NSString stringWithFormat:@"%@/flatten", label]];

    MPSGraphTensor * policyTensor = [self gatherWithUpdatesTensor:flatConvTensor
                                                    indicesTensor:mappingTensor
                                                             axis:1
                                                  batchDimensions:0
                                                             name:[NSString stringWithFormat:@"%@/gather", label]];

    return policyTensor;
}

-(nonnull MPSGraphTensor *) addEncoderLayerWithParent:(MPSGraphTensor * __nonnull)parent
                                        legacyWeights:(lczero::MultiHeadWeights::EncoderLayer &)encoder
                                                heads:(NSUInteger)heads
                                        embeddingSize:(NSUInteger)embeddingSize
                                    smolgenActivation:(NSString * __nullable)smolgenActivation
                                        ffnActivation:(NSString * __nonnull)ffnActivation
                                                alpha:(float)alpha
                                              epsilon:(float)epsilon
                                             normtype:(NSString * __nonnull)normtype
                                                label:(NSString * __nonnull)label
{
    NSUInteger dModel = encoder.mha.q_b.size();
    MPSGraphTensor * mhaQ = [self addFullyConnectedLayerWithParent:parent
                                                    outputChannels:encoder.mha.q_b.size()
                                                           weights:&encoder.mha.q_w[0]
                                                            biases:&encoder.mha.q_b[0]
                                                        activation:nil
                                                             label:[NSString stringWithFormat:@"%@/mhaq/fc", label]];

    MPSGraphTensor * mhaK = [self addFullyConnectedLayerWithParent:parent
                                                    outputChannels:encoder.mha.k_b.size()
                                                           weights:&encoder.mha.k_w[0]
                                                            biases:&encoder.mha.k_b[0]
                                                        activation:nil
                                                             label:[NSString stringWithFormat:@"%@/mhak/fc", label]];

    MPSGraphTensor * mhaV = [self addFullyConnectedLayerWithParent:parent
                                                    outputChannels:encoder.mha.v_b.size()
                                                           weights:&encoder.mha.v_w[0]
                                                            biases:&encoder.mha.v_b[0]
                                                        activation:nil
                                                             label:[NSString stringWithFormat:@"%@/mhav/fc", label]];

    MPSGraphTensor * mha = [self scaledMHAMatmulWithQueries:mhaQ
                                                   withKeys:mhaK
                                                 withValues:mhaV
                                                      heads:heads
                                                     parent:parent
                                                    smolgen:encoder.mha.has_smolgen ? &encoder.mha.smolgen : nil
                                          smolgenActivation:smolgenActivation
                                                      label:[NSString stringWithFormat:@"%@/mha", label]];

    // MHA final dense layer.
    mha = [self addFullyConnectedLayerWithParent:mha
                                  outputChannels:embeddingSize
                                         weights:&encoder.mha.dense_w[0]
                                          biases:&encoder.mha.dense_b[0]
                                      activation:nil
                                           label:[NSString stringWithFormat:@"%@/mha/fc", label]];

    // Skip connection + Layer Norm 1.
    MPSGraphTensor * enc;
    if ([normtype isEqual:@"layernorm"]) {
        enc = [self addLayerNormalizationWithParent:parent
                              scaledSecondaryTensor:mha
                                             gammas:&encoder.ln1_gammas[0]
                                              betas:&encoder.ln1_betas[0]
                                              alpha:alpha
                                            epsilon:epsilon
                                              label:[NSString stringWithFormat:@"%@/ln1", label]];
    }
    else if ([normtype isEqual:@"rmsnorm"]) {
        enc = [self addRmsNormalizationWithParent:parent
                            scaledSecondaryTensor:mha
                                           gammas:&encoder.ln1_gammas[0]
                                            alpha:alpha
                                            label:[NSString stringWithFormat:@"%@/ln1", label]];
    }
    else if ([normtype isEqual:@"skipfirst"]) {
        if (alpha != 1.0) {
            enc = [self constantWithScalar:alpha shape:@[@1] dataType:parent.dataType];
            enc = [self multiplicationWithPrimaryTensor:mha
                                        secondaryTensor:enc
                                                   name:[NSString stringWithFormat:@"%@/multiply", label]];
        }
        enc = [self additionWithPrimaryTensor:parent
                              secondaryTensor:enc
                                         name:[NSString stringWithFormat:@"%@/add", label]];
    }
    else {
        [NSException raise:@"Invalid normalization type."
                    format:@"Invalid normalization type specified: %@", normtype];
    }

    // Feedforward network (FFN).
    MPSGraphTensor * ffn = [self addFullyConnectedLayerWithParent:enc
                                                   outputChannels:encoder.ffn.dense1_b.size()
                                                          weights:&encoder.ffn.dense1_w[0]
                                                           biases:&encoder.ffn.dense1_b[0]
                                                       activation:ffnActivation
                                                            label:[NSString stringWithFormat:@"%@/ffn1", label]];

    ffn = [self addFullyConnectedLayerWithParent:ffn
                                  outputChannels:encoder.ffn.dense2_b.size()
                                         weights:&encoder.ffn.dense2_w[0]
                                          biases:&encoder.ffn.dense2_b[0]
                                      activation:nil
                                           label:[NSString stringWithFormat:@"%@/ffn2", label]];

    // Skip connection + Layer Norm 2.
    if ([normtype isEqual:@"layernorm"]) {
        return [self addLayerNormalizationWithParent:enc
                               scaledSecondaryTensor:ffn
                                              gammas:&encoder.ln2_gammas[0]
                                               betas:&encoder.ln2_betas[0]
                                               alpha:alpha
                                             epsilon:epsilon
                                               label:[NSString stringWithFormat:@"%@/ln2", label]];
    }
    else if ([normtype isEqual:@"rmsnorm"] || [normtype isEqual:@"skipfirst"]) {
        enc = [self addRmsNormalizationWithParent:enc
                            scaledSecondaryTensor:ffn
                                           gammas:&encoder.ln2_gammas[0]
                                            alpha:alpha
                                            label:[NSString stringWithFormat:@"%@/ln1", label]];
    }
    else {
        [NSException raise:@"Invalid normalization type."
                    format:@"Invalid normalization type specified: %@", normtype];
    }
}

-(nonnull MPSGraphTensor *) addLayerNormalizationWithParent:(MPSGraphTensor * __nonnull)parent
                                      scaledSecondaryTensor:(MPSGraphTensor * __nullable)secondary
                                                     gammas:(float * __nonnull)gammas
                                                      betas:(float * __nonnull)betas
                                                      alpha:(float)alpha
                                                    epsilon:(float)epsilon
                                                      label:(NSString * __nonnull)label
{
    if (secondary != nil) {
        if (alpha != 1.0) {
            MPSGraphTensor * alphaTensor = [self constantWithScalar:alpha shape:@[@1] dataType:parent.dataType];
            secondary = [self multiplicationWithPrimaryTensor:secondary
                                              secondaryTensor:alphaTensor
                                                         name:[NSString stringWithFormat:@"%@/multiply", label]];
        }

        parent = [self additionWithPrimaryTensor:parent
                                 secondaryTensor:secondary
                                            name:[NSString stringWithFormat:@"%@/add", label]];
    }

    NSUInteger axis = [parent.shape count] - 1;
    NSUInteger channelSize = [[parent.shape lastObject] intValue];

    MPSGraphTensor * means = [self meanOfTensor:parent
                                           axes:@[@(axis)]
                                           name:[NSString stringWithFormat:@"%@/mean", label]];

    MPSGraphTensor * variances = [self varianceOfTensor:parent
                                                   axes:@[@(axis)]
                                                   name:[NSString stringWithFormat:@"%@/variance", label]];

    NSData * gammaData = [NSData dataWithBytesNoCopy:gammas
                                              length:channelSize * sizeof(float)
                                        freeWhenDone:NO];

    MPSGraphTensor * gammaTensor = [self variableWithData:gammaData
                                                    shape:@[@(channelSize)]
                                                 dataType:MPSDataTypeFloat32
                                                     name:[NSString stringWithFormat:@"%@/gamma", label]];

    NSData * betaData = [NSData dataWithBytesNoCopy:betas
                                             length:channelSize * sizeof(float)
                                       freeWhenDone:NO];

    MPSGraphTensor * betaTensor = [self variableWithData:betaData
                                                   shape:@[@(channelSize)]
                                                dataType:MPSDataTypeFloat32
                                                    name:[NSString stringWithFormat:@"%@/beta", label]];

    return [self normalizationWithTensor:parent
                              meanTensor:means
                          varianceTensor:variances
                             gammaTensor:gammaTensor
                              betaTensor:betaTensor
                                 epsilon:epsilon
                                    name:[NSString stringWithFormat:@"%@/norm", label]];
}


-(nonnull MPSGraphTensor *) addRmsNormalizationWithParent:(MPSGraphTensor * __nonnull)parent
                                    scaledSecondaryTensor:(MPSGraphTensor * __nullable)secondary
                                                   gammas:(float * __nonnull)gammas
                                                    alpha:(float)alpha
                                                    label:(NSString * __nonnull)label
{
    if (secondary != nil) {
        if (alpha != 1.0) {
            MPSGraphTensor * alphaTensor = [self constantWithScalar:alpha shape:@[@1] dataType:parent.dataType];
            secondary = [self multiplicationWithPrimaryTensor:secondary
                                              secondaryTensor:alphaTensor
                                                         name:[NSString stringWithFormat:@"%@/multiply", label]];
        }

        parent = [self additionWithPrimaryTensor:parent
                                 secondaryTensor:secondary
                                            name:[NSString stringWithFormat:@"%@/add", label]];
    }

    NSUInteger axis = [parent.shape count] - 1;
    NSUInteger channelSize = [[parent.shape lastObject] intValue];

    MPSGraphTensor * factor = [self multiplicationWithPrimaryTensor:parent
                                                    secondaryTensor:parent
                                                               name:[NSString stringWithFormat:@"%@/square", label]];

    factor = [self meanOfTensor:factor
                           axes:@[@(axis)]
                           name:[NSString stringWithFormat:@"%@/mean", label]];

    factor = [self squareRootWithTensor:factor
                                   name:[NSString stringWithFormat:@"%@/sqrt", label]];

    NSData * gammaData = [NSData dataWithBytesNoCopy:gammas
                                              length:channelSize * sizeof(float)
                                        freeWhenDone:NO];

    MPSGraphTensor * gammaTensor = [self variableWithData:gammaData
                                                    shape:@[@(channelSize)]
                                                 dataType:MPSDataTypeFloat32
                                                     name:[NSString stringWithFormat:@"%@/gamma", label]];

    factor = [self multiplicationWithPrimaryTensor:factor
                                   secondaryTensor:gammaTensor
                                              name:[NSString stringWithFormat:@"%@/multiply2", label]];

    return [self multiplicationWithPrimaryTensor:parent
                                 secondaryTensor:factor
                                            name:[NSString stringWithFormat:@"%@/multiply3", label]];
}

-(nonnull MPSGraphTensor *) transposeChannelsWithTensor:(MPSGraphTensor * __nonnull)tensor
                                              withShape:(MPSShape * __nonnull)withShape
                                                  label:(NSString * __nonnull)label
{
    MPSGraphTensor * transposeTensor = [self transposeTensor:tensor
                                                   dimension:1
                                               withDimension:2
                                                        name:[NSString stringWithFormat:@"%@/weights_transpose_1", label]];
    transposeTensor = [self transposeTensor:transposeTensor
                                  dimension:2
                              withDimension:3
                                       name:[NSString stringWithFormat:@"%@/weights_transpose_2", label]];

    return [self reshapeTensor:transposeTensor
                     withShape:withShape
                          name:[NSString stringWithFormat:@"%@/reshape", label]];
}

-(nonnull MPSGraphTensor *) scaledMHAMatmulWithQueries:(MPSGraphTensor * __nonnull)queries
                                              withKeys:(MPSGraphTensor * __nonnull)keys
                                            withValues:(MPSGraphTensor * __nonnull)values
                                                 heads:(NSUInteger)heads
                                                parent:(MPSGraphTensor * __nonnull)parent
                                               smolgen:(lczero::MultiHeadWeights::Smolgen * __nullable)smolgen
                                     smolgenActivation:(NSString * __nullable)smolgenActivation
                                                 label:(NSString * __nonnull)label
{
    // Split heads.
    const NSUInteger dmodel = [[queries.shape lastObject] intValue];
    const NSUInteger depth = dmodel / heads;

    queries = [self reshapeTensor:queries withShape:@[@(-1), @64, @(heads), @(depth)] name:[NSString stringWithFormat:@"%@/reshape_q", label]];
    queries = [self transposeTensor:queries dimension:1 withDimension:2 name:[NSString stringWithFormat:@"%@/transpose_q", label]];

    keys = [self reshapeTensor:keys withShape:@[@(-1), @64, @(heads), @(depth)] name:[NSString stringWithFormat:@"%@/reshape_k", label]];
    keys = [self transposeTensor:keys dimension:1 withDimension:2 name:[NSString stringWithFormat:@"%@/transpose_k", label]];

    values = [self reshapeTensor:values withShape:@[@(-1), @64, @(heads), @(depth)] name:[NSString stringWithFormat:@"%@/reshape_v", label]];
    values = [self transposeTensor:values dimension:1 withDimension:2 name:[NSString stringWithFormat:@"%@/transpose_v", label]];

    // Scaled attention matmul.
    keys = [self transposeTensor:keys dimension:2 withDimension:3 name:[NSString stringWithFormat:@"%@/transpose_k_2", label]];
    MPSGraphTensor * attn = [self matrixMultiplicationWithPrimaryTensor:queries
                                                        secondaryTensor:keys
                                                                   name:[NSString stringWithFormat:@"%@/matmul_qk", label]];
    attn = [self divisionWithPrimaryTensor:attn
                           secondaryTensor:[self constantWithScalar:sqrt(depth)
                                                              shape:@[@1]
                                                           dataType:attn.dataType]
                                      name:[NSString stringWithFormat:@"%@/scale", label]];
    // Smolgen.
    if (smolgen != nil) {
        // Smolgen weights.
        // 1. Compressed fully connected layer and reshape.
        NSUInteger hidden_channels = smolgen->compress.size() / [[parent.shape lastObject] intValue];
        MPSGraphTensor * smolgenWeights = [self addFullyConnectedLayerWithParent:parent
                                                                  outputChannels:hidden_channels
                                                                         weights:&smolgen->compress[0]
                                                                          biases:nil
                                                                      activation:nil
                                                                           label:[NSString stringWithFormat:@"%@/smolgen/compress", label]];
        smolgenWeights = [self flatten2DTensor:smolgenWeights
                                          axis:1
                                          name:[NSString stringWithFormat:@"%@/smolgen/flatten", label]];

        // 2. Dense 1 with layer norm.
        smolgenWeights = [self addFullyConnectedLayerWithParent:smolgenWeights
                                                 outputChannels:smolgen->dense1_b.size()
                                                        weights:&smolgen->dense1_w[0]
                                                         biases:&smolgen->dense1_b[0]
                                                     activation:smolgenActivation
                                                          label:[NSString stringWithFormat:@"%@/smolgen/dense_1", label]];

        smolgenWeights = [self addLayerNormalizationWithParent:smolgenWeights
                                         scaledSecondaryTensor:nil
                                                        gammas:&smolgen->ln1_gammas[0]
                                                         betas:&smolgen->ln1_betas[0]
                                                         alpha:0.0
                                                       epsilon:1e-3
                                                         label:[NSString stringWithFormat:@"%@/smolgen/ln1", label]];

        // 3. Dense 2 with layer norm.
        smolgenWeights = [self addFullyConnectedLayerWithParent:smolgenWeights
                                                 outputChannels:smolgen->dense2_b.size()
                                                        weights:&smolgen->dense2_w[0]
                                                         biases:&smolgen->dense2_b[0]
                                                     activation:smolgenActivation
                                                          label:[NSString stringWithFormat:@"%@/smolgen/dense_2", label]];

        smolgenWeights = [self addLayerNormalizationWithParent:smolgenWeights
                                         scaledSecondaryTensor:nil
                                                        gammas:&smolgen->ln2_gammas[0]
                                                         betas:&smolgen->ln2_betas[0]
                                                         alpha:0.0
                                                       epsilon:1e-3
                                                         label:[NSString stringWithFormat:@"%@/smolgen/ln2", label]];

        smolgenWeights = [self reshapeTensor:smolgenWeights
                                   withShape:@[@(-1), @(heads), @(smolgen->dense2_b.size() / heads)]
                                        name:[NSString stringWithFormat:@"%@/smolgen/reshape_1", label]];

        // 4. Global smolgen weights
        smolgenWeights = [self addFullyConnectedLayerWithParent:smolgenWeights
                                                 outputChannels:64 * 64
                                                        weights:_globalSmolgenWeights
                                                         biases:nil
                                                     activation:nil
                                                          label:[NSString stringWithFormat:@"%@/smolgen/global", label]];

        smolgenWeights = [self reshapeTensor:smolgenWeights
                                   withShape:@[@(-1), @(heads), @64, @64]
                                        name:[NSString stringWithFormat:@"%@/smolgen/reshape_2", label]];

        attn = [self additionWithPrimaryTensor:attn
                               secondaryTensor:smolgenWeights
                                          name:[NSString stringWithFormat:@"%@/smolgen_add", label]];
    }

    attn = [self applyActivationWithTensor:attn activation:@"softmax" label:label];

    // matmul(scaled_attention_weights, v).
    attn = [self matrixMultiplicationWithPrimaryTensor:attn
                                       secondaryTensor:values
                                                  name:[NSString stringWithFormat:@"%@/matmul_v", label]];

    attn = [self transposeTensor:attn dimension:1 withDimension:2 name:[NSString stringWithFormat:@"%@/transpose_a", label]];

    return [self reshapeTensor:attn withShape:@[@(-1), @64, @(dmodel)] name:[NSString stringWithFormat:@"%@/reshape_a", label]];
}

-(nonnull MPSGraphTensor *) scaledQKMatmulWithQueries:(MPSGraphTensor * __nonnull)queries
                                             withKeys:(MPSGraphTensor * __nonnull)keys
                                                scale:(float)scale
                                                label:(NSString * __nonnull)label
{
    queries = [self reshapeTensor:queries
                        withShape:@[@(-1), @64, [queries.shape lastObject]]
                             name:[NSString stringWithFormat:@"%@/reshape_q", label]];

    keys = [self reshapeTensor:keys
                     withShape:@[@(-1), @64, [keys.shape lastObject]]
                          name:[NSString stringWithFormat:@"%@/reshape_k", label]];

    keys = [self transposeTensor:keys
                       dimension:1
                   withDimension:2
                            name:[NSString stringWithFormat:@"%@/transpose_k", label]];

    MPSGraphTensor * qkMatmul = [self matrixMultiplicationWithPrimaryTensor:queries
                                                            secondaryTensor:keys
                                                                       name:[NSString stringWithFormat:@"%@/matmul", label]];

    qkMatmul = [self multiplicationWithPrimaryTensor:qkMatmul
                                     secondaryTensor:[self constantWithScalar:scale
                                                                        shape:@[@1] dataType:qkMatmul.dataType]
                                                name:[NSString stringWithFormat:@"%@/scale", label]];
    return qkMatmul;
}

-(nonnull MPSGraphTensor *) attentionPolicyPromoMatmulConcatWithParent:(MPSGraphTensor * __nonnull)parent
                                                              withKeys:(MPSGraphTensor * __nonnull)keys
                                                               weights:(float * __nonnull)weights
                                                             inputSize:(NSUInteger)inputSize
                                                            outputSize:(NSUInteger)outputSize
                                                             sliceFrom:(NSUInteger)sliceFrom
                                                           channelSize:(NSUInteger)channelSize
                                                                 label:(NSString * __nonnull)label
{
    keys = [self reshapeTensor:keys withShape:@[@(-1), @64, @(channelSize)] name:[NSString stringWithFormat:@"%@/slice", label]];

    keys = [self sliceTensor:keys dimension:1 start:sliceFrom length:inputSize name:[NSString stringWithFormat:@"%@/slice", label]];

    NSData * weightData = [NSData dataWithBytesNoCopy:weights
                                               length:outputSize * channelSize * sizeof(float)
                                         freeWhenDone:NO];

    MPSGraphTensor * weightTensor = [self variableWithData:weightData
                                                     shape:@[@(outputSize), @(channelSize)]
                                                  dataType:parent.dataType
                                                      name:[NSString stringWithFormat:@"%@/weights", label]];

    keys = [self transposeTensor:keys dimension:1 withDimension:2 name:[NSString stringWithFormat:@"%@/transpose", label]];

    keys = [self matrixMultiplicationWithPrimaryTensor:weightTensor
                                       secondaryTensor:keys
                                                  name:[NSString stringWithFormat:@"%@/matmul", label]];

    MPSGraphTensor * offset1 = [self  sliceTensor:keys
                                        dimension:1
                                            start:0
                                           length:3
                                             name:[NSString stringWithFormat:@"%@/offset_slice_1", label]];

    MPSGraphTensor * offset2 = [self  sliceTensor:keys
                                        dimension:1
                                            start:3
                                           length:1
                                             name:[NSString stringWithFormat:@"%@/offset_slice_2", label]];

    MPSGraphTensor * promo = [self additionWithPrimaryTensor:offset1
                                             secondaryTensor:offset2
                                                        name:[NSString stringWithFormat:@"%@/offset_add", label]];

    NSMutableArray<MPSGraphTensor *> * stack = [NSMutableArray arrayWithCapacity:inputSize];
    for (NSUInteger i = 0; i < inputSize; i++) {
        [stack addObject:promo];
    }

    promo = [self stackTensors:stack axis:3 name:[NSString stringWithFormat:@"%@/offset_broadcast", label]];

    promo = [self transposeTensor:promo dimension:1 withDimension:3 name:[NSString stringWithFormat:@"%@/offset_transpose", label]];

    promo = [self reshapeTensor:promo withShape:@[@(-1), @3, @64] name:[NSString stringWithFormat:@"%@/offset_reshape", label]];

    parent = [self reshapeTensor:parent withShape:@[@(-1), @64, @64] name:[NSString stringWithFormat:@"%@/parent_reshape", label]];

    return [self concatTensor:parent withTensor:promo dimension:1 name:[NSString stringWithFormat:@"%@/concat", label]];
}

-(nonnull MPSGraphTensor *) positionEncodingWithTensor:(MPSGraphTensor * __nonnull)tensor
                                             withShape:(MPSShape * __nonnull)shape
                                               weights:(const float * __nonnull)encodings
                                                  type:(NSString * __nullable)type
                                                 label:(NSString * __nonnull)label
{
    assert([shape count] == 2 && shape[0] == tensor.shape[1]);

    NSData * encodingData = [NSData dataWithBytesNoCopy:(void *)encodings
                                                 length:[shape[0] intValue] * [shape[1] intValue] * sizeof(float)
                                           freeWhenDone:NO];

    MPSGraphTensor * encodingTensor = [self variableWithData:encodingData
                                                       shape:shape
                                                    dataType:MPSDataTypeFloat32
                                                        name:[NSString stringWithFormat:@"%@/weights", label]];

    MPSGraphTensor * shapeTensor = [self shapeOfTensor:tensor
                                                  name:[NSString stringWithFormat:@"%@/shape", label]];

    // # add positional encoding for each square to the input
    // positional_encoding = tf.broadcast_to(tf.convert_to_tensor(self.POS_ENC, dtype=self.model_dtype),
    //        [tf.shape(flow)[0], 64, tf.shape(self.POS_ENC)[2]])
    // flow = tf.concat([flow, positional_encoding], axis=2)

    // shapeTensor is (b, hw, c) and we want to make it (b, hw, hw). Since we don't know b yet, we have to manipulate this
    // tensor and use it for the broadcast op.
    // @todo look for a better way to do this.
    shapeTensor = [self sliceTensor:shapeTensor
                          dimension:0
                              start:0
                             length:2
                               name:[NSString stringWithFormat:@"%@/shape/slice", label]];

    shapeTensor = [self concatTensor:shapeTensor
                          withTensor:[self constantWithScalar:[[shape lastObject] intValue]
                                                        shape:@[@1]
                                                     dataType:shapeTensor.dataType]
                           dimension:0
                                name:[NSString stringWithFormat:@"%@/shape/concat", label]];

    encodingTensor = [self broadcastTensor:encodingTensor
                             toShapeTensor:shapeTensor
                                      name:[NSString stringWithFormat:@"%@/weights/broadcast", label]];

    encodingTensor = [self reshapeTensor:encodingTensor
                               withShape:@[@(-1), shape[0], shape[1]]
                                    name:[NSString stringWithFormat:@"%@/weights/reshape", label]];

    return [self concatTensor:tensor
                   withTensor:encodingTensor
                    dimension:[tensor.shape count] - 1
                         name:[NSString stringWithFormat:@"%@/concat", label]];
}


-(nonnull MPSGraphTensor *) dynamicPositionEncodingWithTensor:(MPSGraphTensor * __nonnull)tensor
                                                        width:(const NSUInteger)width
                                                      weights:(float * __nonnull)weights
                                                       biases:(float * __nonnull)biases
                                                        label:(NSString * __nonnull)label
{
    MPSGraphTensor * encodingTensor = [self sliceTensor:tensor
                                              dimension:2
                                                  start:0
                                                 length:12
                                                   name:[NSString stringWithFormat:@"%@/slice", label]];

    encodingTensor = [self flatten2DTensor:encodingTensor
                                      axis:1
                                      name:[NSString stringWithFormat:@"%@/flatten", label]];

    encodingTensor = [self addFullyConnectedLayerWithParent:encodingTensor
                                             outputChannels:[tensor.shape[1] intValue] * width
                                                    weights:weights
                                                     biases:biases
                                                 activation:nil
                                                      label:[NSString stringWithFormat:@"%@/dense", label]];

    encodingTensor = [self reshapeTensor:encodingTensor
                               withShape:@[@(-1), tensor.shape[1], @(width)]
                                    name:[NSString stringWithFormat:@"%@/reshape", label]];

    return [self concatTensor:tensor
                   withTensor:encodingTensor
                    dimension:[tensor.shape count] - 1
                         name:[NSString stringWithFormat:@"%@/concat", label]];
}


-(nonnull MPSGraphTensor *) addGatingLayerWithParent:(MPSGraphTensor * __nonnull)parent
                                             weights:(const float * __nonnull)weights
                                       withOperation:(NSString * __nonnull)op
                                               label:(NSString * __nonnull)label
{
    NSData * weightsData = [NSData dataWithBytesNoCopy:(void *)weights
                                                length:[parent sizeOfDimensionsFrom:@1] * sizeof(float)
                                          freeWhenDone:NO];

    MPSGraphTensor * weightsTensor = [self variableWithData:weightsData
                                                      shape:@[parent.shape[2], parent.shape[1]]
                                                   dataType:MPSDataTypeFloat32
                                                       name:[NSString stringWithFormat:@"%@/weights", label]];

    // Leela weights are transposed.
    weightsTensor = [self transposeTensor:weightsTensor
                                dimension:0
                            withDimension:1
                                     name:[NSString stringWithFormat:@"%@/weights_transpose", label]];

    if ([op isEqual:@"add"]) {
        return [self additionWithPrimaryTensor:parent
                               secondaryTensor:weightsTensor
                                          name:[NSString stringWithFormat:@"%@/add", label]];
    }
    else if ([op isEqual:@"mult"]) {
        return [self multiplicationWithPrimaryTensor:parent
                                     secondaryTensor:weightsTensor
                                                name:[NSString stringWithFormat:@"%@/multiply", label]];
    }

    return parent;
}


-(void) setGlobalSmolgenWeights:(float * __nonnull)weights
{
    _globalSmolgenWeights = weights;
}

-(nonnull MPSGraphTensor *) applyActivationWithTensor:(MPSGraphTensor * __nonnull)tensor
                                           activation:(NSString * __nullable)activation
                                                label:(NSString * __nullable)label
{
    if ([activation isEqual:@"relu"]) {
        return [self reLUWithTensor:tensor name:[NSString stringWithFormat:@"%@/relu", label]];
    }
    if ([activation isEqual:@"relu_2"]) {
        tensor = [self reLUWithTensor:tensor name:[NSString stringWithFormat:@"%@/relu", label]];
        return [self multiplicationWithPrimaryTensor:tensor
                                     secondaryTensor:tensor
                                                name:[NSString stringWithFormat:@"%@/square", label]];
    }
    else if ([activation isEqual:@"tanh"]) {
        return [self tanhWithTensor:tensor name:[NSString stringWithFormat:@"%@/tanh", label]];
    }
    else if ([activation isEqual:@"sigmoid"]) {
        return [self sigmoidWithTensor:tensor name:[NSString stringWithFormat:@"%@/sigmoid", label]];
    }
    else if ([activation isEqual:@"softmax"]) {
        return [self softMaxWithTensor:tensor axis:([tensor.shape count] - 1) name:[NSString stringWithFormat:@"%@/softmax", label]];
    }
    else if ([activation isEqual:@"selu"]) {
        return [self seluWithTensor:tensor label:[NSString stringWithFormat:@"%@/mish", label]];
    }
    else if ([activation isEqual:@"mish"]) {
        return [self mishWithTensor:tensor label:[NSString stringWithFormat:@"%@/mish", label]];
    }
    else if ([activation isEqual:@"swish"]) {
        return [self swishWithTensor:tensor beta:1.0 label:[NSString stringWithFormat:@"%@/swish", label]];
    }

    return tensor;
}

-(nonnull MPSGraphTensor *) mishWithTensor:(MPSGraphTensor * __nonnull)tensor
                                     label:(NSString * __nonnull)label
{
    // mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    MPSGraphTensor * mishTensor = [self exponentWithTensor:tensor
                                                      name:[NSString stringWithFormat:@"%@/exp", label]];

    MPSGraphTensor * oneTensor = [self constantWithScalar:1.0 shape:@[@1] dataType:mishTensor.dataType];
    mishTensor = [self additionWithPrimaryTensor:mishTensor
                                 secondaryTensor:oneTensor
                                            name:[NSString stringWithFormat:@"%@/add", label]];

    mishTensor = [self logarithmWithTensor:mishTensor name:[NSString stringWithFormat:@"%@/ln", label]];

    mishTensor = [self tanhWithTensor:mishTensor name:[NSString stringWithFormat:@"%@/tanh", label]];

    mishTensor = [self multiplicationWithPrimaryTensor:mishTensor
                                       secondaryTensor:tensor
                                                  name:[NSString stringWithFormat:@"%@/multiply", label]];

    return mishTensor;
}

-(nonnull MPSGraphTensor *) swishWithTensor:(MPSGraphTensor * __nonnull)tensor
                                       beta:(float)beta
                                      label:(NSString * __nonnull)label
{
    // swish(x) = x * sigmoid(β * x)
    MPSGraphTensor * betaTensor = [self constantWithScalar:beta shape:@[@1] dataType:tensor.dataType];
    MPSGraphTensor * swish = [self multiplicationWithPrimaryTensor:tensor
                                                   secondaryTensor:betaTensor
                                                              name:[NSString stringWithFormat:@"%@/multiply", label]];
    swish = [self sigmoidWithTensor:swish
                               name:[NSString stringWithFormat:@"%@/sigmoid", label]];

    return [self multiplicationWithPrimaryTensor:tensor
                                 secondaryTensor:swish
                                            name:[NSString stringWithFormat:@"%@/multiply_2", label]];

}

-(nonnull MPSGraphTensor *) seluWithTensor:(MPSGraphTensor * __nonnull)tensor
                                     label:(NSString * __nonnull)label
{
    // SELU:
    // if x > 0: return scale * x
    // if x < 0: return scale * alpha * (exp(x) - 1)
    // alpha=1.67326324, scale=1.05070098
    MPSGraphTensor * zero = [self constantWithScalar:0.0 shape:@[@1] dataType:tensor.dataType];
    MPSGraphTensor * scale = [self constantWithScalar:1.05070098 shape:@[@1] dataType:tensor.dataType];
    MPSGraphTensor * alpha = [self constantWithScalar:1.67326324 shape:@[@1] dataType:tensor.dataType];

    MPSGraphTensor * lessThanZero = [self lessThanWithPrimaryTensor:tensor
                                                    secondaryTensor:zero
                                                               name:[NSString stringWithFormat:@"%@/ltzero", label]];

    MPSGraphTensor * greaterThanZero = [self greaterThanOrEqualToWithPrimaryTensor:tensor
                                                                   secondaryTensor:zero
                                                                              name:[NSString stringWithFormat:@"%@/gtzero", label]];

    MPSGraphTensor * scaled = [self multiplicationWithPrimaryTensor:tensor
                                                    secondaryTensor:scale
                                                               name:[NSString stringWithFormat:@"%@/scale", label]];

    scaled = [self multiplicationWithPrimaryTensor:scaled
                                   secondaryTensor:greaterThanZero
                                              name:[NSString stringWithFormat:@"%@/scale_mask", label]];

    MPSGraphTensor * exp = [self exponentWithTensor:tensor
                                               name:[NSString stringWithFormat:@"%@/exp", label]];

    MPSGraphTensor * one = [self constantWithScalar:1.0 shape:@[@1] dataType:tensor.dataType];
    exp = [self subtractionWithPrimaryTensor:exp
                             secondaryTensor:one
                                        name:[NSString stringWithFormat:@"%@/exp_1", label]];

    exp = [self multiplicationWithPrimaryTensor:exp
                                secondaryTensor:alpha
                                           name:[NSString stringWithFormat:@"%@/exp_alpha", label]];

    exp = [self multiplicationWithPrimaryTensor:exp
                                secondaryTensor:scale
                                           name:[NSString stringWithFormat:@"%@/exp_scale", label]];

    exp = [self multiplicationWithPrimaryTensor:exp
                                secondaryTensor:lessThanZero
                                           name:[NSString stringWithFormat:@"%@/exp_mask", label]];

    return [self additionWithPrimaryTensor:scaled secondaryTensor:exp name:[NSString stringWithFormat:@"%@/sum", label]];
}

-(nonnull MPSGraphTensor *) makePolicyHeadWithTensor:(MPSGraphTensor * __nonnull)policy
                                     attentionPolicy:(BOOL)attentionPolicy
                                   convolutionPolicy:(BOOL)convolutionPolicy
                                       attentionBody:(BOOL)attentionBody
                                   defaultActivation:(NSString * __nullable)defaultActivation
                                   smolgenActivation:(NSString * __nullable)smolgenActivation
                                       ffnActivation:(NSString * __nullable)ffnActivation
                                          policyHead:(lczero::MultiHeadWeights::PolicyHead &)head
                                               label:(NSString * __nonnull)label
{
    if (attentionPolicy) {
        // Not implemented yet!
        // tokens = tf.reverse(policy_tokens, axis=[1]) if opponent else policy_tokens

        // 2. Square Embedding: Dense with default activation (or SELU for old ap-mish nets).
        NSUInteger embeddingSize = head.ip_pol_b.size();
        NSUInteger policyDModel = head.ip2_pol_b.size();
        // ap-mish uses hardcoded SELU
        policy = [self addFullyConnectedLayerWithParent:policy
                                         outputChannels:embeddingSize
                                                weights:&head.ip_pol_w[0]
                                                 biases:&head.ip_pol_b[0]
                                             activation:attentionBody ? defaultActivation : @"selu"
                                                  label:[NSString stringWithFormat:@"%@/fc_embed", label]];

        // 3. Encoder layers
        for (NSUInteger i = 0; i < head.pol_encoder.size(); i++) {
            policy = [self addEncoderLayerWithParent:policy
                                       legacyWeights:head.pol_encoder[i]
                                               heads:head.pol_encoder_head_count
                                       embeddingSize:embeddingSize
                                   smolgenActivation:attentionBody ? smolgenActivation : nil
                                       ffnActivation:attentionBody ? ffnActivation : @"selu"
                                               alpha:1.0
                                             epsilon:1e-6
                                            normtype:@"layernorm"
                                               label:[NSString stringWithFormat:@"%@/encoder_%zu", label, i]];
        }

        // 4. Self-attention q and k.
        MPSGraphTensor * queries = [self addFullyConnectedLayerWithParent:policy
                                                           outputChannels:policyDModel
                                                                  weights:&head.ip2_pol_w[0]
                                                                   biases:&head.ip2_pol_b[0]
                                                               activation:nil
                                                                    label:[NSString stringWithFormat:@"%@/self_attention/q", label]];

        MPSGraphTensor * keys = [self addFullyConnectedLayerWithParent:policy
                                                        outputChannels:policyDModel
                                                               weights:&head.ip3_pol_w[0]
                                                                biases:&head.ip3_pol_b[0]
                                                            activation:nil
                                                                 label:[NSString stringWithFormat:@"%@/self_attention/k", label]];

        // 5. matmul(q,k) / sqrt(dk)
        policy = [self scaledQKMatmulWithQueries:queries
                                        withKeys:keys
                                           scale:1.0f / sqrt(policyDModel)
                                           label:[NSString stringWithFormat:@"%@/self_attention/kq", label]];

        // 6. Slice last 8 keys (k[:, 56:, :]) and matmul with policy promotion weights, then concat to matmul_qk.
        policy = [self attentionPolicyPromoMatmulConcatWithParent:policy
                                                         withKeys:keys
                                                          weights:&head.ip4_pol_w[0]
                                                        inputSize:8
                                                       outputSize:4
                                                        sliceFrom:56
                                                      channelSize:policyDModel
                                                            label:[NSString stringWithFormat:@"%@/promo_logits", label]];
    }
    else if (convolutionPolicy) {
        if (attentionBody) {
            [NSException raise:@"Unsupported architecture."
                        format:@"Convolutional policy not supported with attention body."];
        }
        policy = [self addConvolutionBlockWithParent:policy
                                      outputChannels:head.policy1.biases.size()
                                          kernelSize:3
                                             weights:&head.policy1.weights[0]
                                              biases:&head.policy1.biases[0]
                                          activation:defaultActivation
                                               label:[NSString stringWithFormat:@"%@/conv1", label]];

        // No activation.
        policy = [self addConvolutionBlockWithParent:policy
                                      outputChannels:head.policy.biases.size()
                                          kernelSize:3
                                             weights:&head.policy.weights[0]
                                              biases:&head.policy.biases[0]
                                          activation:nil
                                               label:[NSString stringWithFormat:@"%@/conv2", label]];


        /**
         * @todo policy map implementation has bug in MPSGraph (GatherND not working in graph).
         * Implementation of policy map to be done in CPU for now.
         *
         * Reinstate this section when bug is fixed. See comments below.
         *
         // [1858 -> HWC or CHW]
         const bool HWC = false;
         std::vector<uint32_t> policy_map(1858);
         for (const auto& mapping : kConvPolicyMap) {
         if (mapping == -1) continue;
         const auto index = &mapping - kConvPolicyMap;
         const auto displacement = index / 64;
         const auto square = index % 64;
         const auto row = square / 8;
         const auto col = square % 8;
         if (HWC) {
         policy_map[mapping] = ((row * 8) + col) * 80 + displacement;
         } else {
         policy_map[mapping] = ((displacement * 8) + row) * 8 + col;
         }
         }
         policy = builder_->makePolicyMapLayer(policy, &policy_map[0], "policy_map");
         */
    }
    else {
        if (attentionBody) {
            [NSException raise:@"Unsupported architecture."
                        format:@"Classical policy not supported with attention body."];
        }

        const int policySize = head.policy.biases.size();

        policy = [self addConvolutionBlockWithParent:policy
                                      outputChannels:policySize
                                          kernelSize:1
                                             weights:&head.policy.weights[0]
                                              biases:&head.policy.biases[0]
                                          activation:defaultActivation
                                               label:[NSString stringWithFormat:@"%@/conv", label]];

        policy = [self flatten2DTensor:policy
                                  axis:1
                                  name:[NSString stringWithFormat:@"%@/conv/flatten", label]];

        // ip_pol_w and ip_pol_b as used here is for classical policy dense weights,
        // may be worth renaming to dismbiguate policy embedding weights in attention policy.
        policy = [self addFullyConnectedLayerWithParent:policy
                                         outputChannels:head.ip_pol_b.size()
                                                weights:&head.ip_pol_w[0]
                                                 biases:&head.ip_pol_b[0]
                                             activation:nil
                                                  label:[NSString stringWithFormat:@"%@/fc", label]];
    }
    return policy;
}

-(nonnull MPSGraphTensor *) makeValueHeadWithTensor:(MPSGraphTensor * __nonnull)value
                                      attentionBody:(BOOL)attentionBody
                                            wdlHead:(BOOL)wdl
                                  defaultActivation:(NSString * __nullable)defaultActivation
                                          valueHead:(lczero::MultiHeadWeights::ValueHead &)head
                                              label:(NSString * __nonnull)label
{
    if (attentionBody) {
        value = [self addFullyConnectedLayerWithParent:value
                                        outputChannels:head.ip_val_b.size()
                                               weights:&head.ip_val_w[0]
                                                biases:&head.ip_val_b[0]
                                            activation:defaultActivation
                                                 label:[NSString stringWithFormat:@"%@/embedding", label]];
    }
    else {
        value = [self addConvolutionBlockWithParent:value
                                     outputChannels:head.value.biases.size()
                                         kernelSize:1
                                            weights:&head.value.weights[0]
                                             biases:&head.value.biases[0]
                                         activation:defaultActivation
                                              label:[NSString stringWithFormat:@"%@/conv", label]];
    }

    value = [self flatten2DTensor:value
                             axis:1
                             name:@"value/flatten"];

    value = [self addFullyConnectedLayerWithParent:value
                                    outputChannels:head.ip1_val_b.size()
                                           weights:&head.ip1_val_w[0]
                                            biases:&head.ip1_val_b[0]
                                        activation:defaultActivation
                                             label:[NSString stringWithFormat:@"%@/fc1", label]];

    value = [self addFullyConnectedLayerWithParent:value
                                    outputChannels:head.ip2_val_b.size()
                                            weights:&head.ip2_val_w[0]
                                            biases:&head.ip2_val_b[0]
                                        activation:wdl ? @"softmax" : @"tanh"
                                                label:[NSString stringWithFormat:@"%@/fc2", label]];

    return value;
}

@end
