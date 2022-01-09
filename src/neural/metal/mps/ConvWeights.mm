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

#import "ConvWeights.h"

@implementation ConvWeights

-(nonnull instancetype) initWithDevice:(id <MTLDevice> __nonnull)device
                         inputChannels:(NSUInteger)inputChannels
                        outputChannels:(NSUInteger)outputChannels
                           kernelWidth:(NSUInteger)kernelWidth
                          kernelHeight:(NSUInteger)kernelHeight
                                stride:(NSUInteger)stride
                               weights:(float * __nonnull)weights
                                biases:(float * __nonnull)biases
                                 label:(NSString * __nonnull)label
                       fusedActivation:(NSString * __nullable)fusedActivation
{
    self = [super init];
    if( nil == self )
        return nil;
    
    _device = device;

    _label = label;
    _outputChannels = outputChannels;
    _inputChannels = inputChannels;
    _kernelWidth = kernelWidth;
    _kernelHeight = kernelHeight;

    _descriptor = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:kernelWidth
                                                                          kernelHeight:kernelHeight
                                                                  inputFeatureChannels:inputChannels
                                                                 outputFeatureChannels:outputChannels];

    _descriptor.strideInPixelsX = stride;
    _descriptor.strideInPixelsY = stride;
    // Fuse activations into the convolution.
    if ([fusedActivation isEqual:@"relu"]) {
        NSLog(@"Using fused relu.");
        _descriptor.fusedNeuronDescriptor = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:(MPSCNNNeuronTypeReLU)];
    }
    else if ([fusedActivation isEqual:@"tanh"]) {
        NSLog(@"Using fused tanh.");
        _descriptor.fusedNeuronDescriptor = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:(MPSCNNNeuronTypeTanH)];
    }
    else {
        NSLog(@"Using fused none.");
        _descriptor.fusedNeuronDescriptor = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:(MPSCNNNeuronTypeNone)];
    }

    // @todo took stuff from here.
    _weights = weights;
    _biases = biases;
    
    return self;
}

-(MPSDataType) dataType {return  MPSDataTypeFloat32;}
-(nonnull MPSCNNConvolutionDescriptor *) descriptor {return _descriptor;}
-(void * __nonnull) weights {return _weightPointer;}
-(float * __nullable) biasTerms {return _biasPointer;}

-(BOOL) load {
    //NSLog(@"Loading weights...");
    //[self checkpointWithCommandQueue:gCommandQueue];

    // Calculating the size of weights and biases.
    _sizeBiases = _outputChannels * sizeof(float);
    NSUInteger lenWeights = _outputChannels * _kernelHeight * _kernelWidth * _inputChannels;
    _sizeWeights = lenWeights * sizeof(float);
    
    _weightDescriptor = [MPSVectorDescriptor vectorDescriptorWithLength:lenWeights dataType:(MPSDataTypeFloat32)];
    _weightVector = [[MPSVector alloc] initWithDevice:_device descriptor:_weightDescriptor];
    
    _biasDescriptor = [MPSVectorDescriptor vectorDescriptorWithLength:_outputChannels dataType:(MPSDataTypeFloat32)];
    _biasVector = [[MPSVector alloc] initWithDevice:_device descriptor:_biasDescriptor];
    
    //_convWtsAndBias = [[MPSCNNConvolutionWeightsAndBiasesState alloc] initWithWeights:_weightVector.data biases:_biasVector.data];
    
    // Transpose weights as needed from OIHW (Leela) to OHWI (MPSNNGraph).
    NSArray * shape = @[@(_outputChannels), @(_inputChannels), @(_kernelHeight), @(_kernelWidth)];
    MPSNDArrayDescriptor * arrayDesc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32 shape:shape];
    MPSNDArray * initial = [[MPSNDArray alloc] initWithDevice:_device descriptor:arrayDesc];
    [initial writeBytes:_weights strideBytes:nil];
    
    // Note that dimension 0 refers to the fastest changing dimension.
    [arrayDesc transposeDimension:2 withDimension:1];
    [arrayDesc transposeDimension:1 withDimension:0];
    id<MTLCommandQueue> queue = [_device newCommandQueue];
    MPSCommandBuffer *commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
    MPSNDArray * transposed = [initial arrayViewWithCommandBuffer:commandBuffer
                                                       descriptor:arrayDesc
                                                         aliasing:MPSAliasingStrategyDefault];
    float buffer[lenWeights];
    [transposed readBytes:buffer strideBytes:nil];
    
    NSLog(@"Original weights:");
    for (int j = 0; j < 64; ++j) {
        NSLog(@"Weight[%i]: %f", j, _weights[j]);
    }
    
    NSLog(@" ");
    NSLog(@"Transposed weights:");
    for (int j = 0; j < 64; ++j) {
        NSLog(@"Weight[%i]: %f", j, buffer[j]);
    }
    
    NSLog(@" ");
    NSLog(@"Reordered transposed weights:");
    for (int j = 0; j < 64; ++j) {
        NSLog(@"Weight[%i x %i]: %f", j, _inputChannels, buffer[j * _inputChannels]);
    }
    
    // Set weights and biases to the specified values.
    _weightPointer = (float *)_weightVector.data.contents;
    memcpy((void *)_weightPointer, (void *)buffer, _sizeWeights);
    
    _biasPointer = (float *)_biasVector.data.contents;
    memcpy((void *)_biasPointer, (void *)_biases, _sizeBiases);
    
    // Notify event listeners.
    [_weightVector.data didModifyRange:NSMakeRange(0, _sizeWeights)];
    [_biasVector.data didModifyRange:NSMakeRange(0, _sizeBiases)];
    
    return YES;
}

-(void) purge {
    //NSLog(@"Purging weights...");
}

- (NSString * _Nullable)label {
    return _label;
}

-(nonnull id)copyWithZone:(nullable NSZone *)zone {
    /* unimplemented */
    return self;
}

-(void) describeWeights {
    float * p = _weightPointer;
    int lenWeights = _inputChannels * _kernelHeight * _kernelWidth * _outputChannels;
    for (int i=2000; i<3000; i++) {
        NSLog(@"Final Weight[%i]: %f", i, *(p + i));
    }
    
    float * q = _biasPointer;
    for (int i=0; i<_outputChannels; i++) {
        NSLog(@"Final Bias[%i]: %f", i, *(q + i));
    }
}

@end    /* ConvWeights */
