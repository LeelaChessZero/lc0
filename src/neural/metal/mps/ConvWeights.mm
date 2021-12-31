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
{
    self = [super init];
    if( nil == self )
        return nil;

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
    _descriptor.fusedNeuronDescriptor = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:(MPSCNNNeuronTypeNone)];

    // Calculating the size of weights and biases.
    _sizeBiases = _outputChannels * sizeof(float);
    NSUInteger lenWeights = _inputChannels * _kernelHeight * _kernelWidth * _outputChannels;
    _sizeWeights = lenWeights * sizeof(float);


    _weightDescriptor = [MPSVectorDescriptor vectorDescriptorWithLength:lenWeights dataType:(MPSDataTypeFloat32)];
    _weightVector = [[MPSVector alloc] initWithDevice:device descriptor:_weightDescriptor];

    _biasDescriptor = [MPSVectorDescriptor vectorDescriptorWithLength:_outputChannels dataType:(MPSDataTypeFloat32)];
    _biasVector = [[MPSVector alloc] initWithDevice:device descriptor:_biasDescriptor];


    //_convWtsAndBias = [[MPSCNNConvolutionWeightsAndBiasesState alloc] initWithWeights:_weightVector.data biases:_biasVector.data];
    
    // Set weights and biases to the specified values.
    _weightPointer = (float *)_weightVector.data.contents;
    memcpy((void *)_weightPointer, (void *)weights, _sizeWeights);
    
    _biasPointer = (float *)_biasVector.data.contents;
    memcpy((void *)_biasPointer, (void *)biases, _sizeBiases);
    
    
    // Notify event listeners.
    [_weightVector.data didModifyRange:NSMakeRange(0, _sizeWeights)];
    [_biasVector.data didModifyRange:NSMakeRange(0, _sizeBiases)];
    
    return self;
}

-(MPSDataType) dataType {return  MPSDataTypeFloat32;}
-(nonnull MPSCNNConvolutionDescriptor *) descriptor {return _descriptor;}
-(nonnull float *) weights {return _weightPointer;}
-(nonnull float *) biasTerms {return _biasPointer;}

-(BOOL) load {
    //[self checkpointWithCommandQueue:gCommandQueue];
    return YES;
}

-(void) purge {}

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
