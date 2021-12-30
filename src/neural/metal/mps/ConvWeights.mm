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
#import <random>

@implementation ConvWeights

-(nonnull instancetype) initWithDevice:(nonnull id <MTLDevice>)device
                         inputChannels:(NSUInteger)inputChannels
                        outputChannels:(NSUInteger)outputChannels
                           kernelWidth:(NSUInteger)kernelWidth
                          kernelHeight:(NSUInteger)kernelHeight
                                stride:(NSUInteger)stride
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

    // Initializing weights, biases and their corresponding values.
    _weightPointer = (float *)_weightVector.data.contents;
    float zero = 0.f;
    memset_pattern4( (void *)_weightPointer, (char *)&zero, _sizeWeights);
    
    _biasPointer = (float *)_biasVector.data.contents;
    float biasInit = 0.1f;
    memset_pattern4( (void *)_biasPointer, (char *)&biasInit, _sizeBiases);


    // Setting weights to random values.
    /*MPSMatrixRandomDistributionDescriptor *randomDesc = [MPSMatrixRandomDistributionDescriptor uniformDistributionDescriptorWithMinimum:-0.2f maximum:0.2f];
    MPSMatrixRandomMTGP32 *randomKernel = [[MPSMatrixRandomMTGP32 alloc] initWithDevice:device
                                                                    destinationDataType:MPSDataTypeFloat32
                                                                                   seed:_seed
                                                                 distributionDescriptor:randomDesc];

    MPSCommandBuffer *commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:gCommandQueue];
    [randomKernel encodeToCommandBuffer:commandBuffer
                      destinationVector:_weightVector];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];


    [_weightMomentumVector.data didModifyRange:NSMakeRange(0, _sizeWeights)];
    [_weightVelocityVector.data didModifyRange:NSMakeRange(0, _sizeWeights)];
    [_biasVector.data didModifyRange:NSMakeRange(0, _sizeBiases)];
    [_biasMomentumVector.data didModifyRange:NSMakeRange(0, _sizeBiases)];
    [_biasVelocityVector.data didModifyRange:NSMakeRange(0, _sizeBiases)];*/

    return self;
}

-(MPSDataType)  dataType{return  MPSDataTypeFloat32;}
-(MPSCNNConvolutionDescriptor * __nonnull) descriptor{return _descriptor;}
-(void * __nonnull) weights{return _weightPointer;}
-(float * __nullable) biasTerms{return _biasPointer;};

-(BOOL) load{
    //[self checkpointWithCommandQueue:gCommandQueue];
    return YES;
}

-(void) purge{};


- (NSString * _Nullable)label {
    return _label;
}

- (nonnull id)copyWithZone:(nullable NSZone *)zone {
    /* unimplemented */
    return self;
}

@end    /* ConvWeights */
