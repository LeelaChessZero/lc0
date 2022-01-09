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
#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>


static MPSNNDefaultPadding * _Nonnull sameConvPadding = [MPSNNDefaultPadding paddingWithMethod: MPSNNPaddingMethodAddRemainderToTopLeft | MPSNNPaddingMethodAlignCentered | MPSNNPaddingMethodSizeSame];
static MPSNNDefaultPadding * _Nonnull validConvPadding = [MPSNNDefaultPadding paddingWithMethod: MPSNNPaddingMethodAlignCentered | MPSNNPaddingMethodAddRemainderToTopLeft | MPSNNPaddingMethodSizeValidOnly];

static MPSNNDefaultPadding * _Nonnull samePoolingPadding = [MPSNNDefaultPadding paddingForTensorflowAveragePooling];
static MPSNNDefaultPadding * _Nonnull validPoolingPadding = [MPSNNDefaultPadding paddingForTensorflowAveragePoolingValidOnly];

/*!
 *  @class      ConvWeights
 *  @dependency This depends on Metal.framework
 *
 */
@interface ConvWeights : NSObject<MPSCNNConvolutionDataSource> {
@private
    NSUInteger _outputChannels;
    NSUInteger _inputChannels;
    NSUInteger _kernelHeight;
    NSUInteger _kernelWidth;
    MPSCNNConvolutionDescriptor *_descriptor;
    NSString *_label;
    float *_biasPointer, *_weightPointer, *_biases, *_weights;
    size_t _sizeBiases, _sizeWeights;
    unsigned _seed;

    MPSVector *_weightVector, *_biasVector;
    MPSVectorDescriptor *_weightDescriptor;
    MPSVectorDescriptor *_biasDescriptor;
    id<MTLDevice> _device;

@public
    
}

-(nonnull instancetype) initWithDevice:(id <MTLDevice> __nonnull)device
                         inputChannels:(NSUInteger)inputChannels
                        outputChannels:(NSUInteger)outputChannels
                           kernelWidth:(NSUInteger)kernelWidth
                          kernelHeight:(NSUInteger)kernelHeight
                                stride:(NSUInteger)stride
                               weights:(float * __nonnull)weights
                                biases:(float * __nonnull)biases
                                 label:(NSString * __nonnull)label
                       fusedActivation:(NSString * __nullable)fusedActivation;

-(MPSDataType) dataType;
-(MPSCNNConvolutionDescriptor * __nonnull) descriptor;
-(void * __nonnull) weights;
-(float * __nullable) biasTerms;
-(BOOL) load;
-(void) purge;
-(void) describeWeights;

@end    /* ConvWeights */
