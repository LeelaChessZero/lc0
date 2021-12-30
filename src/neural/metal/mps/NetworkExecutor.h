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

#import "MetalNetworkDelegate.h"
#import "ConvWeights.h"

static MPSImageFeatureChannelFormat fcFormat = MPSImageFeatureChannelFormatFloat16;

@interface Lc0NetworkExecutor : NSObject {
 @public
  // Keep the MTLDevice and MTLCommandQueue objects around for ease of use.
  id <MTLDevice> device;
  id <MTLCommandQueue> queue;

  NSArray <ConvWeights *> *allWeights;
  MPSNNGraph *inferenceGraph;
}

-(nonnull instancetype) initWithDevice:(nonnull id <MTLDevice>)inputDevice
                          commandQueue:(nonnull id <MTLCommandQueue>)commandQueue;
-(nonnull MPSNNFilterNode *) buildInferenceGraph;
-(MPSImageBatch * __nullable) encodeInferenceBatchToCommandBuffer:(nonnull id <MTLCommandBuffer>) commandBuffer
                                                    sourceImages:(MPSImageBatch * __nonnull) sourceImage;

@end

