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

// Base Kernel Class
@interface Lc0Kernel : MPSCNNKernel {
    BOOL _nonuniformThreadgroups;
    id <MTLLibrary> _library;
    id <MTLFunction> _kernelFunction;
    id <MTLComputePipelineState> _computePipeline;
}

-(nonnull instancetype) initWithDevice:(nonnull id <MTLDevice>) device;


-(nonnull MPSImageBatch *) encodeBatchToCommandBuffer:(nonnull id <MTLCommandBuffer>) commandBuffer
                                         sourceImages:(MPSImageBatch * __nonnull) sourceImageBatch;


@end


@interface Lc0SeMultiplyAdd : Lc0Kernel

@property(readwrite) NSUInteger gammaChannels;

@property(readwrite) NSUInteger betaChannels;

-(nonnull instancetype) initWithDevice:(nonnull id <MTLDevice>) device
                         gammaChannels:(NSUInteger) gammaChannels
                          betaChannels:(NSUInteger) betaChannels
                               hasRelu:(BOOL) hasRelu;


-(nonnull MPSImageBatch *) encodeBatchToCommandBuffer:(nonnull id <MTLCommandBuffer>) commandBuffer
                                       seSourceImages:(MPSImageBatch * __nonnull) seSourceImageBatch
                                     convSourceImages:(MPSImageBatch * __nonnull) convSourceImageBatch
                                     skipSourceImages:(MPSImageBatch * __nonnull) skipSourceImageBatch;


@end




// Flatten
@interface Lc0Flatten : Lc0Kernel

@end


// Policy Map
@interface Lc0PolicyMap : Lc0Kernel

-(nonnull instancetype) initWithDevice:(nonnull id <MTLDevice>) device
                             policyMap:(nonnull short *) policyMap;

@end
