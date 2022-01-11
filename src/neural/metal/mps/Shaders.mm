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

#import "Shaders.h"

@implementation SEMultiply {
    id <MTLLibrary> _library;
    BOOL _nonuniformThreadgroups;
    id <MTLComputePipelineState> _computePipeline;
}

-(nonnull instancetype) initWithDevice:(nonnull id <MTLDevice>) device
                         gammaChannels:(NSUInteger) gammaChannels
                          betaChannels:(NSUInteger) betaChannels
{
    self = [super initWithDevice:device];
    if (nil == self) return nil;
    
    self.gammaChannels = gammaChannels;
    self.betaChannels = betaChannels;
    
    _library = [device newDefaultLibrary];
    MTLFunctionConstantValues * constantValues = [MTLFunctionConstantValues alloc];
    //[constantValues setConstantValue:_nonuniformThreadgroups type:MTLDataTypeBool atIndex:0];
    
    // Create command encoder.
    NSError * error = nil;
    id<MTLFunction> seMultiply = [_library newFunctionWithName:@"seMultiply" constantValues:constantValues];
    _computePipeline = [device newComputePipelineStateWithFunction:seMultiply error:&error];
    if (!_computePipeline)
    {
        NSLog(@"Failed to create compute pipeline state, error %@", error);
    }

    return self;
}

-(nonnull MPSImageBatch *) encodeBatchToCommandBuffer:(nonnull id <MTLCommandBuffer>) commandBuffer
                                         sourceImages:(MPSImageBatch * __nonnull) sourceImageBatch
{
    MPSImageBatch * resultBatch = @[];
    MPSImage * image = sourceImageBatch[0];
    MPSImageDescriptor * descriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:image.featureChannelFormat
                                                                                     width:image.width
                                                                                    height:image.height
                                                                           featureChannels:image.featureChannels
                                                                            numberOfImages:image.numberOfImages
                                                                                     usage:image.usage];
    resultBatch = [self.destinationImageAllocator imageBatchForCommandBuffer:commandBuffer
                                                             imageDescriptor:descriptor
                                                                      kernel:self
                                                                       count:[sourceImageBatch count]];

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"Encoder";

    [encoder setComputePipelineState:_computePipeline];

//    [encoder setTexture:source index:0];
//    [encoder setTexture:destination index:1];

//    [encoder setBytes(&self.temperature, length: MemoryLayout<Float>.stride, index: 0)
//    [encoder setBytes(&self.tint, length: MemoryLayout<Float>.stride, index: 1)

    MTLSize gridSize = MTLSizeMake(image.width, image.height, image.featureChannels);
    int threadGroupWidth = _computePipeline.threadExecutionWidth;
    int threadGroupHeight = _computePipeline.maxTotalThreadsPerThreadgroup / threadGroupWidth;
    MTLSize threadGroupSize = MTLSizeMake(threadGroupWidth, threadGroupHeight, 1);
    int threadgroupMemoryLength = _computePipeline.threadExecutionWidth * sizeof(vector_float4);


    if (_nonuniformThreadgroups) {
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    } else {
        MTLSize threadGroupCount = MTLSizeMake((gridSize.width + threadGroupSize.width - 1) / threadGroupSize.width,
                                               (gridSize.height + threadGroupSize.height - 1) / threadGroupSize.height,
                                               1);
        [encoder dispatchThreadgroups:threadGroupCount threadsPerThreadgroup:threadGroupSize];
    }

    [encoder endEncoding];

    return resultBatch;
}

@end
