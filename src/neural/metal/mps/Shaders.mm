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

#include "Utilities.h"


@implementation SEMultiply {
    id <MTLLibrary> _library;
    BOOL _nonuniformThreadgroups;
    id <MTLComputePipelineState> _computePipeline;
    BOOL _hasRelu;
}

-(nonnull instancetype) initWithDevice:(nonnull id <MTLDevice>) device
                         gammaChannels:(NSUInteger) gammaChannels
                          betaChannels:(NSUInteger) betaChannels
                               hasRelu:(BOOL) hasRelu
{
    self = [super initWithDevice:device];
    if (nil == self) return nil;
    
    self.gammaChannels = gammaChannels;
    self.betaChannels = betaChannels;
    _hasRelu = hasRelu;
        
    NSError *libraryError = nil;
//    NSString *libraryFile = [[NSBundle mainBundle] pathForResource:@"Shaders.metallib" ofType:@"metallib"];
    _library = [device newLibraryWithFile:@"build/release/src/neural/metal/mps/Shaders.metallib" error:&libraryError];
    //_library = [device newLibraryWithFile:@"Shaders.metallib" error:&libraryError];
    //_library = [device newDefaultLibrary];
    //MTLFunctionConstantValues * constantValues = [MTLFunctionConstantValues alloc];
    //[constantValues setConstantValue:&_nonuniformThreadgroups type:MTLDataTypeBool atIndex:0];
    if (!_library) {
        NSLog(@"Failed to create kernel library, error: %@", libraryError);
        return nil;
    }
    
    // Create command encoder.
    NSError * kernelError = nil;
    //id<MTLFunction> seMultiply = [_library newFunctionWithName:@"seMultiply" constantValues:constantValues error:&kernelError];
    id<MTLFunction> seMultiply = [_library newFunctionWithName:@"seMultiply"];
    if (!seMultiply)
    {
        NSLog(@"Failed to create kernel function, error %@", kernelError);
        return nil;
    }
    NSLog(@"Function: %@", seMultiply);
    
    NSError * pipelineError = nil;
    _computePipeline = [device newComputePipelineStateWithFunction:seMultiply error:&pipelineError];
    if (!_computePipeline)
    {
        NSLog(@"Failed to create compute pipeline state, error %@", pipelineError);
        return nil;
    }

    return self;
}

-(nonnull MPSImageBatch *) encodeBatchToCommandBuffer:(id <MTLCommandBuffer> __nonnull) commandBuffer
                                       seSourceImages:(MPSImageBatch * __nonnull) seSourceImageBatch
                                     convSourceImages:(MPSImageBatch * __nonnull) convSourceImageBatch
                                     skipSourceImages:(MPSImageBatch * __nonnull) skipSourceImageBatch
{
    assert([seSourceImageBatch count] == 2 * [convSourceImageBatch count]);
    MPSImageBatch * resultBatch = @[];
    MPSImage * convImage = convSourceImageBatch[0];
    MPSImageDescriptor * resultDescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:convImage.featureChannelFormat
                                                                                           width:convImage.width
                                                                                          height:convImage.height
                                                                                 featureChannels:convImage.featureChannels
                                                                                  numberOfImages:convImage.numberOfImages
                                                                                           usage:MTLTextureUsageShaderRead/* | MTLTextureUsageShaderWrite*/];
    resultDescriptor.storageMode = MTLStorageModeShared;
    resultBatch = [self.destinationImageAllocator imageBatchForCommandBuffer:commandBuffer
                                                             imageDescriptor:resultDescriptor
                                                                      kernel:self
                                                                       count:[convSourceImageBatch count]];
    
    MPSImageDescriptor * seMultiplierDescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:convImage.featureChannelFormat
                                                                                                 width:convImage.width
                                                                                                height:convImage.height
                                                                                       featureChannels:self.gammaChannels
                                                                                        numberOfImages:convImage.numberOfImages
                                                                                                 usage:MTLTextureUsageShaderRead/* | MTLTextureUsageShaderWrite*/];
    

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"Encoder";

    [encoder setComputePipelineState:_computePipeline];
    
    int bufferSize = convImage.width * convImage.height * convImage.featureChannels * convImage.numberOfImages * [convSourceImageBatch count];
    id<MTLBuffer> buffer = [[commandBuffer device] newBufferWithLength:bufferSize * sizeof(float) options:nil];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    
    NSLog(@"Buffer: %@", buffer);

    //[seSourceImageBatch enumerateObjectsUsingBlock:^(MPSImage * _Nonnull seSourceImage, NSUInteger idx, BOOL * _Nonnull stop) {
//    for (int idx = 0; idx < [seSourceImageBatch count]; idx++) {
    int idx = 0;
        // Get underlying MTLTextures from MPSImages and pass in them into the encoder.
        [encoder setTexture:seSourceImageBatch[idx].texture atIndex:0];
        [encoder setTexture:convSourceImageBatch[idx].texture atIndex:1];
        [encoder setTexture:skipSourceImageBatch[idx].texture atIndex:2];
        [encoder setTexture:resultBatch[idx].texture atIndex:3];

        NSLog(@"Result Image: %@", resultBatch[idx]);
        NSLog(@"Result Texture: %@", resultBatch[idx].texture);
        
//        showRawImageContent(seSourceImage);
//        showRawTextureContent(seSourceImage.texture);
        
//    }];

    int batches = convImage.numberOfImages * [convSourceImageBatch count];
    MTLSize gridSize = MTLSizeMake(convImage.width * convImage.height, convImage.featureChannels, batches);
    int threadGroupWidth = _computePipeline.threadExecutionWidth;
    int threadGroupHeight = _computePipeline.maxTotalThreadsPerThreadgroup / threadGroupWidth;
    MTLSize threadGroupSize = MTLSizeMake(threadGroupWidth, threadGroupHeight, 1);
    
    NSLog(@"Thread info: threadgroup width: %i, height: %i, gridsize: x - %i, y - %i, z - %i", threadGroupWidth, threadGroupHeight, gridSize.width, gridSize.height, gridSize.depth);

    if (/*_nonuniformThreadgroups*/ true) {
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    } else {
        MTLSize threadGroupCount = MTLSizeMake((gridSize.width + threadGroupSize.width - 1) / threadGroupSize.width,
                                               (gridSize.height + threadGroupSize.height - 1) / threadGroupSize.height,
                                               batches);
        [encoder dispatchThreadgroups:threadGroupCount threadsPerThreadgroup:threadGroupSize];
    }

    [encoder endEncoding];
    
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    [blitEncoder synchronizeResource:[resultBatch[0] texture]];
    [blitEncoder endEncoding];
    
    // Ensure results are synced back to the CPU.
    //[resultBatch[0] synchronizeOnCommandBuffer: commandBuffer];
    
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
        // Print the information from the buffer.
        NSLog(@"Debug buffer:");
        listOfFloats((float *)[buffer contents], bufferSize);
    }];

    return resultBatch;
}

@end
