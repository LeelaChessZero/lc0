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
#import <vector>

#pragma mark - Base Kernel Class

@implementation Lc0Kernel

-(nonnull instancetype) initWithDevice:(nonnull id <MTLDevice>) device
                          functionName:(NSString *) functionName
{
    self = [super initWithDevice:device];
    if (nil == self) return nil;
    
    NSError *libraryError = nil;
    //    NSString *libraryFile = [[NSBundle mainBundle] pathForResource:@"Shaders.metallib" ofType:@"metallib"];
    _library = [device newLibraryWithFile:@"build/release/src/neural/metal/mps/Shaders.metallib" error:&libraryError];
    //_library = [device newLibraryWithFile:@"Shaders.metallib" error:&libraryError];
    
    //MTLFunctionConstantValues * constantValues = [MTLFunctionConstantValues alloc];
    //[constantValues setConstantValue:&_nonuniformThreadgroups type:MTLDataTypeBool atIndex:0];
    if (!_library) {
        NSLog(@"Failed to create kernel library, error: %@", libraryError);
        return nil;
    }
    
    // Create command encoder.
    NSError * kernelError = nil;
    // _kernelFunction = [_library newFunctionWithName:functionName constantValues:constantValues error:&kernelError];
    _kernelFunction = [_library newFunctionWithName:functionName];
    if (!_kernelFunction)
    {
        NSLog(@"Failed to create kernel function \"%@\", error %@", functionName, kernelError);
        return nil;
    }
    
    NSError * pipelineError = nil;
    _computePipeline = [device newComputePipelineStateWithFunction:_kernelFunction error:&pipelineError];
    if (!_computePipeline)
    {
        NSLog(@"Failed to create compute pipeline state, error %@", pipelineError);
        return nil;
    }
    
    return self;
}

-(nonnull id<MTLBuffer>) newGridInfoArgumentWithDevice:(id<MTLDevice>)device
                                 gridSize:(MTLSize)gridSize
                                  atIndex:(NSUInteger)index
{
    uint grid[3] = {gridSize.width, gridSize.height, gridSize.depth};
    id<MTLBuffer> argumentBuffer = [device newBufferWithBytes:(void *)&grid[0] length:3 * sizeof(uint) options:nil];
    
    return argumentBuffer;
}



@end


#pragma mark - SE Multiply and Add

@implementation Lc0SeMultiplyAdd {
    BOOL _hasRelu;
}

-(nonnull instancetype) initWithDevice:(nonnull id <MTLDevice>) device
                         gammaChannels:(NSUInteger) gammaChannels
                          betaChannels:(NSUInteger) betaChannels
                               hasRelu:(BOOL) hasRelu
{
    self = [super initWithDevice:device functionName:@"seMultiplyAdd"];
    if (nil == self) return nil;
    
    self.gammaChannels = gammaChannels;
    self.betaChannels = betaChannels;
    _hasRelu = hasRelu;
        
    return self;
}

-(nonnull MPSImageBatch *) encodeBatchToCommandBuffer:(id <MTLCommandBuffer> __nonnull) commandBuffer
                                       seSourceImages:(MPSImageBatch * __nonnull) seSourceImageBatch
                                     convSourceImages:(MPSImageBatch * __nonnull) convSourceImageBatch
                                     skipSourceImages:(MPSImageBatch * __nonnull) skipSourceImageBatch
{
    assert([seSourceImageBatch count] == 2 * [convSourceImageBatch count]);
    MPSImage * convImage = convSourceImageBatch[0];
    MPSImageDescriptor * resultDescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:convImage.featureChannelFormat
                                                                                           width:convImage.width
                                                                                          height:convImage.height
                                                                                 featureChannels:convImage.featureChannels
                                                                                  numberOfImages:convImage.numberOfImages
                                                                                           usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];
    resultDescriptor.storageMode = MTLStorageModeShared;
    
    MPSImageBatch * resultBatch = [self.destinationImageAllocator imageBatchForCommandBuffer:commandBuffer
                                                                             imageDescriptor:resultDescriptor
                                                                                      kernel:self
                                                                                       count:[convSourceImageBatch count]];
    
    int bufferSize = convImage.width * convImage.height * convImage.featureChannels * convImage.numberOfImages * [convSourceImageBatch count];
    id<MTLBuffer> buffer = [[commandBuffer device] newBufferWithLength:bufferSize * sizeof(float) options:nil];
    
    //NSLog(@"Buffer: %@", buffer);
    //NSLog(@"ImageBatch: %@", seSourceImageBatch);

    int batches = convImage.numberOfImages * [convSourceImageBatch count];
    MTLSize gridSize = MTLSizeMake(convImage.width * convImage.height, convImage.featureChannels / 4, batches);
    int threadGroupWidth = _computePipeline.threadExecutionWidth;
    int threadGroupHeight = _computePipeline.maxTotalThreadsPerThreadgroup / threadGroupWidth;
    MTLSize threadGroupSize = MTLSizeMake(threadGroupWidth, threadGroupHeight, 1);
    MTLSize threadGroupCount = MTLSizeMake((gridSize.width + threadGroupSize.width - 1) / threadGroupSize.width,
                                           (gridSize.height + threadGroupSize.height - 1) / threadGroupSize.height,
                                           batches);
    id<MTLBuffer> argumentBuffer = [self newGridInfoArgumentWithDevice:[commandBuffer device]
                                                              gridSize:gridSize
                                                               atIndex:0];
    
    // Encoding one batch at a time.
    // @todo Needs to be optimized, may require copying the images into a large buffer to allow for less loops.
    id<MTLComputeCommandEncoder> encoder;
    for (int idx = 0; idx < [convSourceImageBatch count]; idx++) {
        // Create new encoder to process this image.
        // @todo needs optimization.
        encoder = [commandBuffer computeCommandEncoder];
        //encoder.label = @"Encoder";
        // @todo Need to confirm if multiple images are encoded in same texture2d_array in metal.
        [encoder setComputePipelineState:_computePipeline];
        [encoder setBuffer:buffer offset:0 atIndex:0];
        [encoder setBuffer:argumentBuffer offset:0 atIndex:1];
        
        // Get underlying MTLTextures from MPSImages and pass in them into the encoder.
        [encoder setTexture:seSourceImageBatch[idx].texture atIndex:0];
        [encoder setTexture:convSourceImageBatch[idx].texture atIndex:1];
        [encoder setTexture:skipSourceImageBatch[idx].texture atIndex:2];
        [encoder setTexture:resultBatch[idx].texture atIndex:3];
        
        if (/*_nonuniformThreadgroups*/ true) {
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        } else {
            [encoder dispatchThreadgroups:threadGroupCount threadsPerThreadgroup:threadGroupSize];
        }
        [encoder endEncoding];
    }

    return resultBatch;
}

@end




#pragma mark - Flatten

@implementation Lc0Flatten

-(nonnull instancetype) initWithDevice:(nonnull id <MTLDevice>) device
{
    self = [super initWithDevice:device functionName:@"flatten"];
    if (nil == self) return nil;
    
    return self;
}

-(nonnull MPSImageBatch *) encodeBatchToCommandBuffer:(nonnull id <MTLCommandBuffer>) commandBuffer
                                         sourceImages:(MPSImageBatch * __nonnull) sourceImageBatch
{
    MPSImage * image = sourceImageBatch[0];
    MPSImageDescriptor * descriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:image.featureChannelFormat
                                                                                           width:1
                                                                                          height:1
                                                                                 featureChannels:image.width * image.height * image.featureChannels
                                                                                  numberOfImages:image.numberOfImages
                                                                                           usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];
    descriptor.storageMode = MTLStorageModeShared;
    MPSImageBatch * resultBatch = [self.destinationImageAllocator imageBatchForCommandBuffer:commandBuffer
                                                                             imageDescriptor:descriptor
                                                                                      kernel:self
                                                                                       count:[sourceImageBatch count]];
    
    int batches = image.numberOfImages * [sourceImageBatch count];
    MTLSize gridSize = MTLSizeMake(image.width * image.height / 4, image.featureChannels, batches);
    int threadGroupWidth = _computePipeline.threadExecutionWidth;
    int threadGroupHeight = _computePipeline.maxTotalThreadsPerThreadgroup / threadGroupWidth;
    MTLSize threadGroupSize = MTLSizeMake(threadGroupWidth, threadGroupHeight, 1);
    MTLSize threadGroupCount = MTLSizeMake((gridSize.width + threadGroupSize.width - 1) / threadGroupSize.width,
                                           (gridSize.height + threadGroupSize.height - 1) / threadGroupSize.height,
                                           batches);
    
    id<MTLBuffer> argumentBuffer = [self newGridInfoArgumentWithDevice:[commandBuffer device]
                                                              gridSize:gridSize
                                                               atIndex:0];

    // Encoding one batch at a time.
    // @todo Needs to be optimized, may require copying the images into a large buffer to allow for less loops.
    id<MTLComputeCommandEncoder> encoder;
    for (int idx = 0; idx < [sourceImageBatch count]; idx++) {
        // Create new encoder to process this image.
        // @todo needs optimization.
        encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:_computePipeline];
        [encoder setTexture:sourceImageBatch[idx].texture atIndex:0];
        [encoder setTexture:resultBatch[idx].texture atIndex:1];
        [encoder setBuffer:argumentBuffer offset:0 atIndex:1];

        if (/*_nonuniformThreadgroups*/ true) {
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        } else {
            [encoder dispatchThreadgroups:threadGroupCount threadsPerThreadgroup:threadGroupSize];
        }
        [encoder endEncoding];
    }
    
    return resultBatch;
}

@end


#pragma mark - Policy Map

#define POLICY_OUTPUT_SIZE 1858

@implementation Lc0PolicyMap {
    short * _policyMap;
}

-(nonnull instancetype) initWithDevice:(nonnull id <MTLDevice>) device
                             policyMap:(nonnull short *) policyMap
{
    self = [super initWithDevice:device functionName:@"policyMap"];
    if (nil == self) return nil;
    
    _policyMap = (short *)malloc(POLICY_OUTPUT_SIZE * sizeof(short));
    memcpy((void *)_policyMap, (void *)policyMap, POLICY_OUTPUT_SIZE * sizeof(short));

    return self;
}

-(nonnull MPSImageBatch *) encodeBatchToCommandBuffer:(nonnull id <MTLCommandBuffer>) commandBuffer
                                         sourceImages:(MPSImageBatch * __nonnull) sourceImageBatch
{
    const short policyOutputSize = POLICY_OUTPUT_SIZE;
    MPSImage * image = sourceImageBatch[0];
    MPSImageDescriptor * descriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:image.featureChannelFormat
                                                                                     width:1
                                                                                    height:1
                                                                           featureChannels:policyOutputSize
                                                                            numberOfImages:image.numberOfImages
                                                                                     usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];
    descriptor.storageMode = MTLStorageModeShared;
    MPSImageBatch * resultBatch = [self.destinationImageAllocator imageBatchForCommandBuffer:commandBuffer
                                                                             imageDescriptor:descriptor
                                                                                      kernel:self
                                                                                       count:[sourceImageBatch count]];
    
    int batches = image.numberOfImages * [sourceImageBatch count];
    MTLSize gridSize = MTLSizeMake((policyOutputSize + 3) / 4, batches, 1);
    int threadGroupWidth = _computePipeline.threadExecutionWidth;
    int threadGroupHeight = _computePipeline.maxTotalThreadsPerThreadgroup / threadGroupWidth;
    MTLSize threadGroupSize = MTLSizeMake(threadGroupWidth, threadGroupHeight, 1);
    MTLSize threadGroupCount = MTLSizeMake((gridSize.width + threadGroupSize.width - 1) / threadGroupSize.width,
                                           (gridSize.height + threadGroupSize.height - 1) / threadGroupSize.height,
                                           1);

    id<MTLBuffer> argumentBuffer = [self newGridInfoArgumentWithDevice:[commandBuffer device]
                                                              gridSize:gridSize
                                                               atIndex:0];

    id<MTLBuffer> policyMapBuffer = [[commandBuffer device] newBufferWithBytes:_policyMap
                                                                  length:policyOutputSize * sizeof(short)
                                                                 options:nil];
    
    // Encoding one batch at a time.
    // @todo Needs to be optimized, may require copying the images into a large buffer to allow for less loops.
    id<MTLComputeCommandEncoder> encoder;
    for (int idx = 0; idx < [sourceImageBatch count]; idx++) {
        // Create new encoder to process this image.
        // @todo needs optimization.
        encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:_computePipeline];
        [encoder setTexture:sourceImageBatch[idx].texture atIndex:0];
        [encoder setTexture:resultBatch[idx].texture atIndex:1];
        [encoder setBuffer:policyMapBuffer offset:0 atIndex:0];
        [encoder setBuffer:argumentBuffer offset:0 atIndex:1];

        if (/*_nonuniformThreadgroups*/ true) {
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        } else {
            [encoder dispatchThreadgroups:threadGroupCount threadsPerThreadgroup:threadGroupSize];
        }
        [encoder endEncoding];
    }
    
    return resultBatch;
}

@end
