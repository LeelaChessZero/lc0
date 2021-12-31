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

#import "MetalNetworkBuilder.h"
#import "NetworkExecutor.h"

namespace lczero {
namespace metal_backend {

MetalNetworkBuilder::MetalNetworkBuilder(void) : self(NULL) {}

MetalNetworkBuilder::~MetalNetworkBuilder(void)
{
    [(id)self dealloc];
}

//void MetalNetworkBuilder::init(void* weights, void* options)
void MetalNetworkBuilder::init()
{
    //redirectLogs();
    
    // All metal devices.
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    
    if ([devices count] < 1) {
        // No GPU device.
        [NSException raise:@"Could not find device" format:@"Could not find a GPU or CPU compute device"];
        return;
    }
    
    // Get the metal device and commandQueue to be used.
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == NULL) {
        // Fallback to first device if there is none.
        // @todo allow GPU to be specified via options.
        device = devices[0];
    }
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    // Initialize the metal MPS Graph executor with the device.
    self = [[Lc0NetworkExecutor alloc] initWithDevice:device commandQueue:commandQueue];
    
    NSLog(@"devices: %@", devices);
    NSLog(@"device: %@", device);
    NSLog(@"queue: %@", commandQueue);
    NSLog(@"self: %@", (id)self);
}

void describeWeights(float * weights, float * biases, int inputSize, int channelSize, int kernelSize) {
    float * p = weights;
    int lenWeights = inputSize * kernelSize * kernelSize * channelSize;
    //  for (int i=0; i<lenWeights; i++) {
    for (int i=2000; i<3000; i++) {
        NSLog(@"To Weight[%d]: 1) %f", i, *(p + i));
    }
    
    float * q = biases;
    for (int i=0; i<channelSize; i++) {
        NSLog(@"To Bias[%d]: 1) %f", i, *(q + i));
    }
    
}

void* MetalNetworkBuilder::makeConvolutionBlock(void * previousLayer, int inputSize, int channelSize, int kernelSize,
                                                float * weights, float * biases, bool withRelu, std::string label) {
    
    //describeWeights(weights, biases, inputSize, channelSize, kernelSize);
    
    ConvWeights *convWeights = [[ConvWeights alloc] initWithDevice:[(id)self getDevice]
                                                     inputChannels:inputSize
                                                    outputChannels:channelSize
                                                       kernelWidth:kernelSize
                                                      kernelHeight:kernelSize
                                                            stride:1
                                                           weights:weights
                                                            biases:biases
                                                             label:[NSString stringWithUTF8String:label.c_str()]];
    
    //[convWeights describeWeights];
    MPSNNImageNode * source;
    if (previousLayer == NULL) {
        // If no previous layer is provided, assume input layer and create placeholder input image node.
        source = [MPSNNImageNode nodeWithHandle:nil];
        NSLog(@"no previous layer");
    }
    else {
        source = ((MPSNNFilterNode *)previousLayer).resultImage;
        NSLog(@"previous layer provided");
    }
    
    MPSNNFilterNode *layer = [(id)self convolutionBlockWithSource:source
                                                inputChannels:inputSize
                                               outputChannels:channelSize
                                                  kernelWidth:kernelSize
                                                 kernelHeight:kernelSize
                                                      weights:convWeights
                                                      hasRelu:withRelu ? YES : NO];
    return (void*) layer;
}

void* MetalNetworkBuilder::makeResidualBlock(void * previousLayer, int inputSize, int channelSize, int kernelSize,
                                             float * weights1, float * biases1, float * weights2, float * biases2,
                                             float * seWeights1, float * seBiases1, float * seWeights2, float * seBiases2,
                                             bool withSe, bool withRelu, std::string label) {
    
    NSString * nLabel = [NSString stringWithUTF8String:label.c_str()];
    
    ConvWeights *convWeights1 = [[ConvWeights alloc] initWithDevice:[(id)self getDevice]
                                                     inputChannels:inputSize
                                                    outputChannels:channelSize
                                                       kernelWidth:kernelSize
                                                      kernelHeight:kernelSize
                                                            stride:1
                                                           weights:weights1
                                                            biases:biases1
                                                             label:[NSString stringWithFormat:@"%s/conv1", label.c_str()]];
    
    ConvWeights *convWeights2 = [[ConvWeights alloc] initWithDevice:[(id)self getDevice]
                                                      inputChannels:inputSize
                                                     outputChannels:channelSize
                                                        kernelWidth:kernelSize
                                                       kernelHeight:kernelSize
                                                             stride:1
                                                            weights:weights2
                                                             biases:biases2
                                                              label:[NSString stringWithFormat:@"%s/conv2", label.c_str()]];
    
    MPSNNImageNode * source;
    if (previousLayer == NULL) {
        // If no previous layer is provided, assume input layer and create placeholder input image node.
        source = [MPSNNImageNode nodeWithHandle:nil];
    }
    else {
        source = ((MPSNNFilterNode *)previousLayer).resultImage;
    }
    MPSNNFilterNode *layer = [(id)self residualBlockWithSource:source
                                                 inputChannels:channelSize
                                                outputChannels:channelSize
                                                   kernelWidth:kernelSize
                                                  kernelHeight:kernelSize
                                                      weights1:convWeights1
                                                      weights2:convWeights2
                                                     seWeights:NULL];
    return (void*) layer;
}

void* MetalNetworkBuilder::makeFullyConnectedLayer(void * previousLayer, int inputSize, int outputSize,
                                                   float * weights, float * biases,
                                                   std::string activation, std::string label) {

    ConvWeights *convWeights = [[ConvWeights alloc] initWithDevice:[(id)self getDevice]
                                                      inputChannels:inputSize
                                                     outputChannels:outputSize
                                                        kernelWidth:1
                                                       kernelHeight:1
                                                             stride:1
                                                            weights:weights
                                                             biases:biases
                                                              label:[NSString stringWithUTF8String:label.c_str()]];
    MPSNNImageNode * source;
    if (previousLayer == NULL) {
        // If no previous layer is provided, assume input layer and create placeholder input image node.
        source = [MPSNNImageNode nodeWithHandle:nil];
    }
    else {
        source = ((MPSNNFilterNode *)previousLayer).resultImage;
    }
    
    MPSNNFilterNode *layer = [(id)self fullyConnectedLayerWithSource:source
                                                             weights:convWeights
                                                          activation:activation ? [NSString stringWithUTF8String:activation.c_str()] : NULL];
    
    return (void*) layer;
}

void* MetalNetworkBuilder::makeReshapeLayer(void * previousLayer, int resultWidth, int resultHeight, int resultChannels) {
    MPSNNImageNode * source;
    if (previousLayer == NULL) {
        // If no previous layer is provided, assume input layer and create placeholder input image node.
        source = [MPSNNImageNode nodeWithHandle:nil];
    }
    else {
        source = ((MPSNNFilterNode *)previousLayer).resultImage;
    }
    MPSNNReshapeNode * reshape = [MPSNNReshapeNode nodeWithSource:source
                                                      resultWidth:resultWidth
                                                     resultHeight:resultHeight
                                            resultFeatureChannels:resultChannels];
    
    return (void*) reshape;
}

void MetalNetworkBuilder::forwardEval(void * io, int batchSize)
{
    MPSImageBatch * result = [(id)self runInferenceWithInputs:io batchSize:batchSize];
}

const char * __nonnull MetalNetworkBuilder::getTestData(char * data)
{
    const char * ret = [(id)self getTestData:data];
    
    NSLog(@"Test log: %d", 1000);
    NSLog(@"Value from objc: %s", ret);
    
    return ret;
}

void MetalNetworkBuilder::redirectLogs()
{
    NSArray *allPaths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [allPaths objectAtIndex:0];
    NSString *pathForLog = [documentsDirectory stringByAppendingPathComponent:@"lc0logfile.txt"];
    
    freopen([pathForLog cStringUsingEncoding:NSASCIIStringEncoding], "a+", stderr);
}

}  // namespace metal_backend
}  // namespace lczero
