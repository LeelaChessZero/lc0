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
#import "NetworkGraph.h"

#include "Utilities.h"

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
    self = [[Lc0NetworkGraph alloc] initWithDevice:device commandQueue:commandQueue];
    
    NSLog(@"Initialized NN graph builder on device: %@", device);
}

void* MetalNetworkBuilder::makeConvolutionBlock(void * previousLayer, int inputSize, int channelSize, int kernelSize,
                                                float * weights, float * biases, bool withRelu, std::string label) {
    return [(id)self addConvolutionBlockWithParent:(Lc0GraphNode *)previousLayer
                                     inputChannels:inputSize
                                    outputChannels:channelSize
                                        kernelSize:kernelSize
                                           weights:weights
                                            biases:biases
                                           hasRelu:(BOOL)withRelu
                                             label:[NSString stringWithUTF8String:label.c_str()]];
}

void* MetalNetworkBuilder::makeResidualBlock(void * previousLayer, int inputSize, int channelSize, int kernelSize,
                                             float * weights1, float * biases1, float * weights2, float * biases2,
                                             bool withRelu, std::string label, bool withSe, int seFcOutputs,
                                             float * seWeights1, float * seBiases1, float * seWeights2, float * seBiases2) {

    return [(id)self addResidualBlockWithParent:(Lc0GraphNode *)previousLayer
                                  inputChannels:inputSize
                                 outputChannels:channelSize
                                     kernelSize:kernelSize
                                       weights1:weights1
                                        biases1:biases1
                                       weights2:weights2
                                        biases2:biases2
                                          label:[NSString stringWithUTF8String:label.c_str()]
                                          hasSe:withSe ? YES : NO
                                     seWeights1:seWeights1
                                      seBiases1:seBiases1
                                     seWeights2:seWeights2
                                      seBiases2:seBiases2
                                    seFcOutputs:seFcOutputs];

    /*NSString * nLabel = [NSString stringWithUTF8String:label.c_str()];
    
    ConvWeights *convWeights1 = [[ConvWeights alloc] initWithDevice:[(id)self getDevice]
                                                     inputChannels:inputSize
                                                    outputChannels:channelSize
                                                       kernelWidth:kernelSize
                                                      kernelHeight:kernelSize
                                                            stride:1
                                                           weights:weights1
                                                            biases:biases1
                                                             label:[NSString stringWithFormat:@"%s/conv1", label.c_str()]
                                                    fusedActivation:@"relu"];
    
    ConvWeights *convWeights2 = [[ConvWeights alloc] initWithDevice:[(id)self getDevice]
                                                      inputChannels:inputSize
                                                     outputChannels:channelSize
                                                        kernelWidth:kernelSize
                                                       kernelHeight:kernelSize
                                                             stride:1
                                                            weights:weights2
                                                             biases:biases2
                                                              label:[NSString stringWithFormat:@"%s/conv2", label.c_str()]
                                                    fusedActivation:withRelu ? @"relu" : nil];
    
    MPSNNImageNode * source;
    if (previousLayer == NULL) {
        // If no previous layer is provided, assume input layer and create placeholder input image node.
        source = [MPSNNImageNode nodeWithHandle:nil];
    }
    else {
        source = ((MPSNNFilterNode *)previousLayer).resultImage;
    }
    MPSNNFilterNode *layer = [(id)self residualBlockWithParent:previousLayer
                                                 inputChannels:channelSize
                                                outputChannels:channelSize
                                                   kernelWidth:kernelSize
                                                  kernelHeight:kernelSize
                                                      weights1:convWeights1
                                                      weights2:convWeights2
                                                     seWeights:NULL];
    return (void*) layer;*/
    return (void*) previousLayer;
}

void* MetalNetworkBuilder::makeFullyConnectedLayer(void * previousLayer, int inputSize, int outputSize,
                                                   float * weights, float * biases,
                                                   std::string activation, std::string label) {

    /*ConvWeights *convWeights = [[ConvWeights alloc] initWithDevice:[(id)self getDevice]
                                                      inputChannels:inputSize
                                                     outputChannels:outputSize
                                                        kernelWidth:1
                                                       kernelHeight:1
                                                             stride:1
                                                            weights:weights
                                                             biases:biases
                                                              label:[NSString stringWithUTF8String:label.c_str()]
                                                    fusedActivation:[NSString stringWithUTF8String:activation.c_str()]];
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
                                                          activation:@""];
    
    return (void*) layer;*/
    return (void*) previousLayer;
}

void* MetalNetworkBuilder::makeReshapeLayer(void * previousLayer, int resultWidth, int resultHeight, int resultChannels) {
    /*MPSNNImageNode * source;
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
    
    return (void*) reshape;*/
    return (void*) previousLayer;
}

void* MetalNetworkBuilder::makePolicyMapLayer(void * previousLayer, std::vector<short> * policyMap) {
    return (void*) previousLayer;
}

void* MetalNetworkBuilder::buildGraph(std::vector<void *> * outputs) {
//    NSArray<MPSNNImageNode *> * resultImages = @[];
//    BOOL resultsNeeded[outputs->size()];
//    int i = 0;
//
//    for (const auto& output : *outputs) {
//        resultImages = [resultImages arrayByAddingObject:((MPSNNFilterNode *)output).resultImage];
//        resultsNeeded[i++] = YES;
//    }
//    MPSNNGraph * graph = [(id)self buildGraphWithResultImages:resultImages
//                                             resultsAreNeeded:resultsNeeded];
//
//    NSLog(@"Completed building neural network graph: %@ on device: %@", graph, [(id)self getDevice]);
//
//    return (void*) graph;
}

std::vector<float*> MetalNetworkBuilder::forwardEval(uint64_t * masks, float * values, std::vector<float *> * outputs, int batchSize, int inputChannels)
{
    
    //NSLog(@"%@", ((Lc0NetworkGraph *)self)->inputConv);
    //[(Lc0NetworkGraph *)self)->inputConv encodeToCommandBuffer:commandBuffer sourceImage];

//    std::vector<float*> blah(1);
//    MPSImageBatch * result = [(id)self evalSingleConv:batchSize
//                                                masks:masks
//                                               values:values
//                                        inputChannels:inputChannels
//                                         subBatchSize:1];
//
//    // Conv layer.
//    int imgSz1 = result[0].featureChannels * result[0].height * result[0].width;
//    blah[0] = (float*)malloc(batchSize * imgSz1 * sizeof(float));
//    NSLog(@"batchSize[%i]: allocated for %@ (%i floats)", batchSize, @"Input Conv", batchSize * imgSz1);
//    updateResults(result, blah[0], batchSize, 1);
//    logImageResultInfo(result, @"Input Conv");
//    NSLog(@"%@", listOfFloats(blah[0], batchSize * imgSz1));

//    return blah;
    
    
    NSUInteger subBatchSize = MIN(1, batchSize);
    NSMutableArray<MPSImageBatch*> * results = [(id)self runInferenceWithBatchSize:batchSize
                                                                            masks:masks
                                                                           values:values
                                                                    inputChannels:inputChannels
                                                                     subBatchSize:subBatchSize];
//    NSLog(@"Result images (total of %i image batches)", [results count]);
//    int i = 0;
//    for (MPSImageBatch * batch: results) {
//        NSLog(@"Batch %i: %i images", i++, [batch count]);
//        for (MPSImage * image: batch) {
//            NSLog(@"result - W:%lu, H:%lu, C:%lu, N:%lu, precision:%lu, usage:%lu", image.width, image.height, image.featureChannels, image.numberOfImages, image.precision, image.usage);
//
//        }
//    }
    
    // Create temporary memory to pass results back to MCTS.
    std::vector<float*> output_mems([results count]);
//    NSLog(@"Return vector: size: %i", output_mems.size());
    
    int imgSz;
    NSArray<NSString *> *names = @[@"Extra", @"Policy", @"Value", @"MLH"];
    for (int rsIdx = 0; rsIdx < [results count]; rsIdx++) {
        imgSz = results[rsIdx][0].featureChannels * results[rsIdx][0].height * results[rsIdx][0].width;
        output_mems[rsIdx] = (float*)malloc(batchSize * imgSz * sizeof(float));
//        NSLog(@"batchSize[%i]: allocated for %@ (%i floats)", batchSize, names[rsIdx], batchSize * imgSz);
//        updateResults(results[rsIdx], output_mems[rsIdx], batchSize, subBatchSize);
//        logImageResultInfo(results[rsIdx], names[rsIdx]);
//        NSLog(@"%@", listOfFloats(output_mems[rsIdx], batchSize * imgSz));
//        listOfFloats(output_mems[rsIdx], batchSize * imgSz);
//        showRawImageContent(results[rsIdx][0]);
//        showRawTextureContent(results[rsIdx][0].texture);
    }
    
    return output_mems;
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
