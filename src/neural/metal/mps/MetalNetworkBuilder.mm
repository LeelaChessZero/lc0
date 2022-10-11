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

#import "neural/network_legacy.h"
#import "MetalNetworkBuilder.h"
#import "NetworkGraph.h"

namespace lczero {
namespace metal_backend {

MetalNetworkBuilder::MetalNetworkBuilder(void){}
MetalNetworkBuilder::~MetalNetworkBuilder(void){}

//void MetalNetworkBuilder::init(void* weights, void* options)
std::string MetalNetworkBuilder::init(int gpu_id)
{
    // All metal devices.
    NSArray<id<MTLDevice>> * devices = MTLCopyAllDevices();

    if ([devices count] <= gpu_id) {
        // No GPU device matching ID.
        [NSException raise:@"Could not find device" format:@"Could not find a GPU or CPU compute device with specified id"];
        return "";
    }

    // Initialize the metal MPS Graph executor with the selected device.
    [Lc0NetworkGraph graphWithDevice:devices[gpu_id]
                              index:[NSNumber numberWithInt:gpu_id]];

    this->gpu_id = gpu_id;

    return std::string([devices[gpu_id].name UTF8String]);
}

void MetalNetworkBuilder::build(int kInputPlanes, int channelSize, int kernelSize, LegacyWeights& weights, bool attn_policy, bool conv_policy, bool wdl, bool moves_left, std::string default_activation)
{
    Lc0NetworkGraph * graph = [Lc0NetworkGraph getGraphAt:[NSNumber numberWithInt:this->gpu_id]];
    MPSGraphTensor * layer;
    NSString * defaultActivation = [NSString stringWithUTF8String:default_activation.c_str()];

    // 0. Input placeholder.
    layer = [graph inputPlaceholderWithInputChannels:kInputPlanes
                                              height:8
                                               width:8
                                               label:@"inputs"];

    // 1. Input layer
    layer = [graph addConvolutionBlockWithParent:layer
                                   inputChannels:kInputPlanes
                                  outputChannels:channelSize
                                      kernelSize:kernelSize
                                         weights:&weights.input.weights[0]
                                          biases:&weights.input.biases[0]
                                         activation:defaultActivation
                                           label:@"input/conv"];

    // 2. Residual blocks
    for (size_t i = 0; i < weights.residual.size(); i++) {
        layer = [graph addResidualBlockWithParent:layer
                                    inputChannels:channelSize
                                   outputChannels:channelSize
                                       kernelSize:kernelSize
                                         weights1:&weights.residual[i].conv1.weights[0]
                                          biases1:&weights.residual[i].conv1.biases[0]
                                         weights2:&weights.residual[i].conv2.weights[0]
                                          biases2:&weights.residual[i].conv2.biases[0]
                                            label:[NSString stringWithFormat:@"block_%zu", i]
                                            hasSe:weights.residual[i].has_se ? YES : NO
                                       seWeights1:&weights.residual[i].se.w1[0]
                                        seBiases1:&weights.residual[i].se.b1[0]
                                       seWeights2:&weights.residual[i].se.w2[0]
                                        seBiases2:&weights.residual[i].se.b2[0]
                                      seFcOutputs:weights.residual[i].se.b1.size()
                                       activation:defaultActivation];
    }

    // 3. Policy head.
    MPSGraphTensor * policy;
    if (attn_policy) {
        // 1. NCHW -> NHWC
        policy = [graph transposeChannelsWithTensor:layer label:@"policy/nchw_nhwc"];
        NSUInteger embeddingSize = weights.ip_pol_b.size();
        NSUInteger policyDModel = weights.ip2_pol_b.size();

        // 2. Square Embedding: Dense with SELU
        policy = [graph addFullyConnectedLayerWithParent:policy
                                          inputChannels:channelSize
                                         outputChannels:embeddingSize
                                                weights:&weights.ip_pol_w[0]
                                                 biases:&weights.ip_pol_b[0]
                                             activation:@"selu"
                                                  label:@"policy/fc_embed"];

        // 3. Encoder layers
        for (auto layer : weights.pol_encoder) {
            // TODO: support encoder heads.
            [NSException raise:@"Encoders not supported" format:@"Metal backend doesn't support encoder heads yet."];
        }

        // 4. Self-attention q and k.
        MPSGraphTensor * queries = [graph addFullyConnectedLayerWithParent:policy
                                                            inputChannels:embeddingSize
                                                           outputChannels:policyDModel
                                                                  weights:&weights.ip2_pol_w[0]
                                                                   biases:&weights.ip2_pol_b[0]
                                                               activation:nil
                                                                    label:@"policy/self_attention/q"];

        MPSGraphTensor * keys = [graph addFullyConnectedLayerWithParent:policy
                                                         inputChannels:embeddingSize
                                                        outputChannels:policyDModel
                                                               weights:&weights.ip3_pol_w[0]
                                                                biases:&weights.ip3_pol_b[0]
                                                            activation:nil
                                                                 label:@"policy/self_attention/k"];

        // 5. matmul(q,k) / sqrt(dk)
        policy = [graph scaledKQMatmulWithKeys:keys
                                   withQueries:queries
                                         scale:1.0f / sqrt(policyDModel)
                                         label:@"policy/self_attention/kq"];

        [graph setVariable:@"policy/self_attention/kq" tensor:policy];

        // 6. Slice last 8 keys (k[:, 56:, :]) and matmul with policy promotion weights, then concat to matmul_qk.
        policy = [graph attentionPolicyPromoMatmulConcatWithParent:policy
                                                          withKeys:keys
                                                           weights:&weights.ip4_pol_w[0]
                                                         inputSize:8
                                                        outputSize:4
                                                         sliceFrom:56
                                                       channelSize:policyDModel
                                                             label:@"policy/promo_logits"];
    }
    else if (conv_policy) {
        policy = [graph addConvolutionBlockWithParent:layer
                                        inputChannels:channelSize
                                       outputChannels:channelSize
                                           kernelSize:kernelSize
                                              weights:&weights.policy1.weights[0]
                                               biases:&weights.policy1.biases[0]
                                           activation:defaultActivation
                                                label:@"policy/conv1"];

        // No activation.
        policy = [graph addConvolutionBlockWithParent:policy
                                        inputChannels:channelSize
                                       outputChannels:80
                                           kernelSize:kernelSize
                                              weights:&weights.policy.weights[0]
                                               biases:&weights.policy.biases[0]
                                           activation:nil
                                                label:@"policy/conv2"];


      /**
       * @todo policy map implementation has bug in MPSGraph (GatherND not working in graph).
       * Implementation of policy map to be done in CPU for now.
       *
       * Reinstate this section when bug is fixed. See comments below.
       *
      // [1858 -> HWC or CHW]
      const bool HWC = false;
      std::vector<uint32_t> policy_map(1858);
      for (const auto& mapping : kConvPolicyMap) {
        if (mapping == -1) continue;
        const auto index = &mapping - kConvPolicyMap;
        const auto displacement = index / 64;
        const auto square = index % 64;
        const auto row = square / 8;
        const auto col = square % 8;
        if (HWC) {
          policy_map[mapping] = ((row * 8) + col) * 80 + displacement;
        } else {
          policy_map[mapping] = ((displacement * 8) + row) * 8 + col;
        }
      }
      policy = builder_->makePolicyMapLayer(policy, &policy_map[0], "policy_map");
      */
    }
    else {
        const int policySize = weights.policy.biases.size();

        policy = [graph addConvolutionBlockWithParent:layer
                                        inputChannels:channelSize
                                       outputChannels:policySize
                                           kernelSize:1
                                              weights:&weights.policy.weights[0]
                                               biases:&weights.policy.biases[0]
                                           activation:defaultActivation
                                                label:@"policy/conv"];

        policy = [graph addFullyConnectedLayerWithParent:policy
                                           inputChannels:policySize * 8 * 8
                                          outputChannels:1858
                                                 weights:&weights.ip_pol_w[0]
                                                  biases:&weights.ip_pol_b[0]
                                              activation:nil
                                                   label:@"policy/fc"];
    }

    // 4. Value head.
    MPSGraphTensor * value;
    value = [graph addConvolutionBlockWithParent:layer
                                   inputChannels:channelSize
                                  outputChannels:32
                                      kernelSize:1
                                         weights:&weights.value.weights[0]
                                          biases:&weights.value.biases[0]
                                      activation:defaultActivation
                                           label:@"value/conv"];

    value = [graph addFullyConnectedLayerWithParent:value
                                       inputChannels:32 * 8 * 8
                                      outputChannels:128
                                             weights:&weights.ip1_val_w[0]
                                              biases:&weights.ip1_val_b[0]
                                         activation:defaultActivation
                                               label:@"value/fc1"];

    if (wdl) {
        value = [graph addFullyConnectedLayerWithParent:value
                                          inputChannels:128
                                         outputChannels:3
                                                weights:&weights.ip2_val_w[0]
                                                 biases:&weights.ip2_val_b[0]
                                             activation:@"softmax"
                                                  label:@"value/fc2"];
    }
    else {
        value = [graph addFullyConnectedLayerWithParent:value
                                          inputChannels:128
                                         outputChannels:1
                                                weights:&weights.ip2_val_w[0]
                                                 biases:&weights.ip2_val_b[0]
                                             activation:@"tanh"
                                                  label:@"value/fc2"];
    }

    // 5. Moves left head.
    MPSGraphTensor * mlh;
    if (moves_left) {
        const int mlhChannels = weights.moves_left.biases.size();

        mlh = [graph addConvolutionBlockWithParent:layer
                                     inputChannels:channelSize
                                    outputChannels:mlhChannels
                                        kernelSize:1
                                           weights:&weights.moves_left.weights[0]
                                            biases:&weights.moves_left.biases[0]
                                        activation:defaultActivation
                                             label:@"mlh/conv"];

        mlh = [graph addFullyConnectedLayerWithParent:mlh
                                          inputChannels:mlhChannels * 8 * 8
                                         outputChannels:weights.ip1_mov_b.size()
                                                weights:&weights.ip1_mov_w[0]
                                                 biases:&weights.ip1_mov_b[0]
                                           activation:defaultActivation
                                                  label:@"mlh/fc1"];

        mlh = [graph addFullyConnectedLayerWithParent:mlh
                                          inputChannels:weights.ip1_mov_b.size()
                                         outputChannels:1
                                                weights:&weights.ip2_mov_w[0]
                                                 biases:&weights.ip2_mov_b[0]
                                             activation:@"relu"
                                                  label:@"mlh/fc2"];
    }

    // Select the outputs to be run through the inference graph.
    if (moves_left) {
        [graph setResultTensors:@[policy, value, mlh]];
    }
    else {
        [graph setResultTensors:@[policy, value]];
    }
}

void MetalNetworkBuilder::forwardEval(float * inputs, int batchSize, std::vector<float *> output_mems)
{
    @autoreleasepool {
        Lc0NetworkGraph * graph = [Lc0NetworkGraph getGraphAt:[NSNumber numberWithInt:this->gpu_id]];
        [graph runInferenceWithBatchSize:batchSize inputs:inputs outputs:&output_mems[0]];
    }
}

void MetalNetworkBuilder::saveVariables(std::vector<std::string> names)
{
    Lc0NetworkGraph * graph = [Lc0NetworkGraph getGraphAt:[NSNumber numberWithInt:this->gpu_id]];

    for (const std::string name : names) {
        [graph trackVariable:[NSString stringWithUTF8String:name.c_str()]];
    }
}

void MetalNetworkBuilder::dumpVariables(std::vector<std::string> names, int batches)
{
    Lc0NetworkGraph * graph = [Lc0NetworkGraph getGraphAt:[NSNumber numberWithInt:this->gpu_id]];

    for (const std::string name : names) {
        [graph dumpVariable:[NSString stringWithUTF8String:name.c_str()] batches:batches];
    }
}

}  // namespace metal_backend
}  // namespace lczero
