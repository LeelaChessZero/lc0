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
#import "neural/shared/attention_policy_map.h"
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

void MetalNetworkBuilder::build(int kInputPlanes, LegacyWeights& weights, bool attn_body, bool attn_policy, bool conv_policy, bool wdl, bool moves_left, Activations activations)
{
    Lc0NetworkGraph * graph = [Lc0NetworkGraph getGraphAt:[NSNumber numberWithInt:this->gpu_id]];
    NSString * defaultActivation = [NSString stringWithUTF8String:activations.default_activation.c_str()];
    NSString * smolgenActivation = [NSString stringWithUTF8String:activations.smolgen_activation.c_str()];
    NSString * ffnActivation = [NSString stringWithUTF8String:activations.ffn_activation.c_str()];

    // 0. Input placeholder.
    // @todo - placeholder can be made directly as NHWC to avoid transposes.
    MPSGraphTensor * layer = [graph inputPlaceholderWithInputChannels:kInputPlanes
                                                               height:8
                                                                width:8
                                                                label:@"inputs"];
    
    const NSUInteger kernelSize = 3;

    // Initialize global smolgen weights.
    if (weights.has_smolgen) {
        [graph setGlobalSmolgenWeights:&weights.smolgen_w[0]];
    }

    // Input conv layer only when there are residual blocks.
    if (weights.residual.size() > 0) {

        const NSUInteger channelSize = weights.input.weights.size() / (kInputPlanes * kernelSize * kernelSize);

        // 1. Input layer
        layer = [graph addConvolutionBlockWithParent:layer
                                      outputChannels:channelSize
                                          kernelSize:kernelSize
                                             weights:&weights.input.weights[0]
                                              biases:&weights.input.biases[0]
                                             activation:defaultActivation
                                               label:@"input/conv"];

        // 2. Residual blocks
        for (size_t i = 0; i < weights.residual.size(); i++) {
            layer = [graph addResidualBlockWithParent:layer
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

    }

    // Attention body.
    if (attn_body) {

        assert(weights.ip_emb_b.size() > 0);

        // 1. NCHW -> NHWC
        // @todo if input placeholder is NHWC, then this is not needed for attn_body, but kPosEncoding has to be reordered.
        layer = [graph transposeChannelsWithTensor:layer withShape:@[@(-1), @64, layer.shape[1]] label:@"input/nchw_nhwc"];

        if (weights.residual.size() == 0) {
            // No residual means pure transformer, so process input position encoding.
            layer = [graph positionEncodingWithTensor:layer
                                            withShape:@[@64, @64]
                                              weights:&kPosEncoding[0][0]
                                                 type:nil
                                                label:@"input/position_encoding"];
        }

        // 2a. Input embedding for attention body.
        // if self.arc_encoding: @todo needs to be implemented
        layer = [graph addFullyConnectedLayerWithParent:layer
                                          outputChannels:weights.ip_emb_b.size()
                                                 weights:&weights.ip_emb_w[0]
                                                  biases:&weights.ip_emb_b[0]
                                              activation:defaultActivation
                                                   label:@"input/embedding"];

        // # !!! input gate
        // flow = ma_gating(flow, name=name+'embedding')
        // def ma_gating(inputs, name):
        //     out = Gating(name=name+'/mult_gate', additive=False)(inputs)
        //     out = Gating(name=name+'/add_gate', additive=True)(out)
        if (weights.ip_mult_gate.size() > 0) {
            layer = [graph addGatingLayerWithParent:layer
                                            weights:&weights.ip_mult_gate[0]
                                      withOperation:@"mult"
                                              label:@"input/mult_gate"];
        }
        if (weights.ip_add_gate.size() > 0) {
            layer = [graph addGatingLayerWithParent:layer
                                            weights:&weights.ip_add_gate[0]
                                      withOperation:@"add"
                                              label:@"input/add_gate"];
        }
        // 2b. Attention body encoder layers.
        float alpha = (float) pow(2.0 * weights.encoder.size(), 0.25);
        for (size_t i = 0; i < weights.encoder.size(); i++) {
            layer = [graph addEncoderLayerWithParent:layer
                                       legacyWeights:weights.encoder[i]
                                               heads:weights.encoder_head_count
                                       embeddingSize:weights.ip_emb_b.size()
                                   smolgenActivation:smolgenActivation
                                       ffnActivation:ffnActivation
                                               alpha:alpha
                                               label:[NSString stringWithFormat:@"encoder_%zu", i]];
        }
    }

    // 3. Policy head.
    MPSGraphTensor * policy;
    if (attn_policy) {
        // 1. NCHW -> NHWC
        if (!attn_body) {
            policy = [graph transposeChannelsWithTensor:layer withShape:@[@(-1), @64, layer.shape[1]] label:@"policy/nchw_nhwc"];
        }
        else {
            policy = layer;
        }

        // 2. Square Embedding: Dense with default activation (or SELU for old ap-mish nets).
        NSUInteger embeddingSize = weights.ip_pol_b.size();
        NSUInteger policyDModel = weights.ip2_pol_b.size();
        // ap-mish uses hardcoded SELU
        policy = [graph addFullyConnectedLayerWithParent:policy
                                         outputChannels:embeddingSize
                                                weights:&weights.ip_pol_w[0]
                                                 biases:&weights.ip_pol_b[0]
                                             activation:attn_body ? defaultActivation : @"selu"
                                                  label:@"policy/fc_embed"];

        // 3. Encoder layers
        for (NSUInteger i = 0; i < weights.pol_encoder.size(); i++) {
            policy = [graph addEncoderLayerWithParent:policy
                                        legacyWeights:weights.pol_encoder[i]
                                                heads:weights.pol_encoder_head_count
                                        embeddingSize:embeddingSize
                                    smolgenActivation:attn_body ? smolgenActivation : nil
                                        ffnActivation:attn_body ? ffnActivation : @"selu"
                                                alpha:1.0
                                                label:[NSString stringWithFormat:@"policy/encoder_%zu", i]];
        }

        // 4. Self-attention q and k.
        MPSGraphTensor * queries = [graph addFullyConnectedLayerWithParent:policy
                                                           outputChannels:policyDModel
                                                                  weights:&weights.ip2_pol_w[0]
                                                                   biases:&weights.ip2_pol_b[0]
                                                               activation:nil
                                                                    label:@"policy/self_attention/q"];

        MPSGraphTensor * keys = [graph addFullyConnectedLayerWithParent:policy
                                                        outputChannels:policyDModel
                                                               weights:&weights.ip3_pol_w[0]
                                                                biases:&weights.ip3_pol_b[0]
                                                            activation:nil
                                                                 label:@"policy/self_attention/k"];

        // 5. matmul(q,k) / sqrt(dk)
        policy = [graph scaledQKMatmulWithQueries:queries
                                         withKeys:keys
                                            scale:1.0f / sqrt(policyDModel)
                                            label:@"policy/self_attention/kq"];

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
        if (attn_body) {
            [NSException raise:@"Unsupported architecture."
                        format:@"Convolutional policy not supported with attention body."];
        }
        policy = [graph addConvolutionBlockWithParent:layer
                                       outputChannels:weights.policy1.biases.size()
                                           kernelSize:kernelSize
                                              weights:&weights.policy1.weights[0]
                                               biases:&weights.policy1.biases[0]
                                           activation:defaultActivation
                                                label:@"policy/conv1"];

        // No activation.
        policy = [graph addConvolutionBlockWithParent:policy
                                       outputChannels:weights.policy.biases.size()
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
        if (attn_body) {
            [NSException raise:@"Unsupported architecture."
                        format:@"Classical policy not supported with attention body."];
        }

        const int policySize = weights.policy.biases.size();

        policy = [graph addConvolutionBlockWithParent:layer
                                       outputChannels:policySize
                                           kernelSize:1
                                              weights:&weights.policy.weights[0]
                                               biases:&weights.policy.biases[0]
                                           activation:defaultActivation
                                                label:@"policy/conv"];

        policy = [graph flatten2DTensor:policy
                                   axis:1
                                   name:@"policy/conv/flatten"];

        policy = [graph addFullyConnectedLayerWithParent:policy
                                          outputChannels:weights.ip_pol_b.size()
                                                 weights:&weights.ip_pol_w[0]
                                                  biases:&weights.ip_pol_b[0]
                                              activation:nil
                                                   label:@"policy/fc"];
    }

    // 4. Value head.
    MPSGraphTensor * value;
    if (attn_body) {
        value = [graph addFullyConnectedLayerWithParent:layer
                                         outputChannels:weights.ip_val_b.size()
                                                weights:&weights.ip_val_w[0]
                                                 biases:&weights.ip_val_b[0]
                                             activation:defaultActivation
                                                  label:@"value/embedding"];
    }
    else {
        value = [graph addConvolutionBlockWithParent:layer
                                      outputChannels:weights.value.biases.size()
                                          kernelSize:1
                                             weights:&weights.value.weights[0]
                                              biases:&weights.value.biases[0]
                                          activation:defaultActivation
                                               label:@"value/conv"];
    }

    value = [graph flatten2DTensor:value
                              axis:1
                              name:@"value/flatten"];

    value = [graph addFullyConnectedLayerWithParent:value
                                      outputChannels:weights.ip1_val_b.size()
                                             weights:&weights.ip1_val_w[0]
                                              biases:&weights.ip1_val_b[0]
                                         activation:defaultActivation
                                               label:@"value/fc1"];

    if (wdl) {
        value = [graph addFullyConnectedLayerWithParent:value
                                         outputChannels:weights.ip2_val_b.size()
                                                weights:&weights.ip2_val_w[0]
                                                 biases:&weights.ip2_val_b[0]
                                             activation:@"softmax"
                                                  label:@"value/fc2"];
    }
    else {
        value = [graph addFullyConnectedLayerWithParent:value
                                         outputChannels:weights.ip2_val_b.size()
                                                weights:&weights.ip2_val_w[0]
                                                 biases:&weights.ip2_val_b[0]
                                             activation:@"tanh"
                                                  label:@"value/fc2"];
    }

    // 5. Moves left head.
    MPSGraphTensor * mlh;
    if (moves_left) {
        if (attn_body) {
            mlh = [graph addFullyConnectedLayerWithParent:layer
                                           outputChannels:weights.ip_mov_b.size()
                                                  weights:&weights.ip_mov_w[0]
                                                   biases:&weights.ip_mov_b[0]
                                               activation:defaultActivation
                                                    label:@"moves_left/embedding"];
        }
        else {
            mlh = [graph addConvolutionBlockWithParent:layer
                                        outputChannels:weights.moves_left.biases.size()
                                            kernelSize:1
                                               weights:&weights.moves_left.weights[0]
                                                biases:&weights.moves_left.biases[0]
                                            activation:defaultActivation
                                                 label:@"moves_left/conv"];
        }

        mlh = [graph flatten2DTensor:mlh
                                axis:1
                                name:@"moves_left/flatten"];

        mlh = [graph addFullyConnectedLayerWithParent:mlh
                                         outputChannels:weights.ip1_mov_b.size()
                                                weights:&weights.ip1_mov_w[0]
                                                 biases:&weights.ip1_mov_b[0]
                                           activation:defaultActivation
                                                  label:@"moves_left/fc1"];

        mlh = [graph addFullyConnectedLayerWithParent:mlh
                                         outputChannels:weights.ip2_mov_b.size()
                                                weights:&weights.ip2_mov_w[0]
                                                 biases:&weights.ip2_mov_b[0]
                                             activation:@"relu"
                                                  label:@"moves_left/fc2"];
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

}  // namespace metal_backend
}  // namespace lczero
