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

void MetalNetworkBuilder::build(int kInputPlanes, MultiHeadWeights& weights, InputEmbedding embedding,
                                bool attn_body, bool attn_policy, bool conv_policy, bool wdl, bool moves_left,
                                Activations& activations, std::string& policy_head, std::string& value_head)
{
    Lc0NetworkGraph * graph = [Lc0NetworkGraph getGraphAt:[NSNumber numberWithInt:this->gpu_id]];
    NSString * defaultActivation = [NSString stringWithUTF8String:activations.default_activation.c_str()];
    NSString * smolgenActivation = [NSString stringWithUTF8String:activations.smolgen_activation.c_str()];
    NSString * ffnActivation = [NSString stringWithUTF8String:activations.ffn_activation.c_str()];
    NSString * policyHead = [NSString stringWithUTF8String:policy_head.c_str()];
    NSString * valueHead = [NSString stringWithUTF8String:value_head.c_str()];

    // 0. Input placeholder.
    // @todo - placeholder can be made directly as NHWC to avoid transposes.
    MPSGraphTensor * layer = [graph inputPlaceholderWithInputChannels:kInputPlanes
                                                               height:8
                                                                width:8
                                                                label:@"inputs"];

    const NSUInteger kernelSize = 3;
    const bool isPeDenseEmbedding = embedding == InputEmbedding::INPUT_EMBEDDING_PE_DENSE;

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
        layer = [graph transposeChannelsWithTensor:layer withShape:@[@(-1), @64, layer.shape[1]] label:@"input/nchw_nhwc"];

        // 2a. Input embedding for attention body.
        if (weights.residual.size() == 0) {
            // No residual means pure transformer, so process input position encoding.
            if (isPeDenseEmbedding) {
                // New input position encoding.
                layer = [graph dynamicPositionEncodingWithTensor:layer
                                                           width:weights.ip_emb_preproc_b.size() / 64
                                                         weights:&weights.ip_emb_preproc_w[0]
                                                          biases:&weights.ip_emb_preproc_b[0]
                                                           label:@"input/position_encoding"];
            }
            else {
                // Old input position encoding with map.
                layer = [graph positionEncodingWithTensor:layer
                                                withShape:@[@64, @64]
                                                  weights:&kPosEncoding[0][0]
                                                     type:nil
                                                    label:@"input/position_encoding"];
            }
        }

        // Embedding layer.
        layer = [graph addFullyConnectedLayerWithParent:layer
                                         outputChannels:weights.ip_emb_b.size()
                                                weights:&weights.ip_emb_w[0]
                                                 biases:&weights.ip_emb_b[0]
                                             activation:defaultActivation
                                                  label:@"input/embedding"];

        // Add layernorm for new nets.
        if (isPeDenseEmbedding) {
            layer = [graph addLayerNormalizationWithParent:layer
                                     scaledSecondaryTensor:nil
                                                    gammas:&weights.ip_emb_ln_gammas[0]
                                                     betas:&weights.ip_emb_ln_betas[0]
                                                     alpha:1.0
                                                   epsilon:1e-3
                                                     label:@"input/embedding/ln"];
        }

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

        float alpha = (float) pow(2.0 * weights.encoder.size(), -0.25);
        if (isPeDenseEmbedding) {
            // Input embedding feedforward network added for new multihead nets.
            MPSGraphTensor * ffn = [graph addFullyConnectedLayerWithParent:layer
                                                            outputChannels:weights.ip_emb_ffn.dense1_b.size()
                                                                   weights:&weights.ip_emb_ffn.dense1_w[0]
                                                                    biases:&weights.ip_emb_ffn.dense1_b[0]
                                                                activation:ffnActivation
                                                                     label:@"input/embedding/ffn/dense1"];

            ffn = [graph addFullyConnectedLayerWithParent:ffn
                                           outputChannels:weights.ip_emb_ffn.dense2_b.size()
                                                  weights:&weights.ip_emb_ffn.dense2_w[0]
                                                   biases:&weights.ip_emb_ffn.dense2_b[0]
                                               activation:nil
                                                    label:@"input/embedding/ffn/dense2"];

            // Skip connection + RMS Norm.
            layer = [graph addLayerNormalizationWithParent:layer
                                     scaledSecondaryTensor:ffn
                                                    gammas:&weights.ip_emb_ffn_ln_gammas[0]
                                                     betas:&weights.ip_emb_ffn_ln_betas[0]
                                                     alpha:alpha
                                                   epsilon:1e-3
                                                     label:@"input/embedding/ffn_ln"];
        }

        // 2b. Attention body encoder layers.
        for (size_t i = 0; i < weights.encoder.size(); i++) {
            layer = [graph addEncoderLayerWithParent:layer
                                       legacyWeights:weights.encoder[i]
                                               heads:weights.encoder_head_count
                                       embeddingSize:weights.ip_emb_b.size()
                                   smolgenActivation:smolgenActivation
                                       ffnActivation:ffnActivation
                                               alpha:alpha
                                             epsilon:isPeDenseEmbedding ? 1e-3 : 1e-6
                                            normtype:@"layernorm"
                                               label:[NSString stringWithFormat:@"encoder_%zu", i]];
        }
    }

    // 3. Policy head.
    MPSGraphTensor * policy;
    if (attn_policy && !attn_body) {
        // NCHW -> NHWC
        policy = [graph transposeChannelsWithTensor:layer withShape:@[@(-1), @64, layer.shape[1]] label:@"policy/nchw_nhwc"];
    }
    else {
        policy = layer;
    }

    policy = [graph makePolicyHeadWithTensor:policy
                             attentionPolicy:attn_policy
                           convolutionPolicy:conv_policy
                               attentionBody:attn_body
                           defaultActivation:defaultActivation
                           smolgenActivation:smolgenActivation
                               ffnActivation:ffnActivation
                                  policyHead:weights.policy_heads.at(policy_head)
                                       label:[NSString stringWithFormat:@"policy/%@", policyHead]];

    // 4. Value head.
    MPSGraphTensor * value = [graph makeValueHeadWithTensor:layer
                                              attentionBody:attn_body
                                                    wdlHead:wdl
                                          defaultActivation:defaultActivation
                                                  valueHead:weights.value_heads.at(value_head)
                                                      label:[NSString stringWithFormat:@"value/%@", valueHead]];

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
