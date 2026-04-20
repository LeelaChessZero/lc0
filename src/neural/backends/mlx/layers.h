/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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

#include <mlx/mlx.h>

#include <span>
#include <string>
#include <vector>

#include "mlx_common.h"
#include "neural/network_legacy.h"

namespace lczero {
namespace mlx_backend {

namespace mx = mlx::core;

// Helper to create an MLX array from a float vector.
// If dtype is not float32, the array is converted to the specified dtype.
mx::array MakeArray(const std::vector<float>& data,
                    const std::vector<int>& shape,
                    mx::Dtype dtype = mx::float32);

// Convert convolution weights from OIHW to OHWI format.
// Creates a contiguous array with proper memory layout for MLX conv2d.
// If dtype is not float32, the array is converted to the specified dtype.
mx::array ConvertConvWeightsOIHWtoOHWI(const std::vector<float>& weights,
                                        int outChannels, int inChannels,
                                        int kH, int kW,
                                        mx::Dtype dtype = mx::float32);

// Expand input masks and values to [batch, 112, 8, 8] tensor.
// bit_tensor: pre-computed [1, 1, 64] uint64 array with {1<<0, ..., 1<<63}.
// zero_uint64: pre-computed mx::zeros({1}, mx::uint64) for comparison.
mx::array ExpandInput(const mx::array& masks, const mx::array& values,
                      int batch_size, const mx::array& bit_tensor,
                      const mx::array& zero_uint64);

// Apply activation function by name.
mx::array ApplyActivation(const mx::array& input,
                          const std::string& activation);

// Mish activation: x * tanh(softplus(x))
mx::array Mish(const mx::array& x);

// Swish activation: x * sigmoid(beta * x)
mx::array Swish(const mx::array& x, float beta = 1.0f);

// SELU activation
mx::array Selu(const mx::array& x);

// Convolution block: conv2d + bias + optional activation.
mx::array ConvBlock(const mx::array& input, const mx::array& weights,
                    const mx::array& biases, int kernel_size,
                    const std::string& activation);

// Residual block with optional SE unit.
mx::array ResidualBlock(const mx::array& input, const mx::array& conv1_weights,
                        const mx::array& conv1_biases,
                        const mx::array& conv2_weights,
                        const mx::array& conv2_biases, bool has_se,
                        const mx::array& se_fc1_weights,
                        const mx::array& se_fc1_biases,
                        const mx::array& se_fc2_weights,
                        const mx::array& se_fc2_biases,
                        const std::string& activation);

// SE (Squeeze-and-Excitation) unit.
// Input is NCHW format (conv2 output WITHOUT bias applied).
// conv_bias is the conv2 bias, applied specially in SE.
// skip is the residual skip connection (NCHW format).
mx::array SEUnit(const mx::array& input, const mx::array& conv_bias,
                 const mx::array& skip, const mx::array& fc1_weights,
                 const mx::array& fc1_biases, const mx::array& fc2_weights,
                 const mx::array& fc2_biases, const std::string& activation);

// Fully connected layer: matmul + optional bias + optional activation.
mx::array FullyConnected(const mx::array& input, const mx::array& weights,
                         const mx::array& biases,
                         const std::string& activation);

// Layer normalization using fused mx::fast::layer_norm kernel.
mx::array LayerNorm(const mx::array& input, const mx::array& gammas,
                    const mx::array& betas, float epsilon = 1e-6f);

// Layer normalization with scaled secondary tensor (skip connection).
mx::array LayerNormWithSkip(const mx::array& input, const mx::array& secondary,
                            const mx::array& gammas, const mx::array& betas,
                            float alpha, float epsilon = 1e-6f);

// Layer normalization with scaled secondary tensor (skip connection).
// Overload accepting alpha as a pre-computed mx::array (avoids creating from float each call).
mx::array LayerNormWithSkip(const mx::array& input, const mx::array& secondary,
                            const mx::array& gammas, const mx::array& betas,
                            const mx::array& alpha, float epsilon = 1e-6f);

// RMS normalization.
mx::array RmsNorm(const mx::array& input, const mx::array& gammas,
                  float epsilon = 1e-6f);

// RMS normalization with scaled secondary tensor (skip connection).
mx::array RmsNormWithSkip(const mx::array& input, const mx::array& secondary,
                          const mx::array& gammas, float alpha,
                          float epsilon = 1e-6f);

// Multi-head attention.
// smolgen_attn_weights: pre-computed [batch, heads, 64, 64] attention weights to add to Q@K^T.
// scale: pre-computed 1.0f / sqrt(depth) where depth = dmodel / heads.
mx::array MultiHeadAttention(const mx::array& queries, const mx::array& keys,
                             const mx::array& values, int heads, float scale,
                             const mx::array* smolgen_attn_weights = nullptr);

// Compute smolgen attention weights.
// Returns: [batch, heads, 64, 64] attention weights to be added to Q@K^T before softmax.
// Flow: input -> compress -> dense1+act+ln -> dense2+act+ln -> global -> reshape
mx::array ComputeSmolgen(const mx::array& input, int heads,
                         const mx::array& compress_w,
                         const mx::array& dense1_w, const mx::array& dense1_b,
                         const mx::array& ln1_gammas, const mx::array& ln1_betas,
                         const mx::array& dense2_w, const mx::array& dense2_b,
                         const mx::array& ln2_gammas, const mx::array& ln2_betas,
                         const mx::array& global_w,
                         const std::string& smolgen_activation,
                         float epsilon = 1e-3f);

// Compute smolgen attention weights with type-safe weight variants.
// Uses quantized FC when quantized weight is provided, matmul for float weights.
// WeightVariant ensures exactly one type is present (no null pointer issues).
mx::array ComputeSmolgenQuantized(
    const mx::array& input, int heads,
    const WeightVariant& compress_w,
    const WeightVariant& dense1_w,
    const mx::array& dense1_b,
    const mx::array& ln1_gammas, const mx::array& ln1_betas,
    const WeightVariant& dense2_w,
    const mx::array& dense2_b,
    const mx::array& ln2_gammas, const mx::array& ln2_betas,
    const WeightVariant& global_w,
    const std::string& smolgen_activation,
    float epsilon = 1e-3f);

// Quantize FC weights using MLX's quantize() function.
// Returns a QuantizedWeight struct containing packed weights, scales, and biases.
// Returns nullopt if weight dimensions are not compatible with group_size.
// weights: Float weights in row-major [input_size, output_size] format.
// group_size: Number of elements per quantization group (default 64).
// bits: Quantization bits (default 8 for int8).
std::optional<QuantizedWeight> QuantizeWeights(
    const mx::array& weights,
    int group_size = kDefaultQuantizationGroupSize,
    int bits = kDefaultQuantizationBits);

// Quantized fully connected layer using MLX's quantized_matmul().
// Uses int8 quantized weights with per-group scales and biases.
// input: Activations in float16 format [batch, ..., input_size].
// qw: Quantized weight struct from QuantizeWeights().
// biases: Optional biases to add after matmul.
// activation: Optional activation function name.
mx::array QuantizedFullyConnected(const mx::array& input,
                                  const QuantizedWeight& qw,
                                  const mx::array& biases,
                                  const std::string& activation);

// Scaled Q*K matmul for attention policy.
mx::array ScaledQKMatmul(const mx::array& queries, const mx::array& keys,
                         float scale);

// Scaled Q*K matmul with pre-computed scale array.
mx::array ScaledQKMatmul(const mx::array& queries, const mx::array& keys,
                         const mx::array& scale);

// Attention policy promotion matmul and concat.
mx::array AttentionPolicyPromoMatmulConcat(const mx::array& parent,
                                           const mx::array& keys,
                                           const mx::array& weights,
                                           int channel_size);

}  // namespace mlx_backend
}  // namespace lczero
