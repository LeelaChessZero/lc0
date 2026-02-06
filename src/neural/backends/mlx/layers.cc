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

#include "layers.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <span>

#include "neural/tables/attention_policy_map.h"
#include "neural/tables/policy_map.h"

namespace lczero {
namespace mlx_backend {

namespace mx = mlx::core;

// Helper to create an MLX array from a float vector.
mx::array MakeArray(const std::vector<float>& data,
                    const std::vector<int>& shape, mx::Dtype dtype) {
  mx::Shape mlx_shape(shape.begin(), shape.end());
  mx::array result = mx::array(data.data(), mlx_shape, mx::float32);
  if (dtype != mx::float32) {
    result = mx::astype(result, dtype);
  }
  return result;
}

// Convert convolution weights from OIHW to OHWI format (MLX conv2d expects OHWI).
// This creates a contiguous array with the proper memory layout.
mx::array ConvertConvWeightsOIHWtoOHWI(const std::vector<float>& weights,
                                        int outChannels, int inChannels,
                                        int kH, int kW, mx::Dtype dtype) {
  // Validate inputs to prevent overflow and undefined behavior.
  assert(outChannels > 0 && inChannels > 0 && kH > 0 && kW > 0);
  assert(weights.size() ==
         static_cast<size_t>(outChannels) * static_cast<size_t>(inChannels) *
             static_cast<size_t>(kH) * static_cast<size_t>(kW));

  std::vector<float> converted(weights.size());
  for (int oc = 0; oc < outChannels; oc++) {
    for (int ic = 0; ic < inChannels; ic++) {
      for (int h = 0; h < kH; h++) {
        for (int w = 0; w < kW; w++) {
          size_t srcIdx = ((static_cast<size_t>(oc) * inChannels + ic) * kH + h) * kW + w;
          size_t dstIdx = ((static_cast<size_t>(oc) * kH + h) * kW + w) * inChannels + ic;
          converted[dstIdx] = weights[srcIdx];
        }
      }
    }
  }
  mx::Shape shape = {outChannels, kH, kW, inChannels};
  mx::array result = mx::array(converted.data(), shape, mx::float32);
  if (dtype != mx::float32) {
    result = mx::astype(result, dtype);
  }
  return result;
}

// Expand input masks and values to [batch, 112, 8, 8] tensor.
// bit_tensor: pre-computed [1, 1, 64] uint64 array with {1<<0, ..., 1<<63}.
// zero_uint64: pre-computed mx::zeros({1}, mx::uint64) for comparison.
mx::array ExpandInput(const mx::array& masks, const mx::array& values,
                      int batch_size, const mx::array& bit_tensor,
                      const mx::array& zero_uint64) {
  // Reshape masks to [batch, 112, 1] for broadcasting.
  mx::array mask_expanded = mx::reshape(masks, {batch_size, kInputPlanes, 1});

  // Broadcast masks to [batch, 112, 64].
  mask_expanded = mx::broadcast_to(mask_expanded, {batch_size, kInputPlanes, 64});

  // Bitwise AND with bit indices to extract individual bits.
  mx::array bits = mx::bitwise_and(mask_expanded, bit_tensor);

  // Compare with zero to get boolean mask.
  mx::array bit_mask = mx::not_equal(bits, zero_uint64);

  // Cast to float.
  bit_mask = mx::astype(bit_mask, mx::float32);

  // Reshape values to [batch, 112, 1] for broadcasting.
  mx::array val_expanded = mx::reshape(values, {batch_size, kInputPlanes, 1});

  // Broadcast values to [batch, 112, 64].
  val_expanded = mx::broadcast_to(val_expanded, {batch_size, kInputPlanes, 64});

  // Multiply mask with values.
  mx::array result = mx::multiply(bit_mask, val_expanded);

  // Reshape to [batch, 112, 8, 8] (NCHW format).
  return mx::reshape(result, {batch_size, kInputPlanes, 8, 8});
}

// Mish activation: x * tanh(softplus(x))
// Uses numerically stable formulation matching BLAS backend.
// mish(x) = x * n / (n + 2), where n = e^2 + 2e, e = exp(x)
// Computation is done in float32 for numerical stability.
mx::array Mish(const mx::array& x) {
  mx::Dtype orig_dtype = x.dtype();
  mx::array xf = (orig_dtype != mx::float32) ? mx::astype(x, mx::float32) : x;

  mx::array e = mx::exp(xf);
  mx::array n = mx::add(mx::square(e), mx::multiply(mx::array(2.0f), e));
  mx::array d = mx::divide(xf, mx::add(n, mx::array(2.0f)));

  // For val <= -0.125: return n * d
  // For val > -0.125: return val - 2 * d
  // Both give the same result: x * n / (n + 2)
  // But the second branch is more stable for small negative values.
  mx::array mask = mx::less_equal(xf, mx::array(-0.125f));
  mx::array result_low = mx::multiply(n, d);  // n * d
  mx::array result_high = mx::subtract(xf, mx::multiply(mx::array(2.0f), d));  // x - 2*d
  mx::array result = mx::where(mask, result_low, result_high);

  return (orig_dtype != mx::float32) ? mx::astype(result, orig_dtype) : result;
}

// Swish activation: x * sigmoid(beta * x)
mx::array Swish(const mx::array& x, float beta) {
  mx::array scaled = mx::multiply(x, mx::array(beta));
  return mx::multiply(x, mx::sigmoid(scaled));
}

// SELU activation
// Computation is done in float32 for numerical stability (exp).
mx::array Selu(const mx::array& x) {
  constexpr float alpha = 1.67326324f;
  constexpr float scale = 1.05070098f;

  mx::Dtype orig_dtype = x.dtype();
  mx::array xf = (orig_dtype != mx::float32) ? mx::astype(x, mx::float32) : x;

  // SELU: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
  mx::array positive = mx::maximum(xf, mx::array(0.0f));
  mx::array negative = mx::minimum(xf, mx::array(0.0f));
  mx::array exp_term = mx::multiply(
      mx::array(alpha),
      mx::subtract(mx::exp(negative), mx::array(1.0f)));

  mx::array result = mx::multiply(mx::array(scale), mx::add(positive, exp_term));
  return (orig_dtype != mx::float32) ? mx::astype(result, orig_dtype) : result;
}

// Apply activation function by name.
mx::array ApplyActivation(const mx::array& input,
                          const std::string& activation) {
  if (activation.empty() || activation == "none") {
    return input;
  } else if (activation == "relu") {
    return mx::maximum(input, mx::array(0.0f));
  } else if (activation == "relu_2") {
    mx::array relu_out = mx::maximum(input, mx::array(0.0f));
    return mx::multiply(relu_out, relu_out);
  } else if (activation == "tanh") {
    return mx::tanh(input);
  } else if (activation == "sigmoid") {
    return mx::sigmoid(input);
  } else if (activation == "selu") {
    return Selu(input);
  } else if (activation == "mish") {
    return Mish(input);
  } else if (activation == "swish") {
    return Swish(input);
  } else if (activation == "softmax") {
    // Softmax along the last dimension.
    return mx::softmax(input, -1);
  }
  return input;
}

// Convolution block: conv2d + bias + optional activation.
// MLX uses NHWC format by default, but we work with NCHW for compatibility.
// Weights must be pre-converted to OHWI format using ConvertConvWeightsOIHWtoOHWI().
mx::array ConvBlock(const mx::array& input, const mx::array& weights,
                    const mx::array& biases, int kernel_size,
                    const std::string& activation) {
  // Input is NCHW, MLX conv2d expects NHWC.
  // Transpose input from NCHW to NHWC.
  mx::array input_nhwc = mx::transpose(input, {0, 2, 3, 1});

  // Weights are already in OHWI format (pre-converted at load time).
  // Perform convolution with same padding.
  int pad = kernel_size / 2;
  mx::array conv_out = mx::conv2d(input_nhwc, weights, {1, 1}, {pad, pad});

  // Add bias (biases shape is [C], broadcasts with NHWC last dim).
  conv_out = mx::add(conv_out, biases);

  // Transpose output back to NCHW.
  mx::array output = mx::transpose(conv_out, {0, 3, 1, 2});

  return ApplyActivation(output, activation);
}

// SE (Squeeze-and-Excitation) unit.
// Matches BLAS implementation: conv_bias is applied specially.
mx::array SEUnit(const mx::array& input, const mx::array& conv_bias,
                 const mx::array& skip, const mx::array& fc1_weights,
                 const mx::array& fc1_biases, const mx::array& fc2_weights,
                 const mx::array& fc2_biases, const std::string& activation) {
  // Input is NCHW (conv2 output WITHOUT bias).
  int batch_size = static_cast<int>(input.shape()[0]);
  int channels = static_cast<int>(input.shape()[1]);

  // Global average pooling over H and W, then add conv_bias.
  // pool = mean(input) + conv_bias
  mx::array pooled = mx::mean(input, {2, 3});  // Shape: [batch, channels]
  pooled = mx::add(pooled, conv_bias);  // conv_bias shape is [channels]

  // FC1 with activation.
  // Weights are [input_size, output_size] = [channels, se_channels].
  mx::array fc1_out = mx::matmul(pooled, fc1_weights);
  fc1_out = mx::add(fc1_out, fc1_biases);
  fc1_out = ApplyActivation(fc1_out, activation);

  // FC2 (no activation before split).
  // Weights are [input_size, output_size] = [se_channels, 2*channels].
  mx::array fc2_out = mx::matmul(fc1_out, fc2_weights);
  fc2_out = mx::add(fc2_out, fc2_biases);

  // Split into gamma and beta.
  mx::array gamma = mx::slice(fc2_out, {0, 0}, {batch_size, channels});
  mx::array beta = mx::slice(fc2_out, {0, channels}, {batch_size, 2 * channels});

  // Apply sigmoid to gamma.
  gamma = mx::sigmoid(gamma);

  // beta = fc2[channels:] + gamma * conv_bias (BLAS-style)
  beta = mx::add(beta, mx::multiply(gamma, conv_bias));

  // Reshape for broadcasting: [batch, channels] -> [batch, channels, 1, 1].
  gamma = mx::reshape(gamma, {batch_size, channels, 1, 1});
  beta = mx::reshape(beta, {batch_size, channels, 1, 1});

  // SE output: input * gamma + beta + skip.
  mx::array se_out = mx::multiply(input, gamma);
  se_out = mx::add(se_out, beta);
  se_out = mx::add(se_out, skip);

  return ApplyActivation(se_out, activation);
}

// Residual block with optional SE unit.
mx::array ResidualBlock(const mx::array& input, const mx::array& conv1_weights,
                        const mx::array& conv1_biases,
                        const mx::array& conv2_weights,
                        const mx::array& conv2_biases, bool has_se,
                        const mx::array& se_fc1_weights,
                        const mx::array& se_fc1_biases,
                        const mx::array& se_fc2_weights,
                        const mx::array& se_fc2_biases,
                        const std::string& activation) {
  // First convolution with activation.
  mx::array conv1_out =
      ConvBlock(input, conv1_weights, conv1_biases, 3, activation);

  if (has_se) {
    // For SE: conv2 without bias (bias is handled specially in SEUnit).
    // Do conv2 manually without bias.
    mx::array input_nhwc = mx::transpose(conv1_out, {0, 2, 3, 1});
    mx::array conv2_out = mx::conv2d(input_nhwc, conv2_weights, {1, 1}, {1, 1});
    conv2_out = mx::transpose(conv2_out, {0, 3, 1, 2});  // Back to NCHW.

    // SE unit handles conv2_biases specially.
    return SEUnit(conv2_out, conv2_biases, input, se_fc1_weights, se_fc1_biases,
                  se_fc2_weights, se_fc2_biases, activation);
  } else {
    // Non-SE: conv2 with bias.
    mx::array conv2_out =
        ConvBlock(conv1_out, conv2_weights, conv2_biases, 3, "");

    // Simple residual connection with activation.
    mx::array residual = mx::add(input, conv2_out);
    return ApplyActivation(residual, activation);
  }
}

// Fully connected layer: matmul + optional bias + optional activation.
// Weights are stored as [input_size, output_size] (BLAS convention).
mx::array FullyConnected(const mx::array& input, const mx::array& weights,
                         const mx::array& biases,
                         const std::string& activation) {
  mx::array output = mx::matmul(input, weights);

  if (biases.size() > 0) {
    output = mx::add(output, biases);
  }

  return ApplyActivation(output, activation);
}

// Layer normalization (core implementation with pre-computed epsilon array).
// Mean/variance computation is done in float32 for numerical stability.
mx::array LayerNorm(const mx::array& input, const mx::array& gammas,
                    const mx::array& betas, const mx::array& epsilon) {
  mx::Dtype orig_dtype = input.dtype();

  // Upcast to float32 for mean/variance computation.
  mx::array inputf = (orig_dtype != mx::float32) ? mx::astype(input, mx::float32) : input;

  // Normalize along the last axis.
  int axis = static_cast<int>(input.ndim()) - 1;

  mx::array mean = mx::mean(inputf, std::vector<int>{axis}, true);
  mx::array variance = mx::var(inputf, std::vector<int>{axis}, true);

  mx::array normalized = mx::divide(
      mx::subtract(inputf, mean),
      mx::sqrt(mx::add(variance, epsilon)));

  // Apply gamma and beta (in float32).
  mx::array gammasf = (gammas.dtype() != mx::float32) ? mx::astype(gammas, mx::float32) : gammas;
  mx::array betasf = (betas.dtype() != mx::float32) ? mx::astype(betas, mx::float32) : betas;
  mx::array result = mx::add(mx::multiply(normalized, gammasf), betasf);

  return (orig_dtype != mx::float32) ? mx::astype(result, orig_dtype) : result;
}

// Layer normalization (float epsilon convenience overload).
mx::array LayerNorm(const mx::array& input, const mx::array& gammas,
                    const mx::array& betas, float epsilon) {
  return LayerNorm(input, gammas, betas, mx::array(epsilon));
}

// Layer normalization with scaled secondary tensor (skip connection).
mx::array LayerNormWithSkip(const mx::array& input, const mx::array& secondary,
                            const mx::array& gammas, const mx::array& betas,
                            float alpha, float epsilon) {
  // Add skip connection with optional scaling.
  mx::array combined = (alpha != 1.0f)
      ? mx::add(input, mx::multiply(secondary, mx::array(alpha)))
      : mx::add(input, secondary);

  return LayerNorm(combined, gammas, betas, epsilon);
}

// Overload accepting alpha as a pre-computed mx::array.
mx::array LayerNormWithSkip(const mx::array& input, const mx::array& secondary,
                            const mx::array& gammas, const mx::array& betas,
                            const mx::array& alpha, float epsilon) {
  mx::array combined = mx::add(input, mx::multiply(secondary, alpha));
  return LayerNorm(combined, gammas, betas, epsilon);
}

// LayerNormWithSkip with pre-computed epsilon array and float alpha.
mx::array LayerNormWithSkip(const mx::array& input, const mx::array& secondary,
                            const mx::array& gammas, const mx::array& betas,
                            float alpha, const mx::array& epsilon) {
  mx::array combined = (alpha != 1.0f)
      ? mx::add(input, mx::multiply(secondary, mx::array(alpha)))
      : mx::add(input, secondary);

  return LayerNorm(combined, gammas, betas, epsilon);
}

// LayerNormWithSkip with pre-computed epsilon array and mx::array alpha.
mx::array LayerNormWithSkip(const mx::array& input, const mx::array& secondary,
                            const mx::array& gammas, const mx::array& betas,
                            const mx::array& alpha, const mx::array& epsilon) {
  mx::array combined = mx::add(input, mx::multiply(secondary, alpha));
  return LayerNorm(combined, gammas, betas, epsilon);
}

// RMS normalization.
// Squared mean computation is done in float32 for numerical stability.
mx::array RmsNorm(const mx::array& input, const mx::array& gammas,
                  float epsilon) {
  mx::Dtype orig_dtype = input.dtype();

  // Upcast to float32 for squared mean computation.
  mx::array inputf = (orig_dtype != mx::float32) ? mx::astype(input, mx::float32) : input;

  int axis = static_cast<int>(input.ndim()) - 1;

  // RMS = sqrt(mean(x^2))
  mx::array squared = mx::multiply(inputf, inputf);
  mx::array mean_sq = mx::mean(squared, std::vector<int>{axis}, true);
  mx::array rms = mx::sqrt(mx::add(mean_sq, mx::array(epsilon)));

  // Normalize and apply gamma (in float32).
  mx::array gammasf = (gammas.dtype() != mx::float32) ? mx::astype(gammas, mx::float32) : gammas;
  mx::array result = mx::multiply(mx::divide(inputf, rms), gammasf);

  return (orig_dtype != mx::float32) ? mx::astype(result, orig_dtype) : result;
}

// RMS normalization with scaled secondary tensor (skip connection).
// Skip combination preserves input dtype; RmsNorm handles upcasting.
mx::array RmsNormWithSkip(const mx::array& input, const mx::array& secondary,
                          const mx::array& gammas, float alpha, float epsilon) {
  mx::array combined = (alpha != 1.0f)
      ? mx::add(input, mx::multiply(secondary, mx::array(alpha)))
      : mx::add(input, secondary);

  return RmsNorm(combined, gammas, epsilon);
}

// Compute smolgen attention weights.
// Flow: input -> compress -> dense1+act+ln -> dense2+act+ln -> global -> reshape
// Input: [batch, 64, embedding_size]
// Output: [batch, heads, 64, 64] attention weights to add to Q@K^T
mx::array ComputeSmolgen(const mx::array& input, int heads,
                         const mx::array& compress_w,
                         const mx::array& dense1_w, const mx::array& dense1_b,
                         const mx::array& ln1_gammas, const mx::array& ln1_betas,
                         const mx::array& dense2_w, const mx::array& dense2_b,
                         const mx::array& ln2_gammas, const mx::array& ln2_betas,
                         const mx::array& global_w,
                         const std::string& smolgen_activation,
                         const mx::array& epsilon) {
  int batch_size = static_cast<int>(input.shape()[0]);
  int seq_len = static_cast<int>(input.shape()[1]);     // 64
  int embed_size = static_cast<int>(input.shape()[2]);  // embedding_size
  int hidden = static_cast<int>(compress_w.shape()[1]); // hidden_channels

  // 1. Compress: [batch*64, embed] -> [batch*64, hidden]
  mx::array flat_input = mx::reshape(input, {batch_size * seq_len, embed_size});
  mx::array compressed = mx::matmul(flat_input, compress_w);  // No bias, no activation

  // 2. Reshape to [batch, 64*hidden] for dense1
  mx::array reshaped = mx::reshape(compressed, {batch_size, seq_len * hidden});

  // 3. Dense1 + activation + LayerNorm
  mx::array dense1 = mx::matmul(reshaped, dense1_w);
  dense1 = mx::add(dense1, dense1_b);
  dense1 = ApplyActivation(dense1, smolgen_activation);
  dense1 = LayerNorm(dense1, ln1_gammas, ln1_betas, epsilon);

  // 4. Dense2 + activation + LayerNorm
  mx::array dense2 = mx::matmul(dense1, dense2_w);
  dense2 = mx::add(dense2, dense2_b);
  dense2 = ApplyActivation(dense2, smolgen_activation);
  dense2 = LayerNorm(dense2, ln2_gammas, ln2_betas, epsilon);

  // 5. Global: reshape to [batch*heads, gen_sz_outputs/heads] and apply global weights
  int gen_sz_outputs = static_cast<int>(dense2.shape()[1]);
  int per_head_sz = gen_sz_outputs / heads;
  mx::array per_head = mx::reshape(dense2, {batch_size * heads, per_head_sz});
  mx::array global_out = mx::matmul(per_head, global_w);  // No bias, no activation

  // 6. Reshape to [batch, heads, 64, 64]
  mx::array result = mx::reshape(global_out, {batch_size, heads, seq_len, seq_len});

  return result;
}

// Compute smolgen attention weights with type-safe weight variants.
// Uses quantized FC when quantized weight is provided, matmul for float weights.
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
    const mx::array& epsilon) {
  int batch_size = static_cast<int>(input.shape()[0]);
  int seq_len = static_cast<int>(input.shape()[1]);     // 64
  int embed_size = static_cast<int>(input.shape()[2]);  // embedding_size

  // Get hidden size from weight variant.
  int hidden = std::visit(
      [](auto&& w) -> int {
        using T = std::decay_t<decltype(w)>;
        if constexpr (std::is_same_v<T, QuantizedWeight>) {
          return static_cast<int>(w.scales.shape()[0]);  // output_size from scales
        } else {
          return static_cast<int>(w.get().shape()[1]);
        }
      },
      compress_w);

  // Helper to compute FC/matmul based on weight type.
  auto compute_fc = [](const mx::array& x, const WeightVariant& weight,
                       const mx::array& bias,
                       const std::string& activation) -> mx::array {
    return std::visit(
        [&](auto&& w) -> mx::array {
          using T = std::decay_t<decltype(w)>;
          if constexpr (std::is_same_v<T, QuantizedWeight>) {
            return QuantizedFullyConnected(x, w, bias, activation);
          } else {
            if (bias.size() > 0 || !activation.empty()) {
              return FullyConnected(x, w.get(), bias, activation);
            } else {
              return mx::matmul(x, w.get());
            }
          }
        },
        weight);
  };

  // 1. Compress: [batch*64, embed] -> [batch*64, hidden]
  mx::array flat_input = mx::reshape(input, {batch_size * seq_len, embed_size});
  mx::array compressed = compute_fc(flat_input, compress_w, mx::array{}, "");

  // 2. Reshape to [batch, 64*hidden] for dense1
  mx::array reshaped = mx::reshape(compressed, {batch_size, seq_len * hidden});

  // 3. Dense1 + activation + LayerNorm
  mx::array dense1 =
      compute_fc(reshaped, dense1_w, dense1_b, smolgen_activation);
  dense1 = LayerNorm(dense1, ln1_gammas, ln1_betas, epsilon);

  // 4. Dense2 + activation + LayerNorm
  mx::array dense2 =
      compute_fc(dense1, dense2_w, dense2_b, smolgen_activation);
  dense2 = LayerNorm(dense2, ln2_gammas, ln2_betas, epsilon);

  // 5. Global: reshape to [batch*heads, gen_sz_outputs/heads] and apply global weights
  int gen_sz_outputs = static_cast<int>(dense2.shape()[1]);
  int per_head_sz = gen_sz_outputs / heads;
  mx::array per_head = mx::reshape(dense2, {batch_size * heads, per_head_sz});
  mx::array global_out = compute_fc(per_head, global_w, mx::array{}, "");

  // 6. Reshape to [batch, heads, 64, 64]
  mx::array result =
      mx::reshape(global_out, {batch_size, heads, seq_len, seq_len});

  return result;
}

// Multi-head attention.
// For smolgen path, softmax is computed in float32 for numerical stability.
// scale: pre-computed 1.0f / sqrt(depth) where depth = dmodel / heads.
mx::array MultiHeadAttention(const mx::array& queries, const mx::array& keys,
                             const mx::array& values, int heads, float scale,
                             const mx::array* smolgen_attn_weights) {
  // Input shape: [batch, 64, dmodel]
  int batch_size = static_cast<int>(queries.shape()[0]);
  int seq_len = static_cast<int>(queries.shape()[1]);
  int dmodel = static_cast<int>(queries.shape()[2]);
  int depth = dmodel / heads;
  mx::Dtype orig_dtype = queries.dtype();

  // Reshape to [batch, seq, heads, depth] and transpose to [batch, heads, seq, depth].
  mx::array q = mx::transpose(
      mx::reshape(queries, {batch_size, seq_len, heads, depth}), {0, 2, 1, 3});
  mx::array k = mx::transpose(
      mx::reshape(keys, {batch_size, seq_len, heads, depth}), {0, 2, 1, 3});
  mx::array v = mx::transpose(
      mx::reshape(values, {batch_size, seq_len, heads, depth}), {0, 2, 1, 3});

  mx::array output = [&]() {
    if (smolgen_attn_weights != nullptr && smolgen_attn_weights->size() > 0) {
      // Manual attention for smolgen (additive bias before softmax).
      // Upcast to float32 for softmax stability.
      mx::array qf = (orig_dtype != mx::float32) ? mx::astype(q, mx::float32) : q;
      mx::array kf = (orig_dtype != mx::float32) ? mx::astype(k, mx::float32) : k;
      mx::array vf = (orig_dtype != mx::float32) ? mx::astype(v, mx::float32) : v;
      mx::array smolgenf = (smolgen_attn_weights->dtype() != mx::float32)
          ? mx::astype(*smolgen_attn_weights, mx::float32) : *smolgen_attn_weights;

      mx::array attn = mx::matmul(qf, mx::transpose(kf, {0, 1, 3, 2}));
      attn = mx::multiply(attn, mx::array(scale));
      attn = mx::add(attn, smolgenf);
      attn = mx::softmax(attn, -1);
      mx::array result = mx::matmul(attn, vf);

      return (orig_dtype != mx::float32) ? mx::astype(result, orig_dtype) : result;
    } else {
      // Use optimized SDPA primitive.
      return mx::fast::scaled_dot_product_attention(q, k, v, scale);
    }
  }();

  // Transpose back and reshape to [batch, seq, dmodel].
  return mx::reshape(mx::transpose(output, {0, 2, 1, 3}),
                     {batch_size, seq_len, dmodel});
}

// Scaled Q*K matmul for attention policy (core implementation with array scale).
mx::array ScaledQKMatmul(const mx::array& queries, const mx::array& keys,
                         const mx::array& scale) {
  // Reshape to [batch, 64, channels].
  int batch_size = static_cast<int>(queries.shape()[0]);
  mx::array q = mx::reshape(queries, {batch_size, 64, -1});
  mx::array k = mx::reshape(keys, {batch_size, 64, -1});

  // Transpose keys: [batch, 64, channels] -> [batch, channels, 64].
  mx::array k_t = mx::transpose(k, {0, 2, 1});

  // Matmul: [batch, 64, channels] @ [batch, channels, 64] -> [batch, 64, 64].
  mx::array qk = mx::matmul(q, k_t);

  // Scale.
  return mx::multiply(qk, scale);
}

// Scaled Q*K matmul (float scale convenience overload).
mx::array ScaledQKMatmul(const mx::array& queries, const mx::array& keys,
                         float scale) {
  return ScaledQKMatmul(queries, keys, mx::array(scale));
}

// Attention policy promotion matmul and concat.
// weights: [channel_size, 4] (BLAS convention: [in, out])
// keys: [batch, 64, channel_size]
// Returns: parent concatenated with promotion logits.
mx::array AttentionPolicyPromoMatmulConcat(const mx::array& parent,
                                           const mx::array& keys,
                                           const mx::array& weights,
                                           int slice_from, int channel_size) {
  int batch_size = static_cast<int>(parent.shape()[0]);

  // Reshape keys to [batch, 64, channel_size].
  mx::array k = mx::reshape(keys, {batch_size, 64, channel_size});

  // Slice last 8 keys (from position 56 to 64).
  mx::array k_slice = mx::slice(k, {0, slice_from, 0}, {batch_size, 64, channel_size});
  // k_slice: [batch, 8, channel_size]

  // For batched matmul: k_slice @ weights = [batch, 8, channel_size] @ [channel_size, 4]
  // Result: [batch, 8, 4]
  mx::array promo = mx::matmul(k_slice, weights);

  // Transpose to [batch, 4, 8] for slicing.
  promo = mx::transpose(promo, {0, 2, 1});

  // Process promotion offsets.
  // promo shape is [batch, 4, 8]
  // offset1 = promo[:, :3, :]  (3 promotion types: queen, rook, bishop)
  // offset2 = promo[:, 3:4, :] (knight offset, broadcasted)
  mx::array offset1 = mx::slice(promo, {0, 0, 0}, {batch_size, 3, 8});
  mx::array offset2 = mx::slice(promo, {0, 3, 0}, {batch_size, 4, 8});
  mx::array promo_out = mx::add(offset1, offset2);
  // promo_out: [batch, 3, 8]

  // Broadcast to [batch, 3, 8, 8] - each of 8 source files, 8 destination files.
  promo_out = mx::reshape(promo_out, {batch_size, 3, 1, 8});
  promo_out = mx::broadcast_to(promo_out, {batch_size, 3, 8, 8});

  // Reshape to [batch, 3, 64].
  promo_out = mx::reshape(promo_out, {batch_size, 3, 64});

  // Get the relevant slice from parent (rows 48:56, cols 56:64).
  // parent is [batch, 64, 64] = [batch, from_square, to_square].
  // We want from_square 48:56 (rank 7), to_square 56:64 (rank 8).
  mx::array parent_reshaped = mx::reshape(parent, {batch_size, 64, 64});
  mx::array parent_slice = mx::slice(parent_reshaped, {0, 48, 56}, {batch_size, 56, 64});
  // parent_slice: [batch, 8, 8]
  parent_slice = mx::reshape(parent_slice, {batch_size, 1, 64});
  parent_slice = mx::broadcast_to(parent_slice, {batch_size, 3, 64});

  // Add parent slice to promo.
  promo_out = mx::add(promo_out, parent_slice);

  // Concat with parent: [batch, 64, 64] + [batch, 3, 64] -> [batch, 67, 64].
  // Then reshape to [batch, 67*64] later.
  return mx::concatenate({parent_reshaped, promo_out}, 1);
}

// Policy map layer - maps raw policy to 1858 outputs.
// This is done on CPU as MLX may have issues with gather operations.
void ApplyPolicyMap(std::span<const float> input_data,
                    std::span<float> output_data,
                    std::span<const short> policy_map, size_t input_stride) {
  assert(!input_data.empty());
  assert(!output_data.empty());
  assert(policy_map.size() <= input_stride);

  // Derive batch_size from output buffer size.
  const size_t batch_size = output_data.size() / kNumOutputPolicy;
  assert(output_data.size() == batch_size * kNumOutputPolicy);
  assert(input_data.size() >= batch_size * input_stride);

  // For each batch element, remap policy values.
  for (size_t b = 0; b < batch_size; b++) {
    auto batch_output = output_data.subspan(b * kNumOutputPolicy, kNumOutputPolicy);
    auto batch_input = input_data.subspan(b * input_stride, policy_map.size());

    // Zero-initialize output.
    std::fill(batch_output.begin(), batch_output.end(), 0.0f);

    // Apply mapping: for each input index, write to corresponding output index.
    for (size_t i = 0; i < policy_map.size(); i++) {
      short j = policy_map[i];
      if (j >= 0 && static_cast<size_t>(j) < kNumOutputPolicy) {
        batch_output[j] = batch_input[i];
      }
    }
  }
}

// Gating layer (multiply or add with learned weights).
// Weights are stored as [embedding_size, 64] (BLAS convention).
// Input is [batch, 64, embedding_size].
// For element-wise ops, transpose weights to [64, embedding_size].
mx::array GatingLayer(const mx::array& input, const mx::array& weights,
                      const std::string& operation) {
  // Transpose from [embedding_size, 64] to [64, embedding_size] for broadcasting.
  mx::array weights_t = mx::transpose(weights);

  if (operation == "mult") {
    return mx::multiply(input, weights_t);
  } else if (operation == "add") {
    return mx::add(input, weights_t);
  }
  return input;
}

// Quantize FC weights using MLX's quantize() function.
// MLX quantize() expects weights in [output_size, input_size] format and
// returns [packed, scales, biases].
// Returns nullopt if the weight dimensions are not compatible with the group_size.
std::optional<QuantizedWeight> QuantizeWeights(const mx::array& weights,
                                               int group_size, int bits) {
  // MLX quantize expects weights transposed: [output_size, input_size].
  // Our weights are stored as [input_size, output_size], so transpose first.
  mx::array w_transposed = mx::transpose(weights);

  // Check if the last dimension (input_size after transpose = output_size before)
  // is divisible by group_size. MLX requires this.
  int last_dim = static_cast<int>(w_transposed.shape().back());
  if (last_dim % group_size != 0) {
    // Weight dimensions not compatible with group_size - skip quantization.
    return std::nullopt;
  }

  // Quantize using affine mode (scale + bias per group).
  auto quantized = mx::quantize(w_transposed, group_size, bits, "affine");

  // quantize() returns a vector of 3 arrays: [packed, scales, biases].
  assert(quantized.size() == 3);

  return QuantizedWeight(std::move(quantized[0]), std::move(quantized[1]),
                         std::move(quantized[2]), group_size, bits);
}

// Quantized fully connected layer using MLX's quantized_matmul().
// quantized_matmul expects: x @ w^T where w is quantized (transposed weights).
mx::array QuantizedFullyConnected(const mx::array& input,
                                  const QuantizedWeight& qw,
                                  const mx::array& biases,
                                  const std::string& activation) {
  // Ensure input is float16 for quantized matmul efficiency.
  // Quantized matmul requires float16 input for efficiency.
  mx::array x = (input.dtype() != mx::float16) ? mx::astype(input, mx::float16)
                                               : input;

  // quantized_matmul: x @ w^T where w is the packed quantized weight.
  // transpose=true (default) means the weight is transposed internally.
  mx::array output = mx::quantized_matmul(x, qw.packed, qw.scales, qw.biases,
                                          /*transpose=*/true, qw.group_size,
                                          qw.bits);

  if (biases.size() > 0) {
    // Biases should be float16 for addition.
    mx::array b =
        (biases.dtype() != mx::float16) ? mx::astype(biases, mx::float16) : biases;
    output = mx::add(output, b);
  }

  return ApplyActivation(output, activation);
}

}  // namespace mlx_backend
}  // namespace lczero
