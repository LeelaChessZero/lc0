/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2018 The LCZero Authors

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
 */

#include "neural/mkldnn/fully_connected_layer.h"
#include <mkldnn.h>

#include <algorithm>
#include <cassert>
#include <cmath>

namespace lczero {

void MkldnnFullyConnectedLayer::Forward1D(
    size_t batch_size, const size_t input_size, const size_t output_size,
    const float* inputs, const float* weights, const float* biases,
    bool apply_relu, float* outputs) {
  // more columns, matrix-matrix multiplication
  //
  //             C                     A                         B
  //
  //            outputs      :=       weights        x         inputs
  //
  //   cols:   batch_size (N)       input_size  (K)          batch_size (N)
  //
  //   rows  output_size (M)        output_size (M)         input_size (K)
  //
  mkldnn_sgemm('N', 'T', (int)batch_size, (int)output_size, (int)input_size,
               1.0f, inputs, (int)input_size, weights, (int)input_size, 0.0f,
               outputs, (int)output_size);

  if (apply_relu) {
    for (size_t i = 0; i < batch_size; i++) {
      float* batch_outputs = outputs + i * output_size;
      for (size_t o = 0; o < output_size; o++) {
        float val = biases[o] + batch_outputs[o];
        batch_outputs[o] = val >= 0 ? val : 0;
      }
    }
  } else {
    for (size_t i = 0; i < batch_size; i++) {
      float* batch_outputs = outputs + i * output_size;
      for (size_t o = 0; o < output_size; o++) {
        batch_outputs[o] += biases[o];
      }
    }
  }
}

float MkldnnFullyConnectedLayer::Forward0D(const size_t size, const float* x,
                                           const float* y) {
  // A scalar product, also known as a dot-product.
  float r = 0;
  for (size_t t = 0; t < size; t++) r += x[t] * y[t];
  return r;
}

}  // namespace lczero
