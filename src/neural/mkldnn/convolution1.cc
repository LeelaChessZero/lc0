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

#include "neural/mkldnn/convolution1.h"
#include <mkldnn.h>

namespace lczero {

void MkldnnConvolution1::Forward(const size_t batch_size,
                                 const size_t input_channels,
                                 const size_t output_channels,
                                 const float* input, const float* weights,
                                 float* output) {
  for (size_t i = 0; i < batch_size; i++) {
    const float* batch_input = input + i * kSquares * input_channels;
    float* batch_output = output + i * kSquares * output_channels;

    mkldnn_sgemm('N', 'N', (int)output_channels, kSquares, (int)input_channels,
                 1.0f, weights, (int)input_channels, batch_input, kSquares,
                 0.0f, batch_output, kSquares);
  }
}

}  // namespace lczero
