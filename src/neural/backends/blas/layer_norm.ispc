/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2023 The LCZero Authors

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

export void LayerNorm2DWithSkipConnection(uniform const size_t channels,
                                          uniform float data[],
                                          const uniform float alpha,
                                          const uniform float skip[],
                                          const uniform float gammas[],
                                          const uniform float betas[],
                                          uniform const float epsilon) {
#if 0
    // Faster but potentially less stable version for future testing.
    // One pass mean and variance taken in dimension C. Uses shifted variance calculation.
    float imean = 0;
    float ivar = 0;
    float k = data[0] * alpha;
    if (skip != NULL) {
      k += skip[0];
      foreach (c = 0 ... channels) {
        float t = data[c] * alpha + skip[c];
        data[c] = t;
        t -= k;
        imean += t;
        ivar += t * t;
      }
    } else {
      foreach (c = 0 ... channels) {
        float t = data[c] * alpha;
        data[c] = t;
        t -= k;
        imean += t;
        ivar += t * t;
      }
    }
    float mean = reduce_add(imean) / channels;
    float var = (reduce_add(ivar) - channels * mean * mean) / channels;
    mean += k;
#else
  // Mean taken in dimension C.
  float imean = 0;
  if (skip != NULL) {
    foreach (c = 0 ... channels) {
      data[c] = data[c] * alpha + skip[c];
      imean += data[c];
    }
  } else {
    foreach (c = 0 ... channels) {
      data[c] *= alpha;
      imean += data[c];
    }
  }
  float mean = reduce_add(imean) / channels;

  // Variance.
  float ivar = 0;
  foreach (c = 0 ... channels) {
    float diff = data[c] - mean;
    ivar += diff * diff;
  }
  float var = reduce_add(ivar) / channels;
#endif

  float den = rsqrt(var + epsilon);
  foreach (c = 0 ... channels) {
    data[c] = betas[c] + gammas[c] * (data[c] - mean) * den;
  }
}
