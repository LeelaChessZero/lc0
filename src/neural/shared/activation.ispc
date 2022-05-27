/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2022 The LCZero Authors

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

static inline float mish(float val) {
  float e = exp(val);
  float n = e * e + 2.0f * e;
  float d = val / (n + 2.0f);
  if (val <= -0.5f) {
    return n * d;
  } else {
    return val - 2.0f * d;
  }
}

export void ActivateMish(uniform const size_t len, uniform float gamma,
                         const uniform float data[], const uniform float bias[],
                         uniform float beta, uniform float output[]) {
  foreach (b = 0 ... len) {
    float val = gamma * data[b] + bias[b] + beta;
    output[b] = mish(val);
  }
}
