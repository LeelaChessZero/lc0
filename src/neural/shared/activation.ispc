/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2022-2023 The LCZero Authors

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

export void ActivateRelu(uniform const size_t len, uniform float gamma,
                         const uniform float data[], const uniform float bias[],
                         uniform float beta, uniform float output[]) {
  foreach (b = 0 ... len) {
    float val = gamma * data[b] + bias[b] + beta;
    output[b] = val > 0 ? val : 0;
  }
}

export void ActivateSwish(uniform const size_t len, const uniform float data[],
                          const uniform float bias[], uniform float output[]) {
  foreach (b = 0 ... len) {
    float val = data[b] + bias[b];
    output[b] = val / (1.0f + exp(-val));
  }
}

export void ActivateRelu_2(uniform const size_t len, const uniform float data[],
                           const uniform float bias[], uniform float output[]) {
  foreach (b = 0 ... len) {
    float val = data[b] + bias[b];
    output[b] = val > 0 ? val * val : 0;
  }
}

static inline float selu(float val) {
  float alpha = 1.67326324f, scale = 1.05070098f;
  if (val > 0) {
    return scale * val;
  } else {
    return scale * alpha * (exp(val) - 1.0f);
  }
}

export void ActivateSelu(uniform const size_t len, const uniform float data[],
                         const uniform float bias[], uniform float output[]) {
  foreach (b = 0 ... len) {
    float val = data[b] + bias[b];
    output[b] = selu(val);
  }
}

export void SoftmaxActivation(uniform const size_t size,
                              const uniform float input[],
                              uniform float output[]) {
  float vmax = -3.4e38f;
  foreach (c = 0 ... size) {
    if (input[c] > vmax) vmax = input[c];
  }
  uniform float alpha = reduce_max(vmax);

  float t = 0.0f;
  foreach (c = 0 ... size) {
    float val = exp(input[c] - alpha);
    output[c] = val;
    t += val;
  }
  uniform float denom = 1.0f / reduce_add(t);

  foreach (c = 0 ... size) {
    output[c] *= denom;
  }
}
