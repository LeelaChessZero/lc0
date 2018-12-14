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

// This is ispc version of  WinogradConvolution3::TransformIn.

uniform const size_t kWidth = 8;
uniform const size_t kHeight = 8;
uniform const size_t kSquares = kWidth * kHeight;

uniform const size_t kWtiles = 4;                 //(kWidth + 1) / 2;
uniform const size_t kTiles = kWtiles * kWtiles;  // 16

uniform const size_t kWinogradAlpha = 4;
uniform const size_t kWinogradTile = kWinogradAlpha * kWinogradAlpha;

export void winograd_TransformIn_ispc(uniform size_t batch_size,
                                      const uniform float input[],
                                      uniform size_t channels,
                                      uniform float output[]) {
  float x[kWinogradAlpha][kWinogradAlpha];
  float T1[kWinogradAlpha][kWinogradAlpha];

  for (uniform size_t batch_index = 0; batch_index < batch_size;
       batch_index++) {
    uniform size_t input_batch = batch_index * kWidth * kHeight * channels;
    uniform size_t V_batch = channels * kTiles * batch_index;

    for (uniform int block_y = 0; block_y < kWtiles; block_y++) {
      for (uniform int block_x = 0; block_x < kWtiles; block_x++) {
        const uniform int yin = 2 * block_y - 1;
        const uniform int xin = 2 * block_x - 1;

        foreach (channel = 0 ... channels) {
          size_t V_channel = V_batch + channel;
          size_t input_channel = input_batch + channel * (kWidth * kHeight);

          for (uniform int i = 0; i < kWinogradAlpha; i++) {
            for (uniform int j = 0; j < kWinogradAlpha; j++) {
              if ((yin + i) >= 0 && (xin + j) >= 0 && (yin + i) < kHeight &&
                  (xin + j) < kWidth) {
                x[i][j] = input[input_channel + (yin + i) * kWidth + (xin + j)];
              } else {
                x[i][j] = 0.0f;
              }
            }
          }

          T1[0][0] = x[0][0] - x[2][0];
          T1[0][1] = x[0][1] - x[2][1];
          T1[0][2] = x[0][2] - x[2][2];
          T1[0][3] = x[0][3] - x[2][3];
          T1[1][0] = x[1][0] + x[2][0];
          T1[1][1] = x[1][1] + x[2][1];
          T1[1][2] = x[1][2] + x[2][2];
          T1[1][3] = x[1][3] + x[2][3];
          T1[2][0] = x[2][0] - x[1][0];
          T1[2][1] = x[2][1] - x[1][1];
          T1[2][2] = x[2][2] - x[1][2];
          T1[2][3] = x[2][3] - x[1][3];
          T1[3][0] = x[1][0] - x[3][0];
          T1[3][1] = x[1][1] - x[3][1];
          T1[3][2] = x[1][2] - x[3][2];
          T1[3][3] = x[1][3] - x[3][3];

          const size_t V_incr = channels * kTiles * batch_size;
          const size_t wTile_V =
              V_channel + channels * (block_y * kWtiles + block_x);

          output[wTile_V + V_incr * 0] = T1[0][0] - T1[0][2];
          output[wTile_V + V_incr * 1] = T1[0][1] + T1[0][2];
          output[wTile_V + V_incr * 2] = T1[0][2] - T1[0][1];
          output[wTile_V + V_incr * 3] = T1[0][1] - T1[0][3];
          output[wTile_V + V_incr * 4] = T1[1][0] - T1[1][2];
          output[wTile_V + V_incr * 5] = T1[1][1] + T1[1][2];
          output[wTile_V + V_incr * 6] = T1[1][2] - T1[1][1];
          output[wTile_V + V_incr * 7] = T1[1][1] - T1[1][3];
          output[wTile_V + V_incr * 8] = T1[2][0] - T1[2][2];
          output[wTile_V + V_incr * 9] = T1[2][1] + T1[2][2];
          output[wTile_V + V_incr * 10] = T1[2][2] - T1[2][1];
          output[wTile_V + V_incr * 11] = T1[2][1] - T1[2][3];
          output[wTile_V + V_incr * 12] = T1[3][0] - T1[3][2];
          output[wTile_V + V_incr * 13] = T1[3][1] + T1[3][2];
          output[wTile_V + V_incr * 14] = T1[3][2] - T1[3][1];
          output[wTile_V + V_incr * 15] = T1[3][1] - T1[3][3];
        }
      }
    }
  }
}

export void winograd_TransformOut_ispc(uniform size_t batch_size,
                                       const uniform float input[],
                                       uniform size_t channels,
                                       uniform float output[]) {
  float m[kWinogradTile];

  for (uniform size_t batch_index = 0; batch_index < batch_size;
       batch_index++) {
    const uniform size_t M_batch = channels * kTiles * batch_index;
    const uniform size_t output_batch = batch_index * kSquares * channels;

    for (uniform int block_y = 0; block_y < kWtiles; block_y++) {
      for (uniform int block_x = 0; block_x < kWtiles; block_x++) {
        const uniform int x = 2 * block_x;
        const uniform int y = 2 * block_y;

        foreach (channel = 0 ... channels) {
          const size_t M_channel = M_batch + channel;
          const size_t output_channel = output_batch + channel * kSquares;
          const uniform int b = block_y * kWtiles + block_x;
          const size_t M_wtile = M_channel + channels * b;
          const uniform int M_incr = channels * kTiles * batch_size;

          for (uniform int wTile = 0; wTile < kWinogradTile; wTile++) {
            m[wTile] = input[M_wtile + wTile * M_incr];
          }

          float o11 = m[0 * 4 + 0] + m[0 * 4 + 1] + m[0 * 4 + 2] +
                      m[1 * 4 + 0] + m[1 * 4 + 1] + m[1 * 4 + 2] +
                      m[2 * 4 + 0] + m[2 * 4 + 1] + m[2 * 4 + 2];

          float o12 = m[0 * 4 + 1] - m[0 * 4 + 2] - m[0 * 4 + 3] +
                      m[1 * 4 + 1] - m[1 * 4 + 2] - m[1 * 4 + 3] +
                      m[2 * 4 + 1] - m[2 * 4 + 2] - m[2 * 4 + 3];

          float o21 = m[1 * 4 + 0] + m[1 * 4 + 1] + m[1 * 4 + 2] -
                      m[2 * 4 + 0] - m[2 * 4 + 1] - m[2 * 4 + 2] -
                      m[3 * 4 + 0] - m[3 * 4 + 1] - m[3 * 4 + 2];

          float o22 = m[1 * 4 + 1] - m[1 * 4 + 2] - m[1 * 4 + 3] -
                      m[2 * 4 + 1] + m[2 * 4 + 2] + m[2 * 4 + 3] -
                      m[3 * 4 + 1] + m[3 * 4 + 2] + m[3 * 4 + 3];

          output[output_channel + (y)*kWidth + (x)] = o11;
          output[output_channel + (y)*kWidth + (x + 1)] = o12;
          output[output_channel + (y + 1) * kWidth + (x)] = o21;
          output[output_channel + (y + 1) * kWidth + (x + 1)] = o22;
        }
      }
    }
  }
}
