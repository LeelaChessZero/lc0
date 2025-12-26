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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "utils/fp16_utils.h"

#include <gtest/gtest.h>

namespace lczero {

testing::AssertionResult FP16Equal(const char* a_expr, const char* b_expr,
                                   uint16_t a, uint16_t b) {
  if (a == b) return testing::AssertionSuccess();
  std::ostringstream oss_a;
  oss_a << std::hex << a;
  std::ostringstream oss_b;
  oss_b << std::hex << b;
  return testing::AssertionFailure()
         << "Expected FP16 values to be equal:\n"
         << "  " << a_expr << "\n"
         << "     Which is: 0x" << oss_a.str() << "\n"
         << "  " << b_expr << "\n"
         << "     Which is: 0x" << oss_b.str() << "\n";
}

TEST(FP16, TestNormalConversion) {
  float values[] = {0.0f,
                    -0.000000029802322f,
                    0.000000029802326f,
                    -0.000000059604645f,
                    0.000060975552f,
                    -0.00006103515625f,
                    0.1f,
                    -0.5f,
                    0.99951172f,
                    -1.0f,
                    1.00097656f,
                    -2.0f,
                    3.5f,
                    -4.25f,
                    65488.0f,
                    -65488.004f,
                    65504.0f,
                    -65519.996f,
                    65520.0f,
                    -std::numeric_limits<float>::infinity()};
  uint16_t expected_fp16[] = {0x0,    0x8000, 0x1,    0x8001, 0x3ff,
                              0x8400, 0x2E66, 0xB800, 0x3BFF, 0xBC00,
                              0x3C01, 0xC000, 0x4300, 0xC440, 0x7BFE,
                              0xFBFF, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00};
  float expected_fp32[] = {0.0f,
                           -0.0f,
                           0.000000059604645f,
                           -0.000000059604645f,
                           0.000060975552f,
                           -0.00006103515625f,
                           0.0999755859f,
                           -0.5f,
                           0.99951172f,
                           -1.0f,
                           1.00097656f,
                           -2.0f,
                           3.5f,
                           -4.25f,
                           65472.0f,
                           -65504.0f,
                           65504.0f,
                           -65504.0f,
                           std::numeric_limits<float>::infinity(),
                           -std::numeric_limits<float>::infinity()};
  for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
    uint16_t fp16 = FP32toFP16(values[i]);
    EXPECT_PRED_FORMAT2(FP16Equal, fp16, expected_fp16[i]) << " at index " << i;
    float back = FP16toFP32(fp16);
    EXPECT_FLOAT_EQ(back, expected_fp32[i]) << " at index " << i;
  }
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
