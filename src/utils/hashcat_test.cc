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

#include "utils/hashcat.h"
#include <gtest/gtest.h>

namespace lczero {

TEST(HashCat, TestCollision) {
  uint64_t hash1 = HashCat({0x8000000010500000, 0x4000080000002000,
                            0x8000000000002000, 0x4000000000000000});
  uint64_t hash2 = HashCat({0x4000000010500000, 0x1000080000002000,
                            0x4000000000002000, 0x1000000000000000});
  EXPECT_NE(hash1, hash2);
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}