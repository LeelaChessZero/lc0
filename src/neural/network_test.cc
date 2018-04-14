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

#include <gtest/gtest.h>
#include "neural/loader.h"
#include "neural/network_tf.h"

namespace lczero {

TEST(Network, FakeData) {
  auto weights = LoadWeightsFromFile(
      "../testdata/"
      "218a136a377302cce2c645e6436b0cb8284764319046dbd5f57f7aaeb498580a");
  auto network = MakeTensorflowNetwork(weights);
  auto compute = network->NewComputation();
  for (int j = 0; j < 4; ++j) {
    InputPlanes planes(kInputPlanes);
    for (int i = 0; i < kInputPlanes; ++i) {
      planes[i].mask = 0x230709012008ull;
    }
    compute->AddInput(std::move(planes));
  }
  compute->ComputeBlocking();
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}