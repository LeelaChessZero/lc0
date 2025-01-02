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

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "neural/network.h"
#include "proto/net.pb.h"

namespace lczero {

class OptionsDict;
using FloatVector = std::vector<float>;
using FloatVectors = std::vector<FloatVector>;

using WeightsFile = pblczero::Net;

// Read weights file and fill the weights structure.
WeightsFile LoadWeightsFromFile(const std::string& filename);

// Read weights from the "locations", which is one of:
// * "<autodiscover>" -- tries to find a file which looks like a weights file.
// * "<embed>" -- weights are embedded in the binary.
// * filename -- reads weights from the file.
WeightsFile LoadWeights(std::string_view location);

// Extracts location from the "backend" parameter of options, and loads weights.
WeightsFile LoadWeightsFromOptions(const OptionsDict& options);

// Tries to find a file which looks like a weights file, and located in
// directory of binary_name or one of subdirectories. If there are several such
// files, returns one which has the latest modification date.
std::string DiscoverWeightsFile();

}  // namespace lczero
