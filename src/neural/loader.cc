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

#include "neural/loader.h"
#include <algorithm>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include "utils/exception.h"

namespace lczero {

FloatVectors LoadFloatsFromFile(const std::string& filename) {
  FloatVectors res;

  std::ifstream file(filename.c_str());
  std::string line;
  while (std::getline(file, line)) {
    FloatVector vec;

    std::istringstream iss(line);
    float val;
    while (iss >> val) {
      vec.push_back(val);
    }
    if (!iss.eof()) throw Exception("Cannot read weights from " + filename);

    if (!vec.empty()) {
      res.emplace_back(std::move(vec));
    }
  }
  if (!file.eof()) throw Exception("Cannot read weights from " + filename);

  return res;
}

namespace {
void PopulateLastIntoVector(FloatVectors* vecs, Weights::Vec* out) {
  *out = std::move(vecs->back());
  vecs->pop_back();
}

void PopulateConvBlockWeights(FloatVectors* vecs, Weights::ConvBlock* block) {
  PopulateLastIntoVector(vecs, &block->bn_stddivs);
  PopulateLastIntoVector(vecs, &block->bn_means);
  PopulateLastIntoVector(vecs, &block->biases);
  PopulateLastIntoVector(vecs, &block->weights);
}
}  // namespace

Weights LoadWeightsFromFile(const std::string& filename) {
  FloatVectors vecs = LoadFloatsFromFile(filename);

  if (vecs.size() <= 19)
    throw Exception("Weithts file " + filename +
                    " should have at least 19 lines");
  if (vecs[0][0] != 1) throw Exception("Weights version 1 expected");

  Weights result;
  // Populating backwards.
  PopulateLastIntoVector(&vecs, &result.ip2_val_b);
  PopulateLastIntoVector(&vecs, &result.ip2_val_w);
  PopulateLastIntoVector(&vecs, &result.ip1_val_b);
  PopulateLastIntoVector(&vecs, &result.ip1_val_w);
  PopulateConvBlockWeights(&vecs, &result.value);

  PopulateLastIntoVector(&vecs, &result.ip_pol_b);
  PopulateLastIntoVector(&vecs, &result.ip_pol_w);
  PopulateConvBlockWeights(&vecs, &result.policy);

  // Version, Input + all the residual should be left.
  if ((vecs.size() - 5) % 8 != 0)
    throw Exception("Bad number of lines in weights file");

  const int num_residual = (vecs.size() - 5) / 8;
  result.residual.resize(num_residual);
  for (int i = num_residual - 1; i >= 0; --i) {
    PopulateConvBlockWeights(&vecs, &result.residual[i].conv2);
    PopulateConvBlockWeights(&vecs, &result.residual[i].conv1);
  }

  PopulateConvBlockWeights(&vecs, &result.input);
  return result;
}

std::string DiscoveryWeightsFile(const std::string& binary_name) {
  const int kMinFileSize = 30000000;

  using namespace std::experimental::filesystem;
  std::string path = binary_name;
  auto pos = path.find_last_of("\\/");
  if (pos != std::string::npos) {
    path.resize(pos);
  } else {
    path = ".";
  }

  std::vector<std::pair<file_time_type, std::string>> candidates;
  for (const auto& file : recursive_directory_iterator(
           path, directory_options::skip_permission_denied)) {
    if (!is_regular_file(file.path())) continue;
    if (file_size(file.path()) < kMinFileSize) continue;
    candidates.emplace_back(last_write_time(file.path()), file.path());
  }

  std::sort(candidates.rbegin(), candidates.rend());

  for (const auto& candidate : candidates) {
    std::ifstream file(candidate.second.c_str());
    int val = 0;
    file >> val;
    if (!file.fail() && val == 1) {
      std::cerr << "Found network file: " << candidate.second << std::endl;
      return candidate.second;
    }
  }

  throw Exception("Network weights file not found.");
  return {};
}

}  // namespace lczero