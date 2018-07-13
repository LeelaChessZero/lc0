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
#include <zlib.h>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "lc0net.pb.h"
#include "utils/commandline.h"
#include "utils/exception.h"
#include "utils/filesystem.h"
#include "version.inc"

namespace lczero {

std::string DecompressGzip(const std::string& filename) {
  const int kStartingSize = 8 * 1024 * 1024;  // 8M
  std::string buffer;
  buffer.resize(kStartingSize);
  int bytes_read = 0;

  // Read whole file into a buffer.
  gzFile file = gzopen(filename.c_str(), "rb");
  if (!file) throw Exception("Cannot read weights from " + filename);
  while (true) {
    int sz = gzread(file, &buffer[bytes_read], buffer.size() - bytes_read);
    if (sz == static_cast<int>(buffer.size()) - bytes_read) {
      bytes_read = buffer.size();
      buffer.resize(buffer.size() * 2);
    } else {
      bytes_read += sz;
      buffer.resize(bytes_read);
      break;
    }
  }
  gzclose(file);

  return buffer;
}

void DenormLayer(const lc0::Weights_Layer& layer, FloatVectors& vecs) {
  FloatVector v;
  auto& buffer = layer.params();
  auto data = reinterpret_cast<const std::uint16_t*>(buffer.data());
  int n = buffer.length() / 2;
  v.resize(n);
  for (int i = 0; i < n; i++) {
    v[i] = data[i] / float(0xffff);
    v[i] *= layer.max_val() - layer.min_val();
    v[i] += layer.min_val();
  }
  vecs.emplace_back(v);
}

void DenormConvBlock(const lc0::Weights_ConvBlock& conv, FloatVectors& vecs) {
  DenormLayer(conv.weights(), vecs);
  DenormLayer(conv.biases(), vecs);
  DenormLayer(conv.bn_means(), vecs);
  DenormLayer(conv.bn_stddivs(), vecs);
}

FloatVectors LoadFloatsFromPbFile(const std::string& buffer) {
  auto net = lc0::Net();
  FloatVectors vecs;
  net.ParseFromString(buffer);

  std::string min_version(std::to_string(net.min_version().major()) + ".");
  min_version += std::to_string(net.min_version().minor()) + ".";
  min_version += std::to_string(net.min_version().patch());

  if (net.min_version().major() > LC0_VERSION_MAJOR)
    throw Exception("Weights require at least lc0 version: " + min_version);
  if (net.min_version().minor() > LC0_VERSION_MINOR)
    throw Exception("Weights require at least lc0 version: " + min_version);
  if (net.min_version().patch() > LC0_VERSION_PATCH)
    throw Exception("Weights require at least lc0 version: " + min_version);

  if (net.format().weights_encoding() != lc0::Format::LINEAR16)
    throw Exception("Invalid weight encoding");

  auto& w = net.weights();

  DenormConvBlock(w.input(), vecs);

  for (int i = 0, n = w.residual_size(); i < n; i++) {
    DenormConvBlock(w.residual(i).conv1(), vecs);
    DenormConvBlock(w.residual(i).conv2(), vecs);
  }

  DenormConvBlock(w.policy(), vecs);
  DenormLayer(w.ip_pol_w(), vecs);
  DenormLayer(w.ip_pol_b(), vecs);
  DenormConvBlock(w.value(), vecs);
  DenormLayer(w.ip1_val_w(), vecs);
  DenormLayer(w.ip1_val_b(), vecs);
  DenormLayer(w.ip2_val_w(), vecs);
  DenormLayer(w.ip2_val_b(), vecs);

  return vecs;
}

FloatVectors LoadFloatsFromFile(std::string* buffer) {
  // Parse buffer.
  FloatVectors result;
  FloatVector line;
  (*buffer) += "\n";
  size_t start = 0;
  for (size_t i = 0; i < buffer->size(); ++i) {
    char& c = (*buffer)[i];
    const bool is_newline = (c == '\n' || c == '\r');
    if (!std::isspace(c)) continue;
    if (start < i) {
      // If previous character was not space too.
      c = '\0';
      line.push_back(std::atof(&(*buffer)[start]));
    }
    if (is_newline && !line.empty()) {
      result.emplace_back();
      result.back().swap(line);
    }
    start = i + 1;
  }

  result.erase(result.begin());
  return result;
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
  FloatVectors vecs;
  auto buffer = DecompressGzip(filename);

  if (buffer[0] == '1' && buffer[1] == '\n')
    throw Exception("Weight file no longer supported");
  else if (buffer[0] == '2' && buffer[1] == '\n')
    vecs = LoadFloatsFromFile(&buffer);
  else
    vecs = LoadFloatsFromPbFile(buffer);

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
  if ((vecs.size() - 4) % 8 != 0)
    throw Exception("Bad number of lines in weights file");

  const int num_residual = (vecs.size() - 4) / 8;
  result.residual.resize(num_residual);
  for (int i = num_residual - 1; i >= 0; --i) {
    PopulateConvBlockWeights(&vecs, &result.residual[i].conv2);
    PopulateConvBlockWeights(&vecs, &result.residual[i].conv1);
  }

  PopulateConvBlockWeights(&vecs, &result.input);
  return result;
}

std::string DiscoveryWeightsFile() {
  const int kMinFileSize = 5000000;  // 5 MB

  std::string root_path = CommandLine::BinaryDirectory();

  // Open all files in <binary dir> amd <binary dir>/networks,
  // ones which are >= kMinFileSize are candidates.
  std::vector<std::pair<time_t, std::string> > time_and_filename;
  for (const auto& path : {"", "/networks"}) {
    for (const auto& file : GetFileList(root_path + path)) {
      const std::string filename = root_path + path + "/" + file;
      if (GetFileSize(filename) < kMinFileSize) continue;
      time_and_filename.emplace_back(GetFileTime(filename), filename);
    }
  }

  std::sort(time_and_filename.rbegin(), time_and_filename.rend());

  // Open all candidates, from newest to oldest, possibly gzipped, and try to
  // read version for it. If version is 2, return it.
  for (const auto& candidate : time_and_filename) {
    gzFile file = gzopen(candidate.second.c_str(), "rb");

    if (!file) continue;
    char buf[256];
    int sz = gzread(file, buf, 256);
    gzclose(file);
    if (sz < 0) continue;

    std::string str(buf, buf + sz);
    std::istringstream data(str);
    int val = 0;
    data >> val;
    if (!data.fail() && val == 2) {
      std::cerr << "Found network file: " << candidate.second << std::endl;
      return candidate.second;
    }
  }

  throw Exception("Network weights file not found.");
  return {};
}

}  // namespace lczero
