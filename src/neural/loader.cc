/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2020 The LCZero Authors

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

#include "neural/loader.h"

#include <zlib.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "neural/shared_params.h"
#include "proto/net.pb.h"
#include "utils/commandline.h"
#include "utils/exception.h"
#include "utils/filesystem.h"
#include "utils/optionsdict.h"
#include "version.h"

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace lczero {

namespace {
const std::uint32_t kWeightMagic = 0x1c0;

std::string DecompressGzip(const std::string& filename) {
  const int kStartingSize = 8 * 1024 * 1024;  // 8M
  std::string buffer;
  buffer.resize(kStartingSize);
  int bytes_read = 0;

  // Read whole file into a buffer.
  FILE* fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    throw Exception("Cannot read weights from " + filename);
  }
  if (filename == CommandLine::BinaryName()) {
    // The network file should be appended at the end of the lc0 executable,
    // followed by the network file size and a "Lc0!" (0x2130634c) magic.
    int32_t size, magic;
    if (fseek(fp, -8, SEEK_END) || fread(&size, 4, 1, fp) != 1 ||
        fread(&magic, 4, 1, fp) != 1 || magic != 0x2130634c) {
      fclose(fp);
      throw Exception("No embedded file detected.");
    }
    fseek(fp, -size - 8, SEEK_END);
  }
  fflush(fp);
  gzFile file = gzdopen(dup(fileno(fp)), "rb");
  fclose(fp);
  if (!file) {
    throw Exception("Cannot process file " + filename);
  }
  while (true) {
    const int sz =
        gzread(file, &buffer[bytes_read], buffer.size() - bytes_read);
    if (sz < 0) {
      int errnum;
      throw Exception(gzerror(file, &errnum));
    }
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

void FixOlderWeightsFile(WeightsFile* file) {
  using nf = pblczero::NetworkFormat;
  auto network_format = file->format().network_format().network();
  const auto has_network_format = file->format().has_network_format();

  auto* net = file->mutable_format()->mutable_network_format();
  if (!has_network_format) {
    // Older protobufs don't have format definition.
    net->set_input(nf::INPUT_CLASSICAL_112_PLANE);
    net->set_output(nf::OUTPUT_CLASSICAL);
    net->set_network(nf::NETWORK_CLASSICAL_WITH_HEADFORMAT);
    net->set_value(nf::VALUE_CLASSICAL);
    net->set_policy(nf::POLICY_CLASSICAL);
  } else if (network_format == nf::NETWORK_CLASSICAL) {
    // Populate policyFormat and valueFormat fields in old protobufs
    // without these fields.
    net->set_network(nf::NETWORK_CLASSICAL_WITH_HEADFORMAT);
    net->set_value(nf::VALUE_CLASSICAL);
    net->set_policy(nf::POLICY_CLASSICAL);
  } else if (network_format == nf::NETWORK_SE) {
    net->set_network(nf::NETWORK_SE_WITH_HEADFORMAT);
    net->set_value(nf::VALUE_CLASSICAL);
    net->set_policy(nf::POLICY_CLASSICAL);
  } else if (network_format == nf::NETWORK_SE_WITH_HEADFORMAT &&
             file->weights().encoder().size() > 0) {
    // Attention body network made with old protobuf.
    auto* net = file->mutable_format()->mutable_network_format();
    net->set_network(nf::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT);
    if (file->weights().has_smolgen_w()) {
      // Need to override activation defaults for smolgen.
      net->set_ffn_activation(nf::ACTIVATION_RELU_2);
      net->set_smolgen_activation(nf::ACTIVATION_SWISH);
    }
  } else if (network_format == nf::NETWORK_AB_LEGACY_WITH_MULTIHEADFORMAT) {
    net->set_network(nf::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT);
  }

  // Get updated network format.
  if (file->format().network_format().network() ==
      nf::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT) {
    auto weights = file->weights();
    if (weights.has_policy_heads() && weights.has_value_heads()) {
      CERR << "Weights file has multihead format, updating format flag";
      net->set_network(nf::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT);
      net->set_input_embedding(nf::INPUT_EMBEDDING_PE_DENSE);
    }
    if (!file->format().network_format().has_input_embedding()) {
      net->set_input_embedding(nf::INPUT_EMBEDDING_PE_MAP);
    }
  }
}

WeightsFile ParseWeightsProto(const std::string& buffer) {
  WeightsFile net;
  net.ParseFromString(buffer);

  if (net.magic() != kWeightMagic) {
    throw Exception("Invalid weight file: bad header.");
  }

  const auto min_version =
      GetVersionStr(net.min_version().major(), net.min_version().minor(),
                    net.min_version().patch(), "", "");
  const auto lc0_ver = GetVersionInt();
  const auto net_ver =
      GetVersionInt(net.min_version().major(), net.min_version().minor(),
                    net.min_version().patch());

  FixOlderWeightsFile(&net);

  // Weights files with this signature are also compatible.
  if (net_ver != 0x5c99973 && net_ver > lc0_ver) {
    throw Exception("Invalid weight file: lc0 version >= " + min_version +
                    " required.");
  }

  if (net.has_weights() &&
      net.format().weights_encoding() != pblczero::Format::LINEAR16 &&
      net_ver < GetVersionInt(0, 33, 0)) {
    throw Exception("Invalid weight file: unsupported encoding.");
  }

  return net;
}

using FloatVector = std::vector<float>;
using FloatVectors = std::vector<FloatVector>;

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

void PopulateLastIntoVector(FloatVectors* vecs, pblczero::Weights::Layer* out) {
  out->set_params(
      std::string_view(reinterpret_cast<const char*>(vecs->back().data()),
                       vecs->back().size() * sizeof(float)));
  out->set_encoding(pblczero::Weights::Layer::FLOAT32);
  vecs->pop_back();
}

void PopulateConvBlockWeights(FloatVectors* vecs,
                              pblczero::Weights::ConvBlock* block) {
  PopulateLastIntoVector(vecs, block->mutable_bn_stddivs());
  PopulateLastIntoVector(vecs, block->mutable_bn_means());
  PopulateLastIntoVector(vecs, block->mutable_biases());
  PopulateLastIntoVector(vecs, block->mutable_weights());
}

WeightsFile ParseWeightsTxt(std::string& buffer) {
  WeightsFile net;

  FloatVectors vecs;
  vecs = LoadFloatsFromFile(&buffer);

  auto result = net.mutable_weights();

  // Populating backwards.
  PopulateLastIntoVector(&vecs, result->mutable_ip2_val_b());
  PopulateLastIntoVector(&vecs, result->mutable_ip2_val_w());
  PopulateLastIntoVector(&vecs, result->mutable_ip1_val_b());
  PopulateLastIntoVector(&vecs, result->mutable_ip1_val_w());
  PopulateConvBlockWeights(&vecs, result->mutable_value());

  PopulateLastIntoVector(&vecs, result->mutable_ip_pol_b());
  PopulateLastIntoVector(&vecs, result->mutable_ip_pol_w());
  PopulateConvBlockWeights(&vecs, result->mutable_policy());

  // Input + all the residual should be left.
  if ((vecs.size() - 4) % 8 != 0)
    throw Exception("Invalid weight file: parse error.");

  const int num_residual = (vecs.size() - 4) / 8;

  for (int i = 0; i < num_residual; i++) result->add_residual();

  for (int i = num_residual - 1; i >= 0; --i) {
    auto residual = result->mutable_residual(i);
    PopulateConvBlockWeights(&vecs, residual->mutable_conv2());
    PopulateConvBlockWeights(&vecs, residual->mutable_conv1());
  }
  PopulateConvBlockWeights(&vecs, result->mutable_input());

  FixOlderWeightsFile(&net);
  return net;
}
}  // namespace

WeightsFile LoadWeightsFromFile(const std::string& filename) {
  FloatVectors vecs;
  auto buffer = DecompressGzip(filename);

  if (buffer.size() < 2) {
    throw Exception("Invalid weight file: too small.");
  }
  if (buffer[0] == '1' && buffer[1] == '\n') {
    throw Exception("Invalid weight file: no longer supported.");
  }
  if (buffer[0] == '2' && buffer[1] == '\n') {
    return ParseWeightsTxt(buffer);
  }

  return ParseWeightsProto(buffer);
}

std::optional<WeightsFile> LoadWeights(std::string_view location) {
  std::string net_path = std::string(location);
  if (net_path == SharedBackendParams::kAutoDiscover) {
    net_path = DiscoverWeightsFile();
  } else if (net_path == SharedBackendParams::kEmbed) {
    net_path = CommandLine::BinaryName();
  }
  if (net_path.empty()) return std::nullopt;
  if (location == SharedBackendParams::kEmbed) {
    CERR << "Using embedded weights from binary: " << net_path;
  } else {
    CERR << "Loading weights file from: " << net_path;
  }
  return LoadWeightsFromFile(net_path);
}

std::string DiscoverWeightsFile() {
  const int kMinFileSize = 500000;  // 500 KB

  std::vector<std::string> data_dirs = {CommandLine::BinaryDirectory()};
  const std::string user_data_path = GetUserDataDirectory();
  if (!user_data_path.empty()) {
    data_dirs.emplace_back(user_data_path + "lc0");
  }
  for (const auto& dir : GetSystemDataDirectoryList()) {
    data_dirs.emplace_back(dir + (dir.back() == '/' ? "" : "/") + "lc0");
  }

  for (const auto& dir : data_dirs) {
    // Open all files in <dir> amd <dir>/networks,
    // ones which are >= kMinFileSize are candidates.
    std::vector<std::pair<time_t, std::string> > time_and_filename;
    for (const auto& path : {"", "/networks"}) {
      for (const auto& file : GetFileList(dir + path)) {
        const std::string filename = dir + path + "/" + file;
        if (GetFileSize(filename) < kMinFileSize) continue;
        time_and_filename.emplace_back(GetFileTime(filename), filename);
      }
    }

    std::sort(time_and_filename.rbegin(), time_and_filename.rend());

    // Open all candidates, from newest to oldest, possibly gzipped, and try to
    // read version for it. If version is 2 or if the file is our protobuf,
    // return it.
    for (const auto& candidate : time_and_filename) {
      const gzFile file = gzopen(candidate.second.c_str(), "rb");

      if (!file) continue;
      unsigned char buf[256];
      int sz = gzread(file, buf, 256);
      gzclose(file);
      if (sz < 0) continue;

      // First byte of the protobuf stream is 0x0d for fixed32, so we ignore it
      // as our own magic should suffice.
      const auto magic = buf[1] | (static_cast<uint32_t>(buf[2]) << 8) |
                         (static_cast<uint32_t>(buf[3]) << 16) |
                         (static_cast<uint32_t>(buf[4]) << 24);
      if (magic == kWeightMagic) return candidate.second;
    }
  }
  LOGFILE << "Network weights file not found.";
  return {};
}

}  // namespace lczero
