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

#include "proto/net.pb.h"
#include "utils/commandline.h"
#include "utils/exception.h"
#include "utils/filesystem.h"
#include "utils/logging.h"
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
  if (has_network_format && network_format != nf::NETWORK_CLASSICAL &&
      network_format != nf::NETWORK_SE) {
    // Already in a new format, return unchanged.
    return;
  }

  auto* net = file->mutable_format()->mutable_network_format();
  if (!has_network_format) {
    // Older protobufs don't have format definition.
    net->set_input(nf::INPUT_CLASSICAL_112_PLANE);
    net->set_output(nf::OUTPUT_CLASSICAL);
    net->set_network(nf::NETWORK_CLASSICAL_WITH_HEADFORMAT);
    net->set_value(nf::VALUE_CLASSICAL);
    net->set_policy(nf::POLICY_CLASSICAL);
  } else if (network_format == pblczero::NetworkFormat::NETWORK_CLASSICAL) {
    // Populate policyFormat and valueFormat fields in old protobufs
    // without these fields.
    net->set_network(nf::NETWORK_CLASSICAL_WITH_HEADFORMAT);
    net->set_value(nf::VALUE_CLASSICAL);
    net->set_policy(nf::POLICY_CLASSICAL);
  } else if (network_format == pblczero::NetworkFormat::NETWORK_SE) {
    net->set_network(nf::NETWORK_SE_WITH_HEADFORMAT);
    net->set_value(nf::VALUE_CLASSICAL);
    net->set_policy(nf::POLICY_CLASSICAL);
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
      net.format().weights_encoding() != pblczero::Format::LINEAR16) {
    throw Exception("Invalid weight file: unsupported encoding.");
  }

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
    throw Exception(
        "Text format weights files are no longer supported. Use a command line "
        "tool to convert it to the new format.");
  }

  return ParseWeightsProto(buffer);
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

      std::string str(buf, buf + sz);
      std::istringstream data(str);
      int val = 0;
      data >> val;
      if (!data.fail() && val == 2) {
        CERR << "Found txt network file: " << candidate.second;
        return candidate.second;
      }

      // First byte of the protobuf stream is 0x0d for fixed32, so we ignore it
      // as our own magic should suffice.
      const auto magic = buf[1] | (static_cast<uint32_t>(buf[2]) << 8) |
                         (static_cast<uint32_t>(buf[3]) << 16) |
                         (static_cast<uint32_t>(buf[4]) << 24);
      if (magic == kWeightMagic) {
        CERR << "Found pb network file: " << candidate.second;
        return candidate.second;
      }
    }
  }
  LOGFILE << "Network weights file not found.";
  return {};
}

}  // namespace lczero
