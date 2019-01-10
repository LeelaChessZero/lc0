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

#include "neural/loader.h"

#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/io/coded_stream.h>
#include <zlib.h>
#include <algorithm>
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

namespace lczero {

namespace {
const std::uint32_t kWeightMagic = 0x1c0;

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

WeightsFile ParseWeightsProto(const std::string& buffer) {
  WeightsFile net;
  using namespace google::protobuf::io;

  ArrayInputStream raw_input_stream(buffer.data(), buffer.size());
  CodedInputStream input_stream(&raw_input_stream);
  // Set protobuf limit to 2GB, print warning at 500MB.
  input_stream.SetTotalBytesLimit(2000 * 1000000, 500 * 1000000);

  if (!net.ParseFromCodedStream(&input_stream))
    throw Exception("Invalid weight file: parse error.");

  if (net.magic() != kWeightMagic)
    throw Exception("Invalid weight file: bad header.");

  auto min_version =
      GetVersionStr(net.min_version().major(), net.min_version().minor(),
                    net.min_version().patch(), "");
  auto lc0_ver = GetVersionInt();
  auto net_ver =
      GetVersionInt(net.min_version().major(), net.min_version().minor(),
                    net.min_version().patch());

  if (net_ver > lc0_ver)
    throw Exception("Invalid weight file: lc0 version >= " + min_version +
                    " required.");

  if (net.format().weights_encoding() != pblczero::Format::LINEAR16)
    throw Exception("Invalid weight file: unsupported encoding.");

  // Older protobufs don't have format definition.
  // Populate format fields with legacy (or "classical") formats.
  if (!net.format().has_network_format()) {
    auto net_format = net.mutable_format()->mutable_network_format();
    using nf = pblczero::NetworkFormat;
    net_format->set_input(nf::INPUT_CLASSICAL_112_PLANE);
    net_format->set_output(nf::OUTPUT_CLASSICAL);
    net_format->set_network(nf::NETWORK_CLASSICAL);
  }

  return net;
}

}  // namespace

WeightsFile LoadWeightsFromFile(const std::string& filename) {
  FloatVectors vecs;
  auto buffer = DecompressGzip(filename);

  if (buffer.size() < 2)
    throw Exception("Invalid weight file: too small.");
  else if (buffer[0] == '1' && buffer[1] == '\n')
    throw Exception("Invalid weight file: no longer supported.");
  else if (buffer[0] == '2' && buffer[1] == '\n')
    throw Exception(
        "Text format weights files are no longer supported. Use a command line "
        "tool to convert it to the new format.");

  return ParseWeightsProto(buffer);
}

std::string DiscoverWeightsFile() {
  const int kMinFileSize = 500000;  // 500 KB

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
  // read version for it. If version is 2 or if the file is our protobuf,
  // return it.
  for (const auto& candidate : time_and_filename) {
    gzFile file = gzopen(candidate.second.c_str(), "rb");

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

    // First byte of the protobuf stream is 0x0d for fixed32, so we ignore it as
    // our own magic should suffice.
    auto magic = buf[1] | (static_cast<uint32_t>(buf[2]) << 8) |
                 (static_cast<uint32_t>(buf[3]) << 16) |
                 (static_cast<uint32_t>(buf[4]) << 24);
    if (magic == kWeightMagic) {
      CERR << "Found pb network file: " << candidate.second;
      return candidate.second;
    }
  }

  throw Exception("Network weights file not found.");
  return {};
}

}  // namespace lczero
