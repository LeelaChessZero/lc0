/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

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

uint32_t read_le(const uint8_t *addr) {
  return addr[0] + 256 * addr[1] + 65536 * addr[2] + 16777216 * addr[3];
}

// Read a .zip file containing a network file (just stored, not compressed a
// second time) that is appended at the end of the lc0 executable. Such a zip
// file can be generated using, for example, "zip -0 zipfile.zip net.pb.gz",
// "7z a -mx=0 zipfile.zip net.pb.gz" or the equivalent options of any other
// compression utility.

std::string DecompressEmbedded(std::string str) {
  constexpr int MOD_GZIP_ZLIB_WINDOWSIZE = 15;
  constexpr uint8_t eocd_sig[12] = {0x50, 0x4b, 5, 6, 0, 0, 0, 0, 1, 0, 1, 0};
  constexpr uint8_t header_sig[4] = {0x50, 0x4b, 3, 4};

  // Check if a zip file "end of central directory record" is there, 22 bytes
  // before the file end.
  uint8_t *eocd_addr =
      reinterpret_cast<uint8_t *>(str.data()) + str.size() - 22;
  if (memcmp(eocd_addr, eocd_sig, sizeof(eocd_sig)) != 0) {
    throw Exception("No embeded file detected.");
  }

  // Find the start of the zip file by subtracting the "central directory" size
  // and offset.
  uint8_t *start_addr =
      eocd_addr - read_le(eocd_addr + 12) - read_le(eocd_addr + 16);

  // Check for a local file header.
  if (memcmp(start_addr, header_sig, sizeof(header_sig)) != 0) {
    throw Exception("No embeded file header detected.");
  }

  z_stream zs;
  memset(&zs, 0, sizeof(zs));

  if (inflateInit2(&zs, MOD_GZIP_ZLIB_WINDOWSIZE + 16) != Z_OK) {
    throw Exception("inflateInit failed while decompressing.");
  }

  // Read the first file in the zip, should be a stored network file.
  uint32_t offsets = read_le(start_addr + 26);
  zs.next_in = reinterpret_cast<Bytef *>(start_addr) + 30 + (offsets >> 16) +
               (offsets & 0xffff);
  zs.avail_in = read_le(start_addr + 18);

  std::string filename =
      std::string(reinterpret_cast<char *>(start_addr) + 30, offsets & 0xffff);
  CERR << "Loading embedded weights file: " << filename;

  int ret;
  char outbuffer[32768];
  std::string outstring;

  do {
    zs.next_out = reinterpret_cast<Bytef *>(outbuffer);
    zs.avail_out = sizeof(outbuffer);

    ret = inflate(&zs, 0);

    if (outstring.size() < zs.total_out) {
      outstring.append(outbuffer, zs.total_out - outstring.size());
    }

  } while (ret == Z_OK);

  inflateEnd(&zs);

  if (ret != Z_STREAM_END) {
    std::ostringstream oss;
    oss << "Exception during zlib decompression: (" << ret << ") " << zs.msg;
    throw Exception(oss.str());
  }

  return outstring;
}

std::string DecompressGzip(const std::string& filename) {
  const int kStartingSize = 8 * 1024 * 1024;  // 8M
  std::string buffer;
  buffer.resize(kStartingSize);
  int bytes_read = 0;

  // Read whole file into a buffer.
  gzFile file = gzopen(filename.c_str(), "rb");
#ifdef _WIN32
  if (!file && filename == CommandLine::BinaryName()) {
    // Try again with a .exe suffix.
    file = gzopen((filename + ".exe").c_str(), "rb");
  }
#endif
  if (!file) throw Exception("Cannot read weights from " + filename);
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

  if (filename == CommandLine::BinaryName()) {
    return DecompressEmbedded(buffer);
  }

  return buffer;
}

void FixOlderWeightsFile(WeightsFile* file) {
  using nf = pblczero::NetworkFormat;
  auto network_format = file->format().network_format().network();
  if (file->format().has_network_format() &&
      network_format != nf::NETWORK_CLASSICAL &&
      network_format != nf::NETWORK_SE) {
    // Already in a new format, return unchanged.
    return;
  }

  WeightsFile::Builder builder(*file);

  auto format = file->format().AsBuilder();
  auto net_builder = file->format().network_format().AsBuilder();

  if (!file->format().has_network_format()) {
    // Older protobufs don't have format definition.
    net_builder.set_input(nf::INPUT_CLASSICAL_112_PLANE);
    net_builder.set_output(nf::OUTPUT_CLASSICAL);
    net_builder.set_network(nf::NETWORK_CLASSICAL_WITH_HEADFORMAT);
    net_builder.set_value(nf::VALUE_CLASSICAL);
    net_builder.set_policy(nf::POLICY_CLASSICAL);
  } else if (network_format == pblczero::NetworkFormat::NETWORK_CLASSICAL) {
    // Populate policyFormat and valueFormat fields in old protobufs
    // without these fields.
    net_builder.set_network(nf::NETWORK_CLASSICAL_WITH_HEADFORMAT);
    net_builder.set_value(nf::VALUE_CLASSICAL);
    net_builder.set_policy(nf::POLICY_CLASSICAL);
  } else if (network_format == pblczero::NetworkFormat::NETWORK_SE) {
    net_builder.set_network(nf::NETWORK_SE_WITH_HEADFORMAT);
    net_builder.set_value(nf::VALUE_CLASSICAL);
    net_builder.set_policy(nf::POLICY_CLASSICAL);
  }

  // It's only possible to replace the particular field completely.
  // So first replace network_format in format.
  format.set_network_format(net_builder.Build());
  // Then replace format in WeightsFile.
  builder.set_format(format.Build());
  *file = builder.Build();
}

WeightsFile ParseWeightsProto(const std::string& buffer) {
  WeightsFile net;
  net.ParseFromString(buffer);

  if (net.magic() != kWeightMagic) {
    throw Exception("Invalid weight file: bad header.");
  }

  const auto min_version =
      GetVersionStr(net.min_version().major(), net.min_version().minor(),
                    net.min_version().patch(), "");
  const auto lc0_ver = GetVersionInt();
  const auto net_ver =
      GetVersionInt(net.min_version().major(), net.min_version().minor(),
                    net.min_version().patch());

  FixOlderWeightsFile(&net);

  // Weights files with this signature are also compatible.
  if (net_ver != 0x5c99973 && net_ver > lc0_ver)
    throw Exception("Invalid weight file: lc0 version >= " + min_version +
                    " required.");

  if (net.format().weights_encoding() != pblczero::Format::LINEAR16)
    throw Exception("Invalid weight file: unsupported encoding.");

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

  const std::string root_path = CommandLine::BinaryDirectory();

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

    // First byte of the protobuf stream is 0x0d for fixed32, so we ignore it as
    // our own magic should suffice.
    const auto magic = buf[1] | (static_cast<uint32_t>(buf[2]) << 8) |
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
