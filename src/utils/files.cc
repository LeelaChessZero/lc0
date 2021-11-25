/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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

#include <zlib.h>

#include <cstdio>

#include "utils/exception.h"

namespace lczero {

std::string ReadFileToString(const std::string& filename) {
  std::string result;

  gzFile f = gzopen(filename.c_str(), "rb");
  if (f == Z_NULL) throw Exception("Cannot open file " + filename);

  std::size_t last_offset = 0;
  std::size_t size = 0x10000;

  while (true) {
    const size_t len = size - last_offset;
    result.resize(size);
    size_t count = gzread(f, result.data() + last_offset, len);
    if (count < len) {
      result.resize(last_offset + count);
      break;
    }
    last_offset += len;
    size *= 2;
  }

  gzclose(f);
  result.shrink_to_fit();
  return result;
}

void WriteStringToFile(const std::string& filename,
                       const std::string_view content) {
  std::FILE* f = std::fopen(filename.c_str(), "wb");
  if (f == nullptr) throw Exception("Cannot open file for write: " + filename);
  if (std::fwrite(content.data(), content.size(), 1, f) != 1) {
    throw Exception("Cannot write to file: " + filename);
  }
  std::fclose(f);
}

void WriteStringToGzFile(const std::string& filename,
                         std::string_view content) {
  gzFile f = gzopen(filename.c_str(), "wb");
  if (f == nullptr)
    throw Exception("Cannot open gzfile for write: " + filename);
  if (gzwrite(f, content.data(), content.size()) !=
      static_cast<int>(content.size())) {
    throw Exception("Cannot write to file: " + filename);
  }
  gzclose(f);
}

}  // namespace lczero