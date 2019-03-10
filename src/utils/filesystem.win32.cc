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

#include "utils/exception.h"
#include "utils/filesystem.h"

#include <windows.h>
#undef CreateDirectory

namespace lczero {

void CreateDirectory(const std::string& path) {
  if (CreateDirectoryA(path.c_str(), nullptr)) return;
  if (GetLastError() != ERROR_ALREADY_EXISTS) {
    throw Exception("Cannot create directory: " + path);
  }
}

std::vector<std::string> GetFileList(const std::string& directory) {
  std::vector<std::string> result;
  WIN32_FIND_DATAA dir;
  const auto handle = FindFirstFileA((directory + "\\*").c_str(), &dir);
  if (handle == INVALID_HANDLE_VALUE) return result;
  do {
    if ((dir.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
      result.emplace_back(dir.cFileName);
    }
  } while (FindNextFile(handle, &dir) != 0);
  FindClose(handle);
  return result;
}

uint64_t GetFileSize(const std::string& filename) {
  WIN32_FILE_ATTRIBUTE_DATA s;
  if (!GetFileAttributesExA(filename.c_str(), GetFileExInfoStandard, &s)) {
    throw Exception("Cannot stat file: " + filename);
  }
  return (static_cast<uint64_t>(s.nFileSizeHigh) << 32) + s.nFileSizeLow;
}

time_t GetFileTime(const std::string& filename) {
  WIN32_FILE_ATTRIBUTE_DATA s;
  if (!GetFileAttributesExA(filename.c_str(), GetFileExInfoStandard, &s)) {
    throw Exception("Cannot stat file: " + filename);
  }
  return (static_cast<uint64_t>(s.ftLastWriteTime.dwHighDateTime) << 32) +
         s.ftLastWriteTime.dwLowDateTime;
}

}  // namespace lczero
