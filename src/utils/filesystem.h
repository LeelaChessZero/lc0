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

#pragma once

#include <cstdint>
#include <ctime>
#include <string>
#include <vector>

namespace lczero {

// Creates directory at a given path. Throws exception if cannot.
// Returns silently if already exists.
void CreateDirectory(const std::string& path);

// Returns list of full paths of regular files in this directory.
// Silently returns empty vector on error.
std::vector<std::string> GetFileList(const std::string& directory);

// Returns size of a file, 0 if file doesn't exist or can't be read.
uint64_t GetFileSize(const std::string& filename);

// Returns modification time of a file, 0 if file doesn't exist or can't be read.
time_t GetFileTime(const std::string& filename);

// Returns the base directory relative to which user specific non-essential data
// files are stored or an empty string if unspecified.
std::string GetUserCacheDirectory();

// Returns the base directory relative to which user specific configuration
// files are stored or an empty string if unspecified.
std::string GetUserConfigDirectory();

// Returns the base directory relative to which user specific data files are
// stored or an empty string if unspecified.
std::string GetUserDataDirectory();

// Returns a vector of base directories to search for configuration files.
std::vector<std::string> GetSystemConfigDirectoryList();

// Returns a vector of base directories to search for data files.
std::vector<std::string> GetSystemDataDirectoryList();

}  // namespace lczero
