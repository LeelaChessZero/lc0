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

#include "utils/exception.h"
#include "utils/filesystem.h"

#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>

namespace lczero {

void CreateDirectory(const std::string& path) {
  if (mkdir(path.c_str(), 0777) < 0 && errno != EEXIST) {
    throw Exception("Cannot create directory: " + path);
  }
}

std::vector<std::string> GetFileList(const std::string& directory) {
  std::vector<std::string> result;
  DIR* dir = opendir(directory.c_str());
  if (!dir) return result;
  while (auto* entry = readdir(dir)) {
    if (entry->d_type == DT_REG) {
      result.push_back(entry->d_name);
    }
  }
  closedir(dir);
  return result;
}

uint64_t GetFileSize(const std::string& filename) {
  struct stat s;
  if (stat(filename.c_str(), &s) < 0) {
    throw Exception("Cannot stat file: " + filename);
  }
  return s.st_size;
}

time_t GetFileTime(const std::string& filename) {
  struct stat s;
  if (stat(filename.c_str(), &s) < 0) {
    throw Exception("Cannot stat file: " + filename);
  }
#ifdef __APPLE__
  return s.st_mtimespec.tv_sec;
#else
  return s.st_mtim.tv_sec;
#endif
}

}  // namespace lczero