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

#include "utils/exception.h"
#include "utils/filesystem.h"
#include "utils/string.h"

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
    bool exists = false;
    switch (entry->d_type) {
      case DT_REG:
        exists = true;
        break;
      case DT_LNK:
        // check that the soft link actually points to a regular file.
        const std::string filename = directory + "/" + entry->d_name;
        struct stat s;
        exists =
            stat(filename.c_str(), &s) == 0 && (s.st_mode & S_IFMT) == S_IFREG;
        break;
    }
    if (exists) result.push_back(entry->d_name);
  }
  closedir(dir);
  return result;
}

uint64_t GetFileSize(const std::string& filename) {
  struct stat s;
  if (stat(filename.c_str(), &s) < 0) {
    return 0;
  }
  return s.st_size;
}

time_t GetFileTime(const std::string& filename) {
  struct stat s;
  if (stat(filename.c_str(), &s) < 0) {
    return 0;
  }
#ifdef __APPLE__
  return s.st_mtimespec.tv_sec;
#else
  return s.st_mtim.tv_sec;
#endif
}

namespace {
bool CheckDir(const std::string& dirname) {
  struct stat s;
  return stat(dirname.c_str(), &s) == 0 && S_ISDIR(s.st_mode);
}

}  // namespace

std::string GetUserCacheDirectory() {
#ifdef __APPLE__
  constexpr auto kLocalDir = "Library/Caches/";
#else
  constexpr auto kLocalDir = ".cache/";
  const char *xdg_cache_home = std::getenv("XDG_CACHE_HOME");
  if (xdg_cache_home != NULL && CheckDir(xdg_cache_home)) {
    return std::string(xdg_cache_home) + "/";
  }
#endif
  const char *home = std::getenv("HOME");
  if (home == NULL) return std::string();
  const std::string path = std::string(home) + "/" + kLocalDir;
  if (!CheckDir(path)) return std::string();
  return path;
}

std::string GetUserConfigDirectory() {
#ifdef __APPLE__
  constexpr auto kLocalDir = "Library/Preferences/";
#else
  constexpr auto kLocalDir = ".config/";
  const char *xdg_config_home = std::getenv("XDG_CONFIG_HOME");
  if (xdg_config_home != NULL && CheckDir(xdg_config_home)) {
    return std::string(xdg_config_home) + "/";
  }
#endif
  const char *home = std::getenv("HOME");
  if (home == NULL) return std::string();
  const std::string path = std::string(home) + "/" + kLocalDir;
  if (!CheckDir(path)) return std::string();
  return path;
}

std::string GetUserDataDirectory() {
#ifdef __APPLE__
  constexpr auto kLocalDir = "Library/";
#else
  constexpr auto kLocalDir = ".local/share/";
  const char *xdg_data_home = std::getenv("XDG_DATA_HOME");
  if (xdg_data_home != NULL && CheckDir(xdg_data_home)) {
    return std::string(xdg_data_home) + "/";
  }
#endif
  const char *home = std::getenv("HOME");
  if (home == NULL) return std::string();
  const std::string path = std::string(home) + "/" + kLocalDir;
  if (!CheckDir(path)) return std::string();
  return path;
}

std::vector<std::string> GetSystemConfigDirectoryList() {
#ifdef __APPLE__
  return {};
#else
  std::vector<std::string> result;
  const char *xdg_config_dirs = std::getenv("XDG_CONFIG_DIRS");
  if (xdg_config_dirs == NULL) {
    return {"/etc/xdg/"};
  }
  result = StrSplit(xdg_config_dirs, ":");
  return result;
#endif
}

std::vector<std::string> GetSystemDataDirectoryList() {
#ifdef __APPLE__
  return {};
#else
  std::vector<std::string> result;
  const char *xdg_data_dirs = std::getenv("XDG_DATA_DIRS");
  if (xdg_data_dirs == NULL) {
    return {"/usr/local/share/", "/usr/share/"};
  }
  result = StrSplit(xdg_data_dirs, ":");
  return result;
#endif
}

}  // namespace lczero
