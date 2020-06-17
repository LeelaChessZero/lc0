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

#if __has_include(<filesystem>)
 #include <filesystem>
#else
 #include <experimental/filesystem>
 // This works for the compilers we care about.
 namespace std {
  namespace filesystem = experimental::filesystem;
 }
#endif

namespace lczero {

void CreateDirectory(const std::string& path) {
  try {
    std::filesystem::create_directory(path);
  } catch(std::filesystem::filesystem_error& e) {
    throw Exception("Cannot create directory: " + path);
  }
}

std::vector<std::string> GetFileList(const std::string& directory) {
  std::vector<std::string> result;
  try {
    const std::filesystem::path p(directory);
    if (std::filesystem::exists(p)) {
      for (const auto & entry : std::filesystem::directory_iterator(directory)) {
        if (!std::filesystem::is_symlink(entry.path())) {
          result.push_back(entry.path().filename());
        }
      }
    }
  } catch(std::filesystem::filesystem_error& e) {}
  return result;
}

uint64_t GetFileSize(const std::string& filename) {
  try {
    return static_cast<uint64_t>(std::filesystem::file_size(filename));
  } catch(std::filesystem::filesystem_error&) {}
  return 0;
}

time_t GetFileTime(const std::string& filename) {
  try {
    std::filesystem::path p(filename);
    auto ftime = std::filesystem::last_write_time(p);
    return decltype(ftime)::clock::to_time_t(ftime);
  } catch(std::filesystem::filesystem_error& e) {
    CERR << "GetFileTime Exception: " << e.what();
  }
  return 0;
}

std::string GetUserCacheDirectory() {
#ifdef _WIN32
    return "";
#else
#ifdef __APPLE__
  constexpr auto kLocalDir = "Library/Caches/";
#else
  constexpr auto kLocalDir = ".cache/";
  const char *xdg_cache_home = std::getenv("XDG_CACHE_HOME");
  if (xdg_cache_home != NULL) return std::string(xdg_cache_home) + "/";
#endif
  const char *home = std::getenv("HOME");
  if (home == NULL) return std::string();
  return std::string(home) + "/" + kLocalDir;
#endif
}

std::string GetUserConfigDirectory() {
#ifdef _WIN32
    return "";
#else
#ifdef __APPLE__
  constexpr auto kLocalDir = "Library/Preferences/";
#else
  constexpr auto kLocalDir = ".config/";
  const char *xdg_config_home = std::getenv("XDG_CONFIG_HOME");
  if (xdg_config_home != NULL) return std::string(xdg_config_home) + "/";
#endif
  const char *home = std::getenv("HOME");
  if (home == NULL) return std::string();
  return std::string(home) + "/" + kLocalDir;
#endif
}

std::string GetUserDataDirectory() {
#ifdef _WIN32
    return "";
#else
#ifdef __APPLE__
  constexpr auto kLocalDir = "Library/";
#else
  constexpr auto kLocalDir = ".local/share/";
  const char *xdg_data_home = std::getenv("XDG_DATA_HOME");
  if (xdg_data_home != NULL) return std::string(xdg_data_home) + "/";
#endif
  const char *home = std::getenv("HOME");
  if (home == NULL) return std::string();
  return std::string(home) + "/" + kLocalDir;
#endif
}

}  // namespace lczero
