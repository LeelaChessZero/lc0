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

#pragma once

#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include "utils/mutex.h"

namespace lczero {

class Logging {
 public:
  static Logging& Get();

  // Sets the name of the log. Empty name disables logging.
  void SetFilename(const std::string& filename);

 private:
  // Writes line to the log, and appends new line character.
  void WriteLineRaw(const std::string& line);

  Mutex mutex_;
  std::string filename_ GUARDED_BY(mutex_);
  std::ofstream file_ GUARDED_BY(mutex_);
  std::deque<std::string> buffer_ GUARDED_BY(mutex_);

  Logging() = default;
  friend class LogMessage;
};

class LogMessage : public std::ostringstream {
 public:
  LogMessage(const char* file, int line);
  ~LogMessage();
};

}  // namespace lczero

#define LOGFILE ::lczero::LogMessage(__FILE__, __LINE__)