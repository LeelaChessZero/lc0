/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2019 The LCZero Authors

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

namespace lczero {

class EscCodes {
 public:
  EscCodes() = delete;

  // Try to enable ANSI escape sequences for the current terminal.
  static void Init();

  // Supported ANSI escape sequences.
  static const char* Reset() { return enabled_ ? "\033[0m" : ""; }
  static const char* Bold() { return enabled_ ? "\033[1m" : ""; }
  static const char* Underline() { return enabled_ ? "\033[4m" : ""; }
  static const char* Reverse() { return enabled_ ? "\033[7m" : ""; }
  static const char* Normal() { return enabled_ ? "\033[22m" : ""; }
  static const char* NoUnderline() { return enabled_ ? "\033[24m" : ""; }
  static const char* NoReverse() { return enabled_ ? "\033[27m" : ""; }
  static const char* Black() { return enabled_ ? "\033[30m" : ""; }
  static const char* Red() { return enabled_ ? "\033[31m" : ""; }
  static const char* Green() { return enabled_ ? "\033[32m" : ""; }
  static const char* Yellow() { return enabled_ ? "\033[33m" : ""; }
  static const char* Blue() { return enabled_ ? "\033[34m" : ""; }
  static const char* Magenda() { return enabled_ ? "\033[35m" : ""; }
  static const char* Cyan() { return enabled_ ? "\033[36m" : ""; }
  static const char* White() { return enabled_ ? "\033[37m" : ""; }
  static const char* BlackBg() { return enabled_ ? "\033[40m" : ""; }
  static const char* RedBg() { return enabled_ ? "\033[41m" : ""; }
  static const char* GreenBg() { return enabled_ ? "\033[42m" : ""; }
  static const char* YellowBg() { return enabled_ ? "\033[43m" : ""; }
  static const char* BlueBg() { return enabled_ ? "\033[44m" : ""; }
  static const char* MagendaBg() { return enabled_ ? "\033[45m" : ""; }
  static const char* CyanBg() { return enabled_ ? "\033[46m" : ""; }
  static const char* WhiteBg() { return enabled_ ? "\033[47m" : ""; }

 private:
  static bool enabled_;
};

}  // namespace lczero
