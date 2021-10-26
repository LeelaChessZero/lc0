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

#include <iostream>

#include "lc0ctl/leela2onnx.h"
#include "utils/commandline.h"
#include "utils/optionsparser.h"
#include "version.h"

int main(int argc, const char** argv) {
  using lczero::CommandLine;
  std::cout << "Lc0 tool v" << GetVersionStr() << " built " << __DATE__ << "\n";

  try {
    CommandLine::Init(argc, argv);
    CommandLine::RegisterMode("leela2onnx", "(default) Act as UCI engine");

    if (CommandLine::ConsumeCommand("leela2onnx")) {
      lczero::ConvertLeelaToOnnx();
    } else {
      lczero::OptionsParser options;
      options.ShowHelp();
    }

  } catch (std::exception& e) {
    std::cerr << "Unhandled exception: " << e.what() << std::endl;
    abort();
  }

  return 0;
}
