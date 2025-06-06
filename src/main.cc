/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2021 The LCZero Authors

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

#include "chess/board.h"
#include "engine.h"
#include "search/register.h"
#include "selfplay/loop.h"
#include "tools/backendbench.h"
#include "tools/benchmark.h"
#include "tools/describenet.h"
#include "tools/leela2onnx.h"
#include "tools/onnx2leela.h"
#include "utils/commandline.h"
#include "utils/esc_codes.h"
#include "utils/logging.h"
#include "version.h"

namespace lczero {
void ChooseAndRunEngine() {
  // First try the engine which is explicitly specified on the command line.
  for (const std::string_view search_name :
       SearchManager::Get()->GetSearchNames()) {
    if (CommandLine::ConsumeCommand(search_name)) {
      RunEngine(SearchManager::Get()->GetFactoryByName(search_name));
      return;
    }
  }

  // Then if DEFAULT_SEARCH is defined, run the engine specified by it.
#ifdef DEFAULT_SEARCH
#define STRINGIFY_INTERNAL(x) #x
#define STRINGIFY(x) STRINGIFY_INTERNAL(x)
  SearchFactory* factory =
      SearchManager::Get()->GetFactoryByName(STRINGIFY(DEFAULT_SEARCH));
  if (!factory)
    throw Exception("Unknown search algorithm: " STRINGIFY(DEFAULT_SEARCH));
#undef STRINGIFY
#undef STRINGIFY_INTERNAL
  RunEngine(factory);
  return;
#endif

  // Then try to run the engine which is specified by the name of the binary.
  const std::string& binary_name = CommandLine::BinaryName();
  for (const std::string_view search_name :
       SearchManager::Get()->GetSearchNames()) {
    if (binary_name.find(search_name) != std::string::npos) {
      RunEngine(SearchManager::Get()->GetFactoryByName(search_name));
      return;
    }
  }

  // Finally, run "classic" search through the new API.
  RunEngine(SearchManager::Get()->GetFactoryByName("classic"));
}
}  // namespace lczero

int main(int argc, const char** argv) {
  using namespace lczero;
  EscCodes::Init();
  LOGFILE << "Lc0 started.";
  CERR << EscCodes::Bold() << EscCodes::Red() << "       _";
  CERR << "|   _ | |";
  CERR << "|_ |_ |_|" << EscCodes::Reset() << " v" << GetVersionStr()
       << " built " << __DATE__;

  try {
    InitializeMagicBitboards();

    CommandLine::Init(argc, argv);
    if (CommandLine::BinaryName().find("simple") == std::string::npos) {
      CommandLine::RegisterMode("selfplay", "Play games with itself");
      CommandLine::RegisterMode("benchmark", "Quick benchmark");
      CommandLine::RegisterMode("bench", "Very quick benchmark");
      CommandLine::RegisterMode("backendbench",
                                "Quick benchmark of backend only");
      CommandLine::RegisterMode("leela2onnx", "Convert Leela network to ONNX.");
      CommandLine::RegisterMode("onnx2leela",
                                "Convert ONNX network to Leela net.");
      CommandLine::RegisterMode("describenet",
                                "Shows details about the Leela network.");
    }
    for (const std::string_view search_name :
         SearchManager::Get()->GetSearchNames()) {
      CommandLine::RegisterMode(
          std::string(search_name),
          "Use \"" + std::string(search_name) + "\" search");
    }

    if (CommandLine::ConsumeCommand("selfplay")) {
      // Selfplay mode.
      StdoutUciResponder uci_responder;
      SelfPlayLoop loop(&uci_responder);
      loop.Run();
    } else if (CommandLine::ConsumeCommand("benchmark")) {
      // Benchmark mode, longer version.
      Benchmark benchmark;
      benchmark.Run();
    } else if (CommandLine::ConsumeCommand("bench")) {
      // Benchmark mode, shorter version.
      Benchmark benchmark;
      benchmark.Run(/*run_shorter_benchmark=*/true);
    } else if (CommandLine::ConsumeCommand("backendbench")) {
      // Backend Benchmark mode.
      BackendBenchmark benchmark;
      benchmark.Run();
    } else if (CommandLine::ConsumeCommand("leela2onnx")) {
      lczero::ConvertLeelaToOnnx();
    } else if (CommandLine::ConsumeCommand("onnx2leela")) {
      lczero::ConvertOnnxToLeela();
    } else if (CommandLine::ConsumeCommand("describenet")) {
      lczero::DescribeNetworkCmd();
    } else {
      lczero::ChooseAndRunEngine();
    }
  } catch (std::exception& e) {
    std::cerr << "Unhandled exception: " << e.what() << std::endl;
    abort();
  }
}
