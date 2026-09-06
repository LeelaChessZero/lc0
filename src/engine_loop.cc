/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2025 The LCZero Authors

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

#include "engine_loop.h"

#include <iostream>

#include "engine.h"
#include "neural/shared_params.h"
#include "utils/configfile.h"

namespace lczero {
namespace {
const OptionId kLogFileId{
    {.long_flag = "logfile",
     .uci_option = "LogFile",
     .help_text = "Write log to that file. Special value <stderr> to "
                  "output the log to the console.",
     .short_flag = 'l',
     .visibility = OptionId::kAlwaysVisible}};
}  // namespace

void RunEngine(SearchFactory* factory) {
  CERR << "Search algorithm: " << factory->GetName();
  StdoutUciResponder uci_responder;

  // Populate options from various sources.
  OptionsParser options_parser;
  options_parser.Add<StringOption>(kLogFileId);
  ConfigFile::PopulateOptions(&options_parser);
  Engine::PopulateOptions(&options_parser);
  if (factory) factory->PopulateParams(&options_parser);  // Search params.
  uci_responder.PopulateParams(&options_parser);          // UCI params.
  SharedBackendParams::Populate(&options_parser);

  // Parse flags, show help, initialize logging, read config etc.
  if (!ConfigFile::Init() || !options_parser.ProcessAllFlags()) return;
  const auto options = options_parser.GetOptionsDict();
  Logging::Get().SetFilename(options.Get<std::string>(kLogFileId));

  // Create engine.
  Engine engine(*factory, options);
  UciLoop loop(&uci_responder, &options_parser, &engine);

  // Run the stdin loop.
  std::cout.setf(std::ios::unitbuf);
  std::string line;
  while (std::getline(std::cin, line)) {
    LOGFILE << ">> " << line;
    try {
      if (!loop.ProcessLine(line)) break;
      // Set the log filename for the case it was set in UCI option.
      Logging::Get().SetFilename(options.Get<std::string>(kLogFileId));
    } catch (Exception& ex) {
      uci_responder.SendRawResponse(std::string("error ") + ex.what());
    }
  }
}

}  // namespace lczero
