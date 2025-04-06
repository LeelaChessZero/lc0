/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2024 The LCZero Authors

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
#include "engine_classic.h"
#include "neural/shared_params.h"
#include "utils/configfile.h"

namespace lczero {
namespace {
const OptionId kLogFileId{"logfile", "LogFile",
                          "Write log to that file. Special value <stderr> to "
                          "output the log to the console.",
                          'l'};
}  // namespace

EngineLoop::EngineLoop(StringUciResponder* uci_responder,
                       OptionsParser* options, EngineControllerBase* engine)
    : UciLoop(uci_responder), options_(options), engine_(std::move(engine)) {
  engine_->RegisterUciResponder(uci_responder_);
}

EngineLoop::~EngineLoop() { engine_->UnregisterUciResponder(uci_responder_); }

void EngineLoop::RunLoop() {
  if (!ConfigFile::Init() || !options_->ProcessAllFlags()) return;
  const auto options = options_->GetOptionsDict();
  Logging::Get().SetFilename(options.Get<std::string>(kLogFileId));
  UciLoop::RunLoop();
}

void EngineLoop::CmdUci() {
  uci_responder_->SendId();
  for (const auto& option : options_->ListOptionsUci()) {
    uci_responder_->SendRawResponse(option);
  }
  uci_responder_->SendRawResponse("uciok");
}

void EngineLoop::CmdIsReady() {
  engine_->EnsureReady();
  uci_responder_->SendRawResponse("readyok");
}

void EngineLoop::CmdSetOption(const std::string& name, const std::string& value,
                              const std::string& context) {
  options_->SetUciOption(name, value, context);
  // Set the log filename for the case it was set in UCI option.
  Logging::Get().SetFilename(
      options_->GetOptionsDict().Get<std::string>(kLogFileId));
}

void EngineLoop::CmdUciNewGame() { engine_->NewGame(); }

void EngineLoop::CmdPosition(const std::string& position,
                             const std::vector<std::string>& moves) {
  std::string fen = position;
  if (fen.empty()) {
    fen = ChessBoard::kStartposFen;
  }
  engine_->SetPosition(fen, moves);
}

void EngineLoop::CmdGo(const GoParams& params) { engine_->Go(params); }

void EngineLoop::CmdPonderHit() { engine_->PonderHit(); }

void EngineLoop::CmdStop() { engine_->Stop(); }

namespace {
template <typename EngineType>
void RunEngineInternal(SearchFactory* factory) {
  StdoutUciResponder uci_responder;

  // Populate options from various sources.
  OptionsParser options_parser;
  options_parser.Add<StringOption>(kLogFileId);
  EngineType::PopulateOptions(&options_parser);
  factory->PopulateParams(&options_parser);       // Search params.
  uci_responder.PopulateParams(&options_parser);  // UCI params.
  SharedBackendParams::Populate(&options_parser);

  // Parse flags, show help, initialize logging, read config etc.
  if (!ConfigFile::Init() || !options_parser.ProcessAllFlags()) return;
  const auto options = options_parser.GetOptionsDict();
  Logging::Get().SetFilename(options.Get<std::string>(kLogFileId));

  // Create engine.
  EngineType engine = [&]() {
    if constexpr (std::is_same_v<EngineType, EngineClassic>) {
      return EngineType(options);
    } else {
      return EngineType(*factory, options);
    }
  }();
  EngineLoop loop(&uci_responder, &options_parser, &engine);

  // Run the stdin loop.
  std::cout.setf(std::ios::unitbuf);
  std::string line;
  while (std::getline(std::cin, line)) {
    LOGFILE << ">> " << line;
    try {
      if (!loop.ProcessLine(line)) break;
    } catch (Exception& ex) {
      uci_responder.SendRawResponse(std::string("error ") + ex.what());
    }
  }
}
}  // namespace

void RunEngine(SearchFactory* factory) { RunEngineInternal<Engine>(factory); }
void RunEngineClassic() { RunEngineInternal<EngineClassic>(nullptr); }

}  // namespace lczero
