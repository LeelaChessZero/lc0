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

#include "neural/shared_params.h"
#include "utils/configfile.h"

namespace lczero {
namespace {
const OptionId kLogFileId{"logfile", "LogFile",
                          "Write log to that file. Special value <stderr> to "
                          "output the log to the console.",
                          'l'};
const OptionId kPreload{"preload", "",
                        "Initialize backend and load net on engine startup."};
}  // namespace

EngineLoop::EngineLoop(std::unique_ptr<OptionsParser> options,
                       std::function<std::unique_ptr<EngineControllerBase>(
                           std::unique_ptr<UciResponder> uci_responder,
                           const OptionsDict& options)>
                           engine_factory)
    : options_(std::move(options)),
      engine_(engine_factory(
          std::make_unique<CallbackUciResponder>(
              std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
              std::bind(&UciLoop::SendInfo, this, std::placeholders::_1)),
          options_->GetOptionsDict())) {
  options_->Add<StringOption>(kLogFileId);
  options_->Add<BoolOption>(kPreload) = false;
  SharedBackendParams::Populate(options_.get());
}

void EngineLoop::RunLoop() {
  if (!ConfigFile::Init() || !options_->ProcessAllFlags()) return;
  const auto options = options_->GetOptionsDict();
  Logging::Get().SetFilename(options.Get<std::string>(kLogFileId));
  if (options.Get<bool>(kPreload)) engine_->NewGame();
  UciLoop::RunLoop();
}

void EngineLoop::CmdUci() {
  SendId();
  for (const auto& option : options_->ListOptionsUci()) {
    SendResponse(option);
  }
  SendResponse("uciok");
}

void EngineLoop::CmdIsReady() {
  engine_->EnsureReady();
  SendResponse("readyok");
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

}  // namespace lczero
