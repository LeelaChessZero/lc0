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

#pragma once

#include <memory>
#include <string>

#include "chess/uciloop.h"
#include "utils/optionsparser.h"

namespace lczero {

class EngineControllerBase {
 public:
  virtual ~EngineControllerBase() = default;

  // Blocks.
  virtual void EnsureReady() = 0;

  // Must not block.
  virtual void NewGame() = 0;

  // Blocks.
  virtual void SetPosition(const std::string& fen,
                           const std::vector<std::string>& moves) = 0;

  // Must not block.
  virtual void Go(const GoParams& params) = 0;
  virtual void PonderHit() = 0;
  // Must not block.
  virtual void Stop() = 0;
};

class EngineLoop : public UciLoop {
 public:
  EngineLoop(std::unique_ptr<OptionsParser> options,
             std::function<std::unique_ptr<EngineControllerBase>(
                 std::unique_ptr<UciResponder> uci_responder,
                 const OptionsDict& options)>
                 engine_factory);

  void RunLoop() override;
  void CmdUci() override;
  void CmdIsReady() override;
  void CmdSetOption(const std::string& name, const std::string& value,
                    const std::string& context) override;
  void CmdUciNewGame() override;
  void CmdPosition(const std::string& position,
                   const std::vector<std::string>& moves) override;
  void CmdGo(const GoParams& params) override;
  void CmdPonderHit() override;
  void CmdStop() override;

 private:
  std::unique_ptr<OptionsParser> options_;
  std::unique_ptr<EngineControllerBase> engine_;
};

}  // namespace lczero