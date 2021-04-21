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

#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "chess/callbacks.h"
#include "utils/exception.h"

namespace lczero {

struct GoParams {
  std::optional<std::int64_t> wtime;
  std::optional<std::int64_t> btime;
  std::optional<std::int64_t> winc;
  std::optional<std::int64_t> binc;
  std::optional<int> movestogo;
  std::optional<int> depth;
  std::optional<int> nodes;
  std::optional<std::int64_t> movetime;
  bool infinite = false;
  std::vector<std::string> searchmoves;
  bool ponder = false;
};

class UciLoop {
 public:
  virtual ~UciLoop() {}
  virtual void RunLoop();

  // Sends response to host.
  void SendResponse(const std::string& response);
  // Sends responses to host ensuring they are received as a block.
  virtual void SendResponses(const std::vector<std::string>& responses);
  void SendBestMove(const BestMoveInfo& move);
  void SendInfo(const std::vector<ThinkingInfo>& infos);
  void SendId();

  // Command handlers.
  virtual void CmdUci() { throw Exception("Not supported"); }
  virtual void CmdIsReady() { throw Exception("Not supported"); }
  virtual void CmdSetOption(const std::string& /*name*/,
                            const std::string& /*value*/,
                            const std::string& /*context*/) {
    throw Exception("Not supported");
  }
  virtual void CmdUciNewGame() { throw Exception("Not supported"); }
  virtual void CmdPosition(const std::string& /*position*/,
                           const std::vector<std::string>& /*moves*/) {
    throw Exception("Not supported");
  }
  virtual void CmdFen() { throw Exception("Not supported"); }
  virtual void CmdGo(const GoParams& /*params*/) {
    throw Exception("Not supported");
  }
  virtual void CmdStop() { throw Exception("Not supported"); }
  virtual void CmdPonderHit() { throw Exception("Not supported"); }
  virtual void CmdStart() { throw Exception("Not supported"); }

 private:
  bool DispatchCommand(
      const std::string& command,
      const std::unordered_map<std::string, std::string>& params);
};

}  // namespace lczero
