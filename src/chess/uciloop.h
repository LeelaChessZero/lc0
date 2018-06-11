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
*/

#pragma once

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "chess/callbacks.h"
#include "utils/exception.h"

namespace lczero {

struct GoParams {
  std::int64_t wtime = -1;
  std::int64_t btime = -1;
  std::int64_t winc = -1;
  std::int64_t binc = -1;
  int movestogo = -1;
  int depth = -1;
  int nodes = -1;
  std::int64_t movetime = -1;
  bool infinite = false;
  std::vector<std::string> searchmoves;
};

class UciLoop {
 public:
  virtual ~UciLoop() {}
  virtual void RunLoop();

  // Sends response to host.
  virtual void SendResponse(const std::string& response);
  void SendBestMove(const BestMoveInfo& move);
  void SendInfo(const ThinkingInfo& info);
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
  virtual void CmdGo(const GoParams& /*params*/) {
    throw Exception("Not supported");
  }
  virtual void CmdStop() { throw Exception("Not supported"); }
  virtual void CmdStart() { throw Exception("Not supported"); }

  void SetLogFilename(const std::string& filename);

 private:
  bool DispatchCommand(
      const std::string& command,
      const std::unordered_map<std::string, std::string>& params);

  std::ofstream debug_log_;
};

}  // namespace lczero
