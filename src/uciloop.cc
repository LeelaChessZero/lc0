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

#include "uciloop.h"

#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include "chess/board.h"
#include "engine.h"
#include "ucioptions.h"
#include "utils/exception.h"

namespace lczero {

namespace {
void SendResponse(const std::string& response) {
  static std::mutex output_mutex;
  std::lock_guard<std::mutex> lock(output_mutex);
  std::cout << response << std::endl;
}

void SendBestMove(const BestMoveInfo& move) {
  std::string res = "bestmove " + move.bestmove.as_string();
  if (move.ponder) res += " ponder " + move.ponder.as_string();
  SendResponse(res);
}
void SendInfo(const UciInfo& info) {
  std::string res = "info";

  if (info.depth >= 0) res += " depth " + std::to_string(info.depth);
  if (info.seldepth >= 0) res += " seldepth " + std::to_string(info.seldepth);
  if (info.time >= 0) res += " time " + std::to_string(info.time);
  if (info.nodes >= 0) res += " nodes " + std::to_string(info.nodes);
  if (info.score) res += " score cp " + std::to_string(*info.score);
  if (info.nps >= 0) res += " nps " + std::to_string(info.nps);

  if (!info.pv.empty()) {
    res += " pv";
    for (const auto& move : info.pv) res += " " + move.as_string();
  }
  if (!info.comment.empty()) res += " string " + info.comment;
  SendResponse(res);
}
}  // namespace

void UciLoop(int argc, const char** argv) {
  UciOptions options(argc, argv);
  std::cout.setf(std::ios::unitbuf);
  EngineController engine(SendBestMove, SendInfo);
  engine.GetUciOptions(&options);
  if (!options.ProcessAllFlags()) return;

  std::string line;
  bool options_sent = false;
  while (std::getline(std::cin, line)) {
    try {
      if (line.empty()) continue;

      const auto pos = line.find(' ');
      std::string command;
      std::string params;
      if (pos == std::string::npos) {
        command = line;
      } else {
        command = line.substr(0, pos);
        params = line.substr(pos + 1);
      }

      /// uci
      if (command == "uci") {
        SendResponse("id name The Lc0 chess engine.");
        SendResponse("id author The LCZero Authors.");
        for (const auto& option : options.ListOptionsUci()) {
          SendResponse(option);
        }
        SendResponse("uciok");
        continue;
      }

      /// isready
      if (command == "isready") {
        engine.EnsureReady();
        SendResponse("readyok");
        continue;
      }

      /// setoption
      if (command == "setoption") {
        if (params.substr(0, 5) != "name ") {
          SendResponse("error Bad setoption command: " + line);
          continue;
        }
        params = params.substr(5);
        auto pos = params.find(" value ");
        if (pos == std::string::npos) {
          SendResponse("error Setoption value expected: " + line);
          continue;
        }
        std::string name = params.substr(0, pos);
        std::string value = params.substr(pos + 7);
        options.SetOption(name, value);
        if (options_sent) {
          options.SendOption(name);
        }
        continue;
      }

      /// ucinewgame
      if (command == "ucinewgame") {
        if (!options_sent) {
          options.SendAllOptions();
          options_sent = true;
        }
        engine.NewGame();
        continue;
      }

      /// position
      if (command == "position") {
        if (!options_sent) {
          options.SendAllOptions();
          options_sent = true;
        }
        const std::string kMovesStr(" moves ");
        std::vector<std::string> moves;

        const auto pos = params.find(kMovesStr);
        std::string fen = params.substr(0, pos);

        if (fen == "startpos") {
          fen = ChessBoard::kStartingFen;
        } else if (fen.substr(0, 4) == "fen ") {
          fen = fen.substr(4);
        } else {
          SendResponse("error Bad position specification: " + fen);
          continue;
        }
        if (pos != std::string::npos) {
          std::istringstream iss(params.substr(pos + kMovesStr.size()));
          std::string move;
          while (iss >> move) {
            moves.push_back(move);
          }
        }
        engine.SetPosition(fen, std::move(moves));
        continue;
      }

      // go
      if (command == "go") {
        if (!options_sent) {
          options.SendAllOptions();
          options_sent = true;
        }
        GoParams go_params;
        std::istringstream iss(params);
        std::string token;
        while (iss >> token) {
          if (token == "infinite") {
            go_params.infinite = true;
          }
#define OPTION(x)       \
  if (token == #x) {    \
    iss >> go_params.x; \
    continue;           \
  }
          OPTION(wtime);
          OPTION(btime);
          OPTION(winc);
          OPTION(binc);
          OPTION(movestogo);
          OPTION(depth);
          OPTION(nodes);
          OPTION(movetime);
#undef OPTION
          SendResponse("error Ignoring unknown go option: " + token);
        }
        engine.Go(go_params);
        continue;
      }

      // stop
      if (command == "stop") {
        engine.Stop();
        continue;
      }

      // quit
      if (command == "quit") {
        break;
      }

      SendResponse("error Unknown command: " + command);
    } catch (Exception& ex) {
      SendResponse(std::string("error ") + ex.what());
    }
  }
}

}  // namespace lczero