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

#include "uciloop.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "utils/exception.h"
#include "utils/logging.h"
#include "utils/string.h"
#include "version.h"

namespace lczero {
namespace {

const OptionId kUciChess960{
    {.long_flag = "chess960",
     .uci_option = "UCI_Chess960",
     .help_text = "Castling moves are encoded as \"king takes rook\".",
     .visibility = OptionId::kAlwaysVisible}};
const OptionId kShowWDL{{.long_flag = "show-wdl",
                         .uci_option = "UCI_ShowWDL",
                         .help_text = "Show win, draw and lose probability.",
                         .visibility = OptionId::kAlwaysVisible}};
const OptionId kShowMovesleft{{.long_flag = "show-movesleft",
                               .uci_option = "UCI_ShowMovesLeft",
                               .help_text = "Show estimated moves left.",
                               .visibility = OptionId::kAlwaysVisible}};

const std::unordered_map<std::string, std::unordered_set<std::string>>
    kKnownCommands = {
        {{"uci"}, {}},
        {{"isready"}, {}},
        {{"setoption"}, {"context", "name", "value"}},
        {{"ucinewgame"}, {}},
        {{"position"}, {"fen", "startpos", "moves"}},
        {{"go"},
         {"infinite", "wtime", "btime", "winc", "binc", "movestogo", "depth",
          "mate", "nodes", "movetime", "searchmoves", "ponder"}},
        {{"stop"}, {}},
        {{"ponderhit"}, {}},
        {{"quit"}, {}},
        {{"xyzzy"}, {}},
        {{"fen"}, {}},
};

std::pair<std::string, std::unordered_map<std::string, std::string>>
ParseCommand(const std::string& line) {
  std::unordered_map<std::string, std::string> params;
  std::string* value = nullptr;

  std::istringstream iss(line);
  std::string token;
  iss >> token >> std::ws;

  // If empty line, return empty command.
  if (token.empty()) return {};

  const auto command = kKnownCommands.find(token);
  if (command == kKnownCommands.end()) {
    throw Exception("Unknown command: " + line);
  }

  std::string whitespace;
  while (iss >> token) {
    auto iter = command->second.find(token);
    if (iter == command->second.end()) {
      if (!value) throw Exception("Unexpected token: " + token);
      *value += whitespace + token;
      whitespace = " ";
    } else {
      value = &params[token];
      iss >> std::ws;
      whitespace = "";
    }
  }
  return {command->first, params};
}

std::string GetOrEmpty(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& key) {
  const auto iter = params.find(key);
  if (iter == params.end()) return {};
  return iter->second;
}

int GetNumeric(const std::unordered_map<std::string, std::string>& params,
               const std::string& key) {
  const auto iter = params.find(key);
  if (iter == params.end()) {
    throw Exception("Unexpected error");
  }
  const std::string& str = iter->second;
  try {
    if (str.empty()) {
      throw Exception("expected value after " + key);
    }
    return std::stoi(str);
  } catch (std::invalid_argument&) {
    throw Exception("invalid value " + str);
  } catch (const std::out_of_range&) {
    throw Exception("out of range value " + str);
  }
}

bool ContainsKey(const std::unordered_map<std::string, std::string>& params,
                 const std::string& key) {
  return params.find(key) != params.end();
}
}  // namespace

UciLoop::UciLoop(StringUciResponder* uci_responder, OptionsParser* options,
                 EngineControllerBase* engine)
    : uci_responder_(uci_responder), options_(options), engine_(engine) {
  engine_->RegisterUciResponder(uci_responder_);
}

UciLoop::~UciLoop() { engine_->UnregisterUciResponder(uci_responder_); }

bool UciLoop::DispatchCommand(
    const std::string& command,
    const std::unordered_map<std::string, std::string>& params) {
  if (command == "uci") {
    uci_responder_->SendId();
    for (const auto& option : options_->ListOptionsUci()) {
      uci_responder_->SendRawResponse(option);
    }
    uci_responder_->SendRawResponse("uciok");
  } else if (command == "isready") {
    engine_->EnsureReady();
    uci_responder_->SendRawResponse("readyok");
  } else if (command == "setoption") {
    options_->SetUciOption(GetOrEmpty(params, "name"),
                           GetOrEmpty(params, "value"),
                           GetOrEmpty(params, "context"));
  } else if (command == "ucinewgame") {
    engine_->NewGame();
  } else if (command == "position") {
    if (ContainsKey(params, "fen") == ContainsKey(params, "startpos")) {
      throw Exception("Position requires either fen or startpos");
    }
    const std::vector<std::string> moves =
        StrSplitAtWhitespace(GetOrEmpty(params, "moves"));
    const std::string fen = GetOrEmpty(params, "fen");
    engine_->SetPosition(fen.empty() ? ChessBoard::kStartposFen : fen, moves);
  } else if (command == "go") {
    GoParams go_params;
    if (ContainsKey(params, "infinite")) {
      if (!GetOrEmpty(params, "infinite").empty()) {
        throw Exception("Unexpected token " + GetOrEmpty(params, "infinite"));
      }
      go_params.infinite = true;
    }
    if (ContainsKey(params, "searchmoves")) {
      go_params.searchmoves =
          StrSplitAtWhitespace(GetOrEmpty(params, "searchmoves"));
    }
    if (ContainsKey(params, "ponder")) {
      if (!GetOrEmpty(params, "ponder").empty()) {
        throw Exception("Unexpected token " + GetOrEmpty(params, "ponder"));
      }
      go_params.ponder = true;
    }
#define UCIGOOPTION(x)                    \
  if (ContainsKey(params, #x)) {          \
    go_params.x = GetNumeric(params, #x); \
  }
    UCIGOOPTION(wtime);
    UCIGOOPTION(btime);
    UCIGOOPTION(winc);
    UCIGOOPTION(binc);
    UCIGOOPTION(movestogo);
    UCIGOOPTION(depth);
    UCIGOOPTION(mate);
    UCIGOOPTION(nodes);
    UCIGOOPTION(movetime);
#undef UCIGOOPTION
    engine_->Go(go_params);
  } else if (command == "stop") {
    engine_->Stop();
  } else if (command == "ponderhit") {
    engine_->PonderHit();
  } else if (command == "xyzzy") {
    uci_responder_->SendRawResponse("Nothing happens.");
  } else if (command == "quit") {
    return false;
  } else {
    throw Exception("Unknown command: " + command);
  }
  return true;
}

bool UciLoop::ProcessLine(const std::string& line) {
  auto command = ParseCommand(line);
  // Ignore empty line.
  if (command.first.empty()) return true;
  return DispatchCommand(command.first, command.second);
}

void StringUciResponder::PopulateParams(OptionsParser* options) {
  options->Add<BoolOption>(kUciChess960) = false;
  options->Add<BoolOption>(kShowWDL) = true;
  options->Add<BoolOption>(kShowMovesleft) = false;
  options_ = &options->GetOptionsDict();
}

bool StringUciResponder::IsChess960() const {
  return options_ ? options_->Get<bool>(kUciChess960) : false;
}

void StringUciResponder::SendRawResponse(const std::string& response) {
  SendRawResponses({response});
}

void StringUciResponder::SendId() {
  SendRawResponse("id name Lc0 v" + GetVersionStr());
  SendRawResponse("id author The LCZero Authors.");
}

void StringUciResponder::OutputBestMove(BestMoveInfo* info) {
  const bool c960 = IsChess960();
  std::string res = "bestmove " + info->bestmove.ToString(c960);
  if (!info->ponder.is_null()) res += " ponder " + info->ponder.ToString(c960);
  if (info->player != -1) res += " player " + std::to_string(info->player);
  if (info->game_id != -1) res += " gameid " + std::to_string(info->game_id);
  if (info->is_black)
    res += " side " + std::string(*info->is_black ? "black" : "white");
  SendRawResponse(res);
}

void StringUciResponder::OutputThinkingInfo(std::vector<ThinkingInfo>* infos) {
  std::vector<std::string> reses;
  const bool c960 = IsChess960();
  for (const auto& info : *infos) {
    std::string res = "info";
    if (info.player != -1) res += " player " + std::to_string(info.player);
    if (info.game_id != -1) res += " gameid " + std::to_string(info.game_id);
    if (info.is_black)
      res += " side " + std::string(*info.is_black ? "black" : "white");
    if (info.depth >= 0)
      res += " depth " + std::to_string(std::max(info.depth, 1));
    if (info.seldepth >= 0) res += " seldepth " + std::to_string(info.seldepth);
    if (info.time >= 0) res += " time " + std::to_string(info.time);
    if (info.nodes >= 0) res += " nodes " + std::to_string(info.nodes);
    if (info.mate) res += " score mate " + std::to_string(*info.mate);
    if (info.score) res += " score cp " + std::to_string(*info.score);
    if (info.wdl && options_ && options_->Get<bool>(kShowWDL)) {
      res += " wdl " + std::to_string(info.wdl->w) + " " +
             std::to_string(info.wdl->d) + " " + std::to_string(info.wdl->l);
    }
    if (info.moves_left && options_ && options_->Get<bool>(kShowMovesleft)) {
      res += " movesleft " + std::to_string(*info.moves_left);
    }
    if (info.hashfull >= 0) res += " hashfull " + std::to_string(info.hashfull);
    if (info.nps >= 0) res += " nps " + std::to_string(info.nps);
    if (info.tb_hits >= 0) res += " tbhits " + std::to_string(info.tb_hits);
    if (info.multipv >= 0) res += " multipv " + std::to_string(info.multipv);

    if (!info.pv.empty()) {
      res += " pv";
      for (const auto& move : info.pv) res += " " + move.ToString(c960);
    }
    if (!info.comment.empty()) res += " string " + info.comment;
    reses.push_back(std::move(res));
  }
  SendRawResponses(reses);
}

void StdoutUciResponder::SendRawResponses(
    const std::vector<std::string>& responses) {
  static std::mutex output_mutex;
  std::lock_guard<std::mutex> lock(output_mutex);
  for (auto& response : responses) {
    LOGFILE << "<< " << response;
    std::cout << response << std::endl;
  }
}

}  // namespace lczero
