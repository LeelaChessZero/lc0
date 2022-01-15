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

#include "selfplay/loop.h"

#include <optional>

#include "selfplay/tournament.h"
#include "utils/configfile.h"
#include "utils/optionsparser.h"

namespace lczero {

namespace {
const OptionId kInteractiveId{
    "interactive", "", "Run in interactive mode with UCI-like interface."};

const OptionId kLogFileId{"logfile", "LogFile",
  "Write log to that file. Special value <stderr> to "
  "output the log to the console."};

std::string MoveToSan(const Move in_move, const ChessBoard& board) {
  auto move = in_move;
  if (board.flipped()) move.Mirror();
  auto from = move.from();
  auto to = board.GetModernMove(move).to();

  std::string res;

  if (board.pawns().get(from)) {
    if (board.theirs().get(to) ||
        (from.row() == 4 && board.en_passant().get(7, to.col()))) {
      res = std::string(1, 'a' + in_move.from().col()) + 'x' +
            in_move.to().as_string();
    } else {
      res = in_move.to().as_string();
    }

    auto promotion = move.promotion();
    switch (promotion) {
      case Move::Promotion::Queen:
        return res + "=Q";
      case Move::Promotion::Rook:
        return res + "=R";
      case Move::Promotion::Bishop:
        return res + "=B";
      case Move::Promotion::Knight:
        return res + "=N";
      default:
        return res;
    }
  }

  if ((board.ours() & board.rooks()).get(to) &&
      (board.ours() & board.kings()).get(from)) {
    if (from.col() < to.col()) return "O-O";
    return "O-O-O";
  }

  int count = 0;
  int c_count = 0;
  int r_count = 0;
  if (board.kings().get(from)) {
    res = 'K';
    count = 1;
  } else if (board.bishops().get(from)) {
    res = 'B';
    for (auto sq : board.bishops() & board.ours()) {
      int dx = abs(sq.row() - to.row());
      int dy = abs(sq.col() - to.col());
      if (dx != dy) continue;
      count++;
      if (sq.col() == from.col()) c_count++;
      if (sq.row() == from.row()) r_count++;
    }
  } else if (board.queens().get(from)) {
    res = 'Q';
    for (auto sq : board.queens() & board.ours()) {
      int dx = abs(sq.row() - to.row());
      int dy = abs(sq.col() - to.col());
      if (dx != dy && dx != 0 && dy != 0) continue;
      count++;
      if (sq.col() == from.col()) c_count++;
      if (sq.row() == from.row()) r_count++;
    }
  } else if (board.rooks().get(from)) {
    res = 'R';
    for (auto sq : board.rooks() & board.ours()) {
      int dx = abs(sq.row() - to.row());
      int dy = abs(sq.col() - to.col());
      if (dx != 0 && dy != 0) continue;
      count++;
      if (sq.col() == from.col()) c_count++;
      if (sq.row() == from.row()) r_count++;
    }
  } else {
    res = 'N';
    for (auto sq : board.knights() & board.ours()) {
      int dx = abs(sq.row() - to.row());
      int dy = abs(sq.col() - to.col());
      if (dx + dy != 3 || dx == 0 || dy == 0) continue;
      count++;
      if (sq.col() == from.col()) c_count++;
      if (sq.row() == from.row()) r_count++;
    }
  }
  if (count > 1) {
    if (c_count == 1) {
      res += std::string(1, 'a' + in_move.from().col());
    } else if (r_count == 1) {
      res += std::string(1, '1' + in_move.from().row());
    } else {
      res += in_move.from().as_string();
    }
  }
  if (board.theirs().get(to)) {
    res += 'x';
  }
  res += in_move.to().as_string();
  return res;
}
}  // namespace

SelfPlayLoop::SelfPlayLoop() {}

SelfPlayLoop::~SelfPlayLoop() {
  if (tournament_) tournament_->Abort();
  if (thread_) thread_->join();
}

void SelfPlayLoop::RunLoop() {
  SelfPlayTournament::PopulateOptions(&options_);

  options_.Add<BoolOption>(kInteractiveId) = false;
  options_.Add<StringOption>(kLogFileId);

  if (!options_.ProcessAllFlags()) return;
  
  Logging::Get().SetFilename(options_.GetOptionsDict().Get<std::string>(kLogFileId));

  if (options_.GetOptionsDict().Get<bool>(kInteractiveId)) {
    UciLoop::RunLoop();
  } else {
    // Send id before starting tournament to allow wrapping client to know
    // who we are.
    SendId();
    SelfPlayTournament tournament(
        options_.GetOptionsDict(),
        std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
        std::bind(&UciLoop::SendInfo, this, std::placeholders::_1),
        std::bind(&SelfPlayLoop::SendGameInfo, this, std::placeholders::_1),
        std::bind(&SelfPlayLoop::SendTournament, this, std::placeholders::_1));
    tournament.RunBlocking();
  }
}

void SelfPlayLoop::CmdUci() {
  SendId();
  for (const auto& option : options_.ListOptionsUci()) {
    SendResponse(option);
  }
  SendResponse("uciok");
}

void SelfPlayLoop::CmdStart() {
  if (tournament_) return;
  tournament_ = std::make_unique<SelfPlayTournament>(
      options_.GetOptionsDict(),
      std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
      std::bind(&UciLoop::SendInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendGameInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendTournament, this, std::placeholders::_1));
  thread_ =
      std::make_unique<std::thread>([this]() { tournament_->RunBlocking(); });
}

void SelfPlayLoop::CmdStop() {
  tournament_->Stop();
  tournament_->Wait();
}

void SelfPlayLoop::SendGameInfo(const GameInfo& info) {
  std::vector<std::string> responses;
  // Send separate resign report before gameready as client gameready parsing
  // will easily get confused by adding new parameters as both training file
  // and move list potentially contain spaces.
  if (info.min_false_positive_threshold) {
    std::string resign_res = "resign_report";
    resign_res +=
        " fp_threshold " + std::to_string(*info.min_false_positive_threshold);
    responses.push_back(resign_res);
  }
  std::string res = "gameready";
  if (!info.training_filename.empty())
    res += " trainingfile " + info.training_filename;
  if (info.game_id != -1) res += " gameid " + std::to_string(info.game_id);
  res += " play_start_ply " + std::to_string(info.play_start_ply);
  if (info.is_black)
    res += " player1 " + std::string(*info.is_black ? "black" : "white");
  if (info.game_result != GameResult::UNDECIDED) {
    res += std::string(" result ") +
           ((info.game_result == GameResult::DRAW)
                ? "draw"
                : (info.game_result == GameResult::WHITE_WON) ? "whitewon"
                                                              : "blackwon");
  }
  if (!info.moves.empty()) {
    res += " san ";
    ChessBoard board;
    int move_no = 1;
    if (!info.initial_fen.empty() &&
        info.initial_fen != ChessBoard::kStartposFen) {
      board.SetFromFen(info.initial_fen, nullptr, &move_no);
    } else {
      board.SetFromFen(ChessBoard::kStartposFen);
    }
    if (board.flipped()) res += std::to_string(move_no) + "...";

    for (auto move : info.moves) {
      if (!board.flipped()) {
        res += std::to_string(move_no) + ".";
      }
      res += MoveToSan(move, board) + " ";

      if (board.flipped()) {
        move.Mirror();
        move_no++;
      }
      board.ApplyMove(move);
      board.Mirror();
    }
    res.pop_back();  // Remove last space.
  }
  if (!info.initial_fen.empty() &&
      info.initial_fen != ChessBoard::kStartposFen) {
    res += " from_fen " + info.initial_fen;
  }
  responses.push_back(res);
  SendResponses(responses);
}

void SelfPlayLoop::CmdSetOption(const std::string& name,
                                const std::string& value,
                                const std::string& context) {
  options_.SetUciOption(name, value, context);
}

void SelfPlayLoop::SendTournament(const TournamentInfo& info) {
  const int winp1 = info.results[0][0] + info.results[0][1];
  const int losep1 = info.results[2][0] + info.results[2][1];
  const int draws = info.results[1][0] + info.results[1][1];

  // Initialize variables.
  float percentage = -1;
  std::optional<float> elo;
  std::optional<float> los;

  // Only caculate percentage if any games at all (avoid divide by 0).
  if ((winp1 + losep1 + draws) > 0) {
    percentage =
        (static_cast<float>(draws) / 2 + winp1) / (winp1 + losep1 + draws);
  }
  // Calculate elo and los if percentage strictly between 0 and 1 (avoids divide
  // by 0 or overflow).
  if ((percentage < 1) && (percentage > 0))
    elo = -400 * log(1 / percentage - 1) / log(10);
  if ((winp1 + losep1) > 0) {
    los = .5f +
          .5f * std::erf((winp1 - losep1) / std::sqrt(2.0 * (winp1 + losep1)));
  }
  std::ostringstream oss;
  oss << "tournamentstatus";
  if (info.finished) oss << " final";
  oss << " P1: +" << winp1 << " -" << losep1 << " =" << draws;

  if (percentage > 0) {
    oss << " Win: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (percentage * 100.0f) << "%";
  }
  if (elo) {
    oss << " Elo: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (*elo);
  }
  if (los) {
    oss << " LOS: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (*los * 100.0f) << "%";
  }

  oss << " P1-W: +" << info.results[0][0] << " -" << info.results[2][0] << " ="
      << info.results[1][0];
  oss << " P1-B: +" << info.results[0][1] << " -" << info.results[2][1] << " ="
      << info.results[1][1];
  oss << " npm " + std::to_string(static_cast<double>(info.nodes_total_) /
                                  info.move_count_);
  oss << " nodes " + std::to_string(info.nodes_total_);
  oss << " moves " + std::to_string(info.move_count_);
  SendResponse(oss.str());
}

}  // namespace lczero
