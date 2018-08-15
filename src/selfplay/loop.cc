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
  Toolkit and the the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "selfplay/loop.h"
#include "selfplay/tournament.h"
#include "utils/configfile.h"
#include <iomanip>
#include <sstream>
#include <cmath>

namespace lczero {

namespace {
const char* kInteractive = "Run in interactive mode with uci-like interface";
}  // namespace

SelfPlayLoop::SelfPlayLoop() {}

SelfPlayLoop::~SelfPlayLoop() {
  if (tournament_) tournament_->Abort();
  if (thread_) thread_->join();
}

void SelfPlayLoop::RunLoop() {
  options_.Add<BoolOption>(kInteractive, "interactive") = false;
  SelfPlayTournament::PopulateOptions(&options_);

  if (!options_.ProcessAllFlags()) return;
  if (options_.GetOptionsDict().Get<bool>(kInteractive)) {
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
  options_.SendAllOptions();
  tournament_ = std::make_unique<SelfPlayTournament>(
      options_.GetOptionsDict(),
      std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
      std::bind(&UciLoop::SendInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendGameInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendTournament, this, std::placeholders::_1));
  thread_ =
      std::make_unique<std::thread>([this]() { tournament_->RunBlocking(); });
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
    res += " moves";
    for (const auto& move : info.moves) res += " " + move.as_string();
  }
  responses.push_back(res);
  SendResponses(responses);
}

void SelfPlayLoop::CmdSetOption(const std::string& name,
                                const std::string& value,
                                const std::string& context) {
  options_.SetOption(name, value, context);
}

void SelfPlayLoop::SendTournament(const TournamentInfo& info) {
  std::string res = "tournamentstatus";
  if (info.finished) res += " final";
  res += " win " + std::to_string(info.results[0][0]) + " " +
         std::to_string(info.results[0][1]);
  res += " lose " + std::to_string(info.results[2][0]) + " " +
         std::to_string(info.results[2][1]);
  res += " draw " + std::to_string(info.results[1][0]) + " " +
         std::to_string(info.results[1][1]);
  SendResponse(res);

  // Compute player1's elo relative to player 2.
  auto wins = info.results[0][0]+info.results[0][1];
  auto losses = info.results[2][0] + info.results[2][1];
  auto draws = info.results[1][0] + info.results[1][1];
  auto n = wins + losses + draws;
  if (n == 0) {
    // no games
    return;
  }
  double w, l, d;
  w = wins / (double)n;
  l = losses / (double)n;
  d = draws / (double)n;
  // mu is the estimate of the expected score
  auto mu = w + d / 2.0;
  auto devW = w * pow(1.0 - mu, 2);
  auto devL = l * pow(mu, 2);
  auto devD = d * pow(0.5 - mu, 2);
  // The variance of player1's score from a single game is
  // approximated as devW + devL + devD; she scores
  // 1 with probability ~w, 0.5 w.p. ~d, and 0 w.p. ~l.
  // We approximate the variance of the average of n of these as
  // (devW + devL + devD) / n by the central limit theorem
  auto stddev = sqrt((devW + devL + devD) / n);
  // The 95% confidence interval is thus given by
  // (mu - stddev * 1.96, mu + stddev * 1.96)
  auto muLow = mu - stddev * 1.96;
  auto muHigh = mu + stddev * 1.96;
  auto eloDelta = [](const double score) {
    // Compute the difference in elo between players, for a given score
    return -400. * log10(1.0 / score - 1.0);
  };
  auto inBounds = [](const double score) {
    // is the score within the bounds of what we can estimate elo and elo bounds for?
    return score > 0 && score < 1;
  };
  std::ostringstream out;
  if (!inBounds(mu) || !inBounds(muLow) || !inBounds(muHigh)) {
    return;
  }
  out << std::setprecision(3)
      << "The estimated elo of player1 relative to player2 is "
      << eloDelta(mu) << " with confidence interval (" << eloDelta(muLow)
      << " , " << eloDelta(muHigh) << ")";
  SendResponse(out.str());
}

}  // namespace lczero
