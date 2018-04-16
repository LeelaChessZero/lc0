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

#include <algorithm>
#include <functional>

#include "engine.h"
#include "mcts/search.h"
#include "neural/loader.h"
#include "neural/network_random.h"
#include "neural/network_tf.h"

namespace lczero {
namespace {
const int kDefaultThreads = 2;
const char* kThreadsOption = "Number of worker threads";

const char* kAutoDiscover = "<autodiscover>";

SearchLimits PopulateSearchLimits(int ply, bool is_black,
                                  const GoParams& params) {
  SearchLimits limits;
  limits.nodes = params.nodes;
  limits.time_ms = params.movetime;
  int64_t time = (is_black ? params.btime : params.wtime);
  if (params.infinite || time < 0) return limits;
  int increment = std::max(int64_t(0), is_black ? params.binc : params.winc);

  int movestogo = params.movestogo < 0 ? 50 : params.movestogo;
  limits.time_ms = (time + (increment * (movestogo - 1))) * 0.95 / movestogo;

  return limits;
}

}  // namespace

EngineController::EngineController(BestMoveInfo::Callback best_move_callback,
                                   UciInfo::Callback info_callback)
    : best_move_callback_(best_move_callback), info_callback_(info_callback) {}

void EngineController::GetUciOptions(UciOptions* options) {
  uci_options_ = options;
  using namespace std::placeholders;
  options->Add(std::make_unique<StringOption>(
      "Network weights file path", kAutoDiscover,
      std::bind(&EngineController::SetNetworkPath, this, _1), "weights", 'w'));

  options->Add(std::make_unique<SpinOption>(kThreadsOption, kDefaultThreads, 1,
                                            128, std::function<void(int)>{},
                                            "threads", 't'));
  options->Add(std::make_unique<SpinOption>(
      "NNCache size", 200000, 0, 999999999,
      std::bind(&EngineController::SetCacheSize, this, _1)));

  Search::PopulateUciParams(options);
}

void EngineController::SetNetworkPath(const std::string& path) {
  SharedLock lock(busy_mutex_);
  std::string net_path;
  if (path == kAutoDiscover) {
    net_path = DiscoveryWeightsFile(
        uci_options_ ? uci_options_->GetProgramName() : ".");
  } else {
    net_path = path;
  }
  Weights weights = LoadWeightsFromFile(net_path);
  // TODO Make backend selection.
  network_ = MakeTensorflowNetwork(weights);
  // network_ = MakeRandomNetwork();
}

void EngineController::SetCacheSize(int size) { cache_.SetCapacity(size); }

void EngineController::NewGame() {
  SharedLock lock(busy_mutex_);
  search_.reset();
  node_pool_.reset();
}

void EngineController::SetPosition(const std::string& fen,
                                   const std::vector<std::string>& moves_str) {
  SharedLock lock(busy_mutex_);
  search_.reset();
  ChessBoard starting_board;
  int no_capture_ply;
  int full_moves;
  starting_board.SetFromFen(fen, &no_capture_ply, &full_moves);

  if (!node_pool_) node_pool_ = std::make_unique<NodePool>();
  if (!tree_) tree_ = std::make_unique<NodeTree>(node_pool_.get());

  std::vector<Move> moves;
  for (const auto& move : moves_str) moves.emplace_back(move);
  tree_->ResetToPosition(starting_board, moves, no_capture_ply, full_moves);
}

void EngineController::Go(const GoParams& params) {
  if (!tree_) {
    SetPosition(ChessBoard::kStartingFen, {});
  }

  auto limits = PopulateSearchLimits(tree_->GetPlyCount(),
                                     tree_->IsBlackToMove(), params);

  search_ = std::make_unique<Search>(
      tree_->GetCurrentHead(), tree_->GetNodePool(), network_.get(),
      best_move_callback_, info_callback_, limits, *uci_options_, &cache_);

  search_->StartThreads(uci_options_->GetIntValue(kThreadsOption));
}

void EngineController::Stop() {
  if (search_) search_->Stop();
}

}  // namespace lczero