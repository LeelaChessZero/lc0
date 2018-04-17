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
                                   UciInfo::Callback info_callback,
                                   const OptionsDict& options)
    : options_(options),
      best_move_callback_(best_move_callback),
      info_callback_(info_callback) {}

void EngineController::GetUciOptions(UciOptions* options) {
  using namespace std::placeholders;

  options->Add<StringOption>(
      "Network weights file path", "weights", 'w',
      std::bind(&EngineController::SetNetworkPath, this, _1)) = kAutoDiscover;

  options->Add<SpinOption>(kThreadsOption, 1, 128, "threads", 't') =
      kDefaultThreads;

  options->Add<SpinOption>(
      "NNCache size", 0, 999999999, "nncache", '\0',
      std::bind(&EngineController::SetCacheSize, this, _1)) = 200000;

  Search::PopulateUciParams(options);
}

void EngineController::SetNetworkPath(const std::string& path) {
  SharedLock lock(busy_mutex_);
  std::string net_path;
  if (path == kAutoDiscover) {
    net_path = DiscoveryWeightsFile();
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

  if (!node_pool_) node_pool_ = std::make_unique<NodePool>();
  if (!tree_) tree_ = std::make_unique<NodeTree>(node_pool_.get());

  std::vector<Move> moves;
  for (const auto& move : moves_str) moves.emplace_back(move);
  tree_->ResetToPosition(fen, moves);
}

void EngineController::Go(const GoParams& params) {
  if (!tree_) {
    SetPosition(ChessBoard::kStartingFen, {});
  }

  auto limits = PopulateSearchLimits(tree_->GetPlyCount(),
                                     tree_->IsBlackToMove(), params);

  search_ = std::make_unique<Search>(
      tree_->GetCurrentHead(), tree_->GetNodePool(), network_.get(),
      best_move_callback_, info_callback_, limits, options_, &cache_);

  search_->StartThreads(options_.Get<int>(kThreadsOption));
}

void EngineController::Stop() {
  if (search_) search_->Stop();
}

}  // namespace lczero