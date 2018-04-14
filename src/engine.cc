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
  int increment = std::max(0l, is_black ? params.binc : params.winc);

  // During first few moves policy network is mostly fine, so don't search deep.
  if (ply < 4 && (limits.nodes < 0 || limits.nodes > 400)) limits.nodes = 400;

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
      "NNCache size", 100000, 0, 999999999,
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
  // network_ = MakeTensorflowNetwork(weights);
  network_ = MakeRandomNetwork();
}

void EngineController::SetCacheSize(int size) { cache_.SetCapacity(size); }

void EngineController::NewGame() {
  SharedLock lock(busy_mutex_);
  search_.reset();
  node_pool_.reset();
  current_head_ = nullptr;
  gamebegin_node_ = nullptr;
}

void EngineController::MakeMove(Move move) {
  if (current_head_->board.flipped()) move.Mirror();

  Node* new_head = nullptr;
  for (Node* n = current_head_->child; n; n = n->sibling) {
    if (n->move == move) {
      new_head = n;
      break;
    }
  }
  node_pool_->ReleaseAllChildrenExceptOne(current_head_, new_head);
  if (!new_head) {
    new_head = node_pool_->GetNode();
    current_head_->child = new_head;
    new_head->parent = current_head_;
    new_head->board = current_head_->board;
    const bool capture = new_head->board.ApplyMove(move);
    new_head->board.Mirror();
    new_head->ply_count = current_head_->ply_count + 1;
    new_head->no_capture_ply = capture ? 0 : current_head_->no_capture_ply + 1;
    new_head->repetitions = ComputeRepetitions(new_head);
    new_head->move = move;
  }
  current_head_ = new_head;
}

void EngineController::SetPosition(const std::string& fen,
                                   const std::vector<std::string>& moves) {
  SharedLock lock(busy_mutex_);
  search_.reset();
  ChessBoard starting_board;
  int no_capture_ply;
  int full_moves;
  starting_board.SetFromFen(fen, &no_capture_ply, &full_moves);

  if (gamebegin_node_ && gamebegin_node_->board != starting_board) {
    // Completely different position.
    node_pool_.reset();
    current_head_ = nullptr;
    gamebegin_node_ = nullptr;
  }

  if (!node_pool_) node_pool_ = std::make_unique<NodePool>();

  if (!gamebegin_node_) {
    gamebegin_node_ = node_pool_->GetNode();
    gamebegin_node_->board = starting_board;
    gamebegin_node_->no_capture_ply = no_capture_ply;
    gamebegin_node_->ply_count =
        full_moves * 2 - (starting_board.flipped() ? 1 : 2);
  }

  current_head_ = gamebegin_node_;
  for (const auto& move : moves) {
    MakeMove(move);
  }
  node_pool_->ReleaseChildren(current_head_);
}

void EngineController::Go(const GoParams& params) {
  if (!current_head_) {
    SetPosition(ChessBoard::kStartingFen, {});
  }

  auto limits = PopulateSearchLimits(current_head_->ply_count,
                                     current_head_->board.flipped(), params);

  search_ = std::make_unique<Search>(
      current_head_, node_pool_.get(), network_.get(), best_move_callback_,
      info_callback_, limits, uci_options_, &cache_);

  search_->StartThreads(uci_options_ ? uci_options_->GetIntValue(kThreadsOption)
                                     : kDefaultThreads);
}

void EngineController::Stop() {
  if (search_) search_->Stop();
}

}  // namespace lczero