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

#include <shared_mutex>
#include "mcts/search.h"
#include "neural/network.h"
#include "uciloop.h"
#include "ucioptions.h"
#include "utils/readprefmutex.h"

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
};

class EngineController {
 public:
  EngineController(BestMoveInfo::Callback best_move_callback,
                   UciInfo::Callback info_callback);

  ~EngineController() {
    // Make sure search is destructed first, and it still may be running in
    // a separate thread.
    search_.reset();
  }

  void GetUciOptions(UciOptions* options);

  // Blocks.
  void EnsureReady() { std::unique_lock<rp_shared_mutex> lock(busy_mutex_); }

  // Must not block.
  void NewGame();

  // Blocks.
  void SetPosition(const std::string& fen,
                   const std::vector<std::string>& moves);

  // Must not block.
  void Go(const GoParams& params);
  // Must not block.
  void Stop();
  void SetNetworkPath(const std::string& path);

 private:
  void MakeMove(Move move);
  void CreateNodeRoot();

  UciOptions* uci_options_ = nullptr;

  BestMoveInfo::Callback best_move_callback_;
  UciInfo::Callback info_callback_;

  std::unique_ptr<Network> network_;
  // Locked means that there is some work to wait before responding readyok.
  rp_shared_mutex busy_mutex_;
  using SharedLock = std::shared_lock<rp_shared_mutex>;

  std::unique_ptr<NodePool> node_pool_;
  Node* current_head_ = nullptr;
  Node* gamebegin_node_ = nullptr;
  std::unique_ptr<Search> search_;
};

}  // namespace lczero