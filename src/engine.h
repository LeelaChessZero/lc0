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

#include "mcts/search.h"
#include "neural/cache.h"
#include "neural/network.h"
#include "optionsparser.h"
#include "uciloop.h"
#include "utils/mutex.h"

// CUDNN eval
// comment/disable this to enable tensor flow path
#define CUDNN_EVAL 1

namespace lczero {

class EngineController {
 public:
  EngineController(BestMoveInfo::Callback best_move_callback,
                   ThinkingInfo::Callback info_callback,
                   const OptionsDict& options);

  ~EngineController() {
    // Make sure search is destructed first, and it still may be running in
    // a separate thread.
    search_.reset();
  }

  void PopulateOptions(OptionsParser* options);

  // Blocks.
  void EnsureReady() { std::unique_lock<RpSharedMutex> lock(busy_mutex_); }

  // Must not block.
  void NewGame();

  // Blocks.
  void SetPosition(const std::string& fen,
                   const std::vector<std::string>& moves);

  // Must not block.
  void Go(const GoParams& params);
  // Must not block.
  void Stop();
  void SetCacheSize(int size);

 private:
  void UpdateNetwork();

  const OptionsDict& options_;

  BestMoveInfo::Callback best_move_callback_;
  ThinkingInfo::Callback info_callback_;

  NNCache cache_;
  std::unique_ptr<Network> network_;

  // Locked means that there is some work to wait before responding readyok.
  RpSharedMutex busy_mutex_;
  using SharedLock = std::shared_lock<RpSharedMutex>;

  std::unique_ptr<NodePool> node_pool_;
  std::unique_ptr<Search> search_;
  std::unique_ptr<NodeTree> tree_;

  // Store current network settings to track when they change so that they
  // are reloaded.
  std::string network_path_;
  std::string backend_;
  std::string backend_options_;
};

class EngineLoop : public UciLoop {
 public:
  EngineLoop();

  void RunLoop() override;
  void CmdUci() override;
  void CmdIsReady() override;
  void CmdSetOption(const std::string& name, const std::string& value,
                    const std::string& context) override;
  void CmdUciNewGame() override;
  void CmdPosition(const std::string& position,
                   const std::vector<std::string>& moves) override;
  void CmdGo(const GoParams& params) override;
  void CmdStop() override;

 private:
  void EnsureOptionsSent();

  OptionsParser options_;
  bool options_sent_ = false;
  EngineController engine_;
};

}  // namespace lczero