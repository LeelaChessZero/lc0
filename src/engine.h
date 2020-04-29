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

#pragma once

#include <optional>

#include "chess/uciloop.h"
#include "mcts/search.h"
#include "neural/cache.h"
#include "neural/factory.h"
#include "neural/network.h"
#include "syzygy/syzygy.h"
#include "utils/mutex.h"
#include "utils/optionsparser.h"

namespace lczero {

struct CurrentPosition {
  std::string fen;
  std::vector<std::string> moves;
};

class EngineController {
 public:
  EngineController(std::unique_ptr<UciResponder> uci_responder,
                   const OptionsDict& options);

  ~EngineController() {
    // Make sure search is destructed first, and it still may be running in
    // a separate thread.
    search_.reset();
  }

  void PopulateOptions(OptionsParser* options);

  // Blocks.
  void EnsureReady();

  // Must not block.
  void NewGame();

  // Blocks.
  void SetPosition(const std::string& fen,
                   const std::vector<std::string>& moves);

  // Must not block.
  void Go(const GoParams& params);
  void PonderHit();
  // Must not block.
  void Stop();

 private:
  void UpdateFromUciOptions();

  void SetupPosition(const std::string& fen,
                     const std::vector<std::string>& moves);
  void ResetMoveTimer();
  void CreateFreshTimeManager();

  const OptionsDict& options_;

  std::unique_ptr<UciResponder> uci_responder_;

  // Locked means that there is some work to wait before responding readyok.
  RpSharedMutex busy_mutex_;
  using SharedLock = std::shared_lock<RpSharedMutex>;

  std::unique_ptr<TimeManager> time_manager_;
  std::unique_ptr<Search> search_;
  std::unique_ptr<NodeTree> tree_;
  std::unique_ptr<SyzygyTablebase> syzygy_tb_;
  std::unique_ptr<Network> network_;
  NNCache cache_;

  // Store current TB and network settings to track when they change so that
  // they are reloaded.
  std::string tb_paths_;
  NetworkFactory::BackendConfiguration network_configuration_;

  // The current position as given with SetPosition. For normal (ie. non-ponder)
  // search, the tree is set up with this position, however, during ponder we
  // actually search the position one move earlier.
  std::optional<CurrentPosition> current_position_;
  GoParams go_params_;

  std::optional<std::chrono::steady_clock::time_point> move_start_time_;

  // If true we can reset move_start_time_ in "Go".
  bool strict_uci_timing_;
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
  void CmdPonderHit() override;
  void CmdStop() override;

 private:
  OptionsParser options_;
  EngineController engine_;
};

}  // namespace lczero
