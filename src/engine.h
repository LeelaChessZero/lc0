/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

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

#include <vector>

#include "engine_loop.h"
#include "neural/memcache.h"
#include "search/search.h"

namespace lczero {

class Engine : public EngineControllerBase {
 public:
  Engine(const SearchFactory&, const OptionsDict&);

  static void PopulateOptions(OptionsParser*);

  void EnsureReady() override {};
  void NewGame() override;
  void SetPosition(const std::string& fen,
                   const std::vector<std::string>& moves) override;
  void Go(const GoParams& params) override;
  void PonderHit() override {}
  void Stop() override;

  void RegisterUciResponder(UciResponder*) override;
  void UnregisterUciResponder(UciResponder*) override;

 private:
  void UpdateBackendConfig();
  void EnsureSearchStopped();
  void EnsureSyzygyTablebasesLoaded();

  UciResponderForwarder uci_forwarder_;
  const OptionsDict& options_;
  std::unique_ptr<SearchBase> search_;  // absl_notnull
  std::string backend_name_;
  std::unique_ptr<CachingBackend> backend_;  // absl_nullable

  // Remember previous tablebase paths to detect when to reload them.
  std::string previous_tb_paths_;
  std::unique_ptr<SyzygyTablebase> syzygy_tb_;  // absl_nullable

  bool search_initialized_ = false;
};

}  // namespace lczero