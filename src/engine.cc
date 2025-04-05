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

#include "engine.h"

#include <algorithm>

#include "chess/gamestate.h"
#include "chess/position.h"
#include "neural/backend.h"
#include "neural/memcache.h"
#include "neural/register.h"
#include "neural/shared_params.h"

namespace lczero {

Engine::Engine(std::unique_ptr<SearchBase> search, const OptionsDict& opts)
    : options_(opts), search_(std::move(search)) {}

Engine::~Engine() {}

namespace {
GameState MakeGameState(const std::string& fen,
                        const std::vector<std::string>& moves) {
  GameState state;
  state.startpos = Position::FromFen(fen);
  ChessBoard cur_board = state.startpos.GetBoard();
  state.moves.reserve(moves.size());
  for (const auto& move : moves) {
    Move m = cur_board.ParseMove(move);
    state.moves.push_back(m);
    cur_board.ApplyMove(m);
    cur_board.Mirror();
  }
  return state;
}
}  // namespace

void Engine::EnsureSearchStopped() {
  search_->AbortSearch();
  search_->WaitSearch();
}

void Engine::UpdateBackendConfig() {
  const std::string backend_name =
      options_.Get<std::string>(SharedBackendParams::kBackendId);
  const size_t cache_size =
      options_.Get<int>(SharedBackendParams::kNNCacheSizeId);
  if (!backend_ || backend_name != backend_name_ ||
      backend_->UpdateConfiguration(options_) == Backend::NEED_RESTART) {
    backend_name_ = backend_name;
    backend_ = CreateMemCache(BackendManager::Get()->CreateFromParams(options_),
                              cache_size);
    search_->SetBackend(backend_.get());
  } else {
    backend_->SetCacheSize(cache_size);
  }
}

void Engine::SetPosition(const std::string& fen,
                         const std::vector<std::string>& moves) {
  UpdateBackendConfig();
  EnsureSearchStopped();
  search_->SetPosition(MakeGameState(fen, moves));
  search_initialized_ = true;
}

void Engine::NewGame() { SetPosition(ChessBoard::kStartposFen, {}); }

void Engine::Go(const GoParams& params) {
  if (!search_initialized_) NewGame();
  search_->StartSearch(params);
}

void Engine::Stop() {
  if (search_) search_->StopSearch();
}

}  // namespace lczero
