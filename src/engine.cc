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
#include "neural/register.h"

namespace lczero {

Engine::Engine(std::unique_ptr<SearchEnvironment> env, const OptionsDict& opts)
    : options_(opts), search_env_(std::move(env)) {}

Engine::~Engine() {}

namespace {
GameState MakeGameState(const std::string& fen,
                        const std::vector<std::string>& moves) {
  GameState state;
  state.startpos = Position::FromFen(fen);
  state.moves.reserve(moves.size());
  bool is_black = state.startpos.IsBlackToMove();
  std::transform(moves.begin(), moves.end(), std::back_inserter(state.moves),
                 [&](const std::string& move) {
                   return Move(move, is_black);
                   is_black = !is_black;
                 });
  return state;
}
}  // namespace

void Engine::EnsureSearchStopped() {
  if (!search_) return;
  search_->Abort();
  search_->Wait();
}

void Engine::EnsureBackendCreated() {
  if (backend_) return;
  backend_ = BackendManager::Get()->CreateFromParams(options_);
  search_env_->SetBackend(backend_.get());
}

void Engine::SetPosition(const std::string& fen,
                         const std::vector<std::string>& moves) {
  EnsureBackendCreated();
  EnsureSearchStopped();
  search_ = search_env_->CreateSearch(MakeGameState(fen, moves));
}

void Engine::NewGame() { SetPosition(ChessBoard::kStartposFen, {}); }

void Engine::Go(const GoParams& params) {
  if (!search_) NewGame();
  search_->Start(params);
}

void Engine::Stop() {
  if (search_) search_->Stop();
}

}  // namespace lczero
