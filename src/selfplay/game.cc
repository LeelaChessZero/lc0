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

#include "selfplay/game.h"

namespace lczero {

SelfPlayGame::SelfPlayGame(PlayerOptions player1, PlayerOptions player2,
                           bool shared_tree, NodePool* node_pool)
    : options_{player1, player2}, node_pool_(node_pool) {
  tree_[0] = std::make_shared<NodeTree>(node_pool_);
  tree_[0]->ResetToPosition(ChessBoard::kStartingFen, {});

  if (shared_tree) {
    tree_[1] = tree_[0];
  } else {
    tree_[1] = std::make_shared<NodeTree>(node_pool_);
    tree_[1]->ResetToPosition(ChessBoard::kStartingFen, {});
  }
}

namespace {

SelfPlayGame::GameResult ComputeGameResult(Node* node) {
  const auto& board = node->board;
  auto valid_moves = board.GenerateValidMoves();
  if (valid_moves.empty()) {
    if (board.IsUnderCheck()) {
      // Checkmate.
      return board.flipped() ? SelfPlayGame::WHITE_WON
                             : SelfPlayGame::BLACK_WON;
    }
    // Stalemate.
    return SelfPlayGame::DRAW;
  }

  if (!board.HasMatingMaterial()) return SelfPlayGame::DRAW;
  if (node->no_capture_ply >= 100) return SelfPlayGame::DRAW;
  if (node->ComputeRepetitions() >= 2) return SelfPlayGame::DRAW;

  return SelfPlayGame::UNDECIDED;
}

}  // namespace

void SelfPlayGame::Play() {
  bool blacks_move = false;

  while (!abort_) {
    game_result_ = ComputeGameResult(tree_[0]->GetCurrentHead());
    if (game_result_ != UNDECIDED) break;

    const int idx = blacks_move ? 1 : 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (abort_) break;
      search_ = std::make_unique<Search>(
          tree_[idx]->GetCurrentHead(), node_pool_, options_[idx].network,
          options_[idx].best_move_callback, options_[idx].info_callback,
          options_[idx].search_limits, *options_[idx].uci_options,
          options_[idx].cache);
    }

    search_->RunSingleThreaded();
    if (abort_) break;

    Move move = search_->GetBestMove().first;
    tree_[0]->MakeMove(move);
    if (tree_[0] != tree_[1]) tree_[1]->MakeMove(move);
  }
}

void SelfPlayGame::Abort() {
  std::lock_guard<std::mutex> lock(mutex_);
  abort_ = true;
  if (search_) search_->Abort();
}

}  // namespace lczero