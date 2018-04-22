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
#include "neural/writer.h"

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

GameInfo::GameResult ComputeGameResult(Node* node) {
  // Checks whether the passed node is an endgame, and if yes, which exactly
  // (win/lose/draw).
  const auto& board = node->board;
  auto valid_moves = board.GenerateValidMoves();
  if (valid_moves.empty()) {
    if (board.IsUnderCheck()) {
      // Checkmate.
      return board.flipped() ? GameInfo::WHITE_WON : GameInfo::BLACK_WON;
    }
    // Stalemate.
    return GameInfo::DRAW;
  }

  if (!board.HasMatingMaterial()) return GameInfo::DRAW;
  if (node->no_capture_ply >= 100) return GameInfo::DRAW;
  if (node->ComputeRepetitions() >= 2) return GameInfo::DRAW;

  return GameInfo::UNDECIDED;
}

}  // namespace

void SelfPlayGame::Play(int white_threads, int black_threads) {
  bool blacks_move = false;

  // Do moves while not end of the game. (And while not abort_)
  while (!abort_) {
    game_result_ = ComputeGameResult(tree_[0]->GetCurrentHead());

    // If endgame, stop.
    if (game_result_ != GameInfo::UNDECIDED) break;

    // Initialize search.
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

    // Do search.
    search_->RunBlocking(blacks_move ? black_threads : white_threads);
    if (abort_) break;

    // Append training data.
    training_data_.push_back(
        tree_[0]->GetCurrentHead()->GetV3TrainingData(GameInfo::UNDECIDED));

    // Add best move to the tree.
    Move move = search_->GetBestMove().first;
    tree_[0]->MakeMove(move);
    if (tree_[0] != tree_[1]) tree_[1]->MakeMove(move);
    blacks_move = !blacks_move;
  }
}

std::vector<Move> SelfPlayGame::GetMoves() const {
  std::vector<Move> moves;
  for (Node* node = tree_[0]->GetCurrentHead();
       node != tree_[0]->GetGameBeginNode(); node = node->parent) {
    moves.push_back(node->move);
  }
  std::reverse(moves.begin(), moves.end());
  return moves;
}

void SelfPlayGame::Abort() {
  std::lock_guard<std::mutex> lock(mutex_);
  abort_ = true;
  if (search_) search_->Abort();
}

void SelfPlayGame::WriteTrainingData(TrainingDataWriter* writer) const {
  bool black_to_move = tree_[0]->GetGameBeginNode()->board.flipped();
  for (auto chunk : training_data_) {
    if (game_result_ == GameInfo::WHITE_WON) {
      chunk.result = black_to_move ? -1 : 1;
    } else if (game_result_ == GameInfo::BLACK_WON) {
      chunk.result = black_to_move ? 1 : -1;
    } else {
      chunk.result = 0;
    }
    writer->WriteChunk(chunk);
    black_to_move = !black_to_move;
  }
}

}  // namespace lczero