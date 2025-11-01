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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "chess/bitboard.h"
#include "chess/position.h"
#include "utils/exception.h"

namespace lczero {

// Is sent when search decides on the best move.
struct BestMoveInfo {
  BestMoveInfo(Move bestmove, Move ponder = Move{})
      : bestmove(bestmove), ponder(ponder) {}
  Move bestmove;
  Move ponder;
  // Those are extensions and not really UCI protocol.
  // 1 if it's "player1", 2 if it's "player2"
  int player = -1;
  // Index of the game in the tournament (0-based).
  int game_id = -1;
  // The color of the player, if known.
  std::optional<bool> is_black;
};

// Is sent during the search.
struct ThinkingInfo {
  // Full depth.
  int depth = -1;
  // Maximum depth.
  int seldepth = -1;
  // Time since start of thinking.
  int64_t time = -1;
  // Nodes visited.
  int64_t nodes = -1;
  // Nodes per second.
  int nps = -1;
  // Evaluations per second.
  int eps = -1;
  // Hash fullness * 1000
  int hashfull = -1;
  // Moves to mate.
  std::optional<int> mate = std::nullopt;
  // Win in centipawns.
  std::optional<int> score = std::nullopt;
  // Win/Draw/Lose probability * 1000.
  struct WDL {
    int w;
    int d;
    int l;
  };
  std::optional<WDL> wdl = std::nullopt;
  // Number of successful TB probes (not the same as playouts ending in TB hit).
  int tb_hits = -1;
  // Best line found. Moves are from perspective of white player.
  std::vector<Move> pv = {};
  // Multipv index.
  int multipv = -1;
  // Freeform comment.
  std::string comment = "";

  // Those are extensions and not really UCI protocol.
  // 1 if it's "player1", 2 if it's "player2"
  int player = -1;
  // Index of the game in the tournament (0-based).
  int game_id = -1;
  // The color of the player, if known.
  std::optional<bool> is_black = std::nullopt;
  // Moves left
  std::optional<int> moves_left = std::nullopt;
};

// Is sent when a single game is finished.
struct GameInfo {
  // Game result.
  GameResult game_result = GameResult::UNDECIDED;
  // Name of the file with training data.
  std::string training_filename;
  // Initial fen of the game.
  std::string initial_fen;
  // Game moves.
  std::vector<Move> moves;
  // Ply within moves that the game actually started.
  int play_start_ply;
  // Index of the game in the tournament (0-based).
  int game_id = -1;
  // The color of the player1, if known.
  std::optional<bool> is_black;
  // Minimum resign threshold which would have resulted in a false positive
  // if resign had of been enabled.
  // Only provided if the game wasn't played with resign enabled.
  std::optional<float> min_false_positive_threshold;

  using Callback = std::function<void(const GameInfo&)>;
};

// Is sent in the end of tournament and also during the tournament.
struct TournamentInfo {
  // Did tournament finish, so those results are final.
  bool finished = false;

  // Player1's [win/draw/lose] as [white/black].
  // e.g. results[2][1] is how many times player 1 lost as black.
  int results[3][2] = {{0, 0}, {0, 0}, {0, 0}};
  int move_count_ = 0;
  uint64_t nodes_total_ = 0;

  using Callback = std::function<void(const TournamentInfo&)>;
};

// A class which knows how to output UCI responses.
class UciResponder {
 public:
  virtual ~UciResponder() = default;
  virtual void OutputBestMove(BestMoveInfo* info) = 0;
  virtual void OutputThinkingInfo(std::vector<ThinkingInfo>* infos) = 0;
};

// The responder which forwards the output to another responder, with
// observer-like subscription model.
class UciResponderForwarder : public UciResponder {
 public:
  void OutputBestMove(BestMoveInfo* info) override {
    if (wrapped_) wrapped_->OutputBestMove(info);
  }
  void OutputThinkingInfo(std::vector<ThinkingInfo>* infos) override {
    if (wrapped_) wrapped_->OutputThinkingInfo(infos);
  }
  void Register(UciResponder* wrapped) {
    if (wrapped_) {
      throw Exception("UciResponderForwarder already has a wrapped responder");
    }
    wrapped_ = wrapped;
  }
  void Unregister(UciResponder* wrapped) {
    if (wrapped_ != wrapped) {
      throw Exception(
          "UciResponderForwarder doesn't have this wrapped responder");
    }
    wrapped_ = nullptr;
  }

 private:
  UciResponder* wrapped_ = nullptr;
};

// The responder which calls callbacks. Used for easier transition from old
// code.
class CallbackUciResponder : public UciResponder {
 public:
  using ThinkingCallback =
      std::function<void(const std::vector<ThinkingInfo>&)>;
  using BestMoveCallback = std::function<void(const BestMoveInfo&)>;

  CallbackUciResponder(BestMoveCallback bestmove, ThinkingCallback info)
      : bestmove_callback_(bestmove), info_callback_(info) {}

  void OutputBestMove(BestMoveInfo* info) { bestmove_callback_(*info); }
  void OutputThinkingInfo(std::vector<ThinkingInfo>* infos) {
    info_callback_(*infos);
  }

 private:
  const BestMoveCallback bestmove_callback_;
  const ThinkingCallback info_callback_;
};

// The responnder which doesn't own the parent. Used to transition from old code
// where we need to create a copy.
class NonOwningUciRespondForwarder : public UciResponder {
 public:
  NonOwningUciRespondForwarder(UciResponder* parent) : parent_(parent) {}
  virtual void OutputBestMove(BestMoveInfo* info) {
    parent_->OutputBestMove(info);
  }
  virtual void OutputThinkingInfo(std::vector<ThinkingInfo>* infos) {
    parent_->OutputThinkingInfo(infos);
  }

 private:
  UciResponder* const parent_;
};

// Base class for uci response transformations.
class TransformingUciResponder : public UciResponder {
 public:
  TransformingUciResponder(std::unique_ptr<UciResponder> parent)
      : parent_(std::move(parent)) {}

  virtual void TransformBestMove(BestMoveInfo*) {}
  virtual void TransformThinkingInfo(std::vector<ThinkingInfo>*) {}

 private:
  virtual void OutputBestMove(BestMoveInfo* info) {
    TransformBestMove(info);
    parent_->OutputBestMove(info);
  }
  virtual void OutputThinkingInfo(std::vector<ThinkingInfo>* infos) {
    TransformThinkingInfo(infos);
    parent_->OutputThinkingInfo(infos);
  }
  std::unique_ptr<UciResponder> parent_;
};

}  // namespace lczero
