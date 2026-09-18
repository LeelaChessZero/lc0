/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2024 The LCZero Authors

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

#include "selfplay/multigame.h"

namespace lczero {

void Evaluator::Reset(const PlayerOptions& player) {
  comp_ = player.backend->CreateComputation();
}

class PolicyEvaluator : public Evaluator {
 public:
  void Gather(classic::NodeTree* tree, std::vector<Move> moves) override {
    const auto& history = tree->GetPositionHistory();
    v_.emplace(moves.size());
    comp_->AddInput(
        EvalPosition{
            .pos = history.GetPositions(),
            .legal_moves = moves,
        },
        EvalResultPtr{.p = v_.back()});
  }

  void MakeBestMove(classic::NodeTree* tree) {
    auto v = std::move(v_.front());
    v_.pop();
    size_t best_idx = std::max_element(v.begin(), v.end()) - v.begin();
    for (size_t idx = 0; auto edge : tree->GetCurrentHead()->Edges()) {
      if (idx++ == best_idx) {
        tree->MakeMove(edge.GetMove());
        break;
      }
    }
  }
};

class ValueEvaluator : public Evaluator {
 public:
  void Gather(classic::NodeTree* tree, std::vector<Move> moves) override {
    PositionHistory history = tree->GetPositionHistory();
    v_.emplace(moves.size());
    for (size_t idx = 0; idx < moves.size(); idx++) {
      history.Append(moves[idx]);
      auto result = history.ComputeGameResult();
      if (result == GameResult::UNDECIDED) {
        comp_->AddInput(
            EvalPosition{
                .pos = history.GetPositions(),
                .legal_moves = {},
            },
            EvalResultPtr{.q = &v_.back()[idx]});
      } else if (result == GameResult::DRAW) {
        v_.back()[idx] = 0;
      } else {
        // A legal move to a non-drawn terminal without tablebases must be a
        // win. At this point values are in opponent's perspective.
        v_.back()[idx] = -1;
      }
      history.Pop();
    }
  }

  void MakeBestMove(classic::NodeTree* tree) {
    auto v = std::move(v_.front());
    v_.pop();
    size_t best_idx = std::min_element(v.begin(), v.end()) - v.begin();
    for (size_t idx = 0; auto edge : tree->GetCurrentHead()->Edges()) {
      if (idx++ == best_idx) {
        tree->MakeMove(edge.GetMove());
        break;
      }
    }
  }
};

void Evaluator::Run() { comp_->ComputeBlocking(); }

MultiSelfPlayGames::MultiSelfPlayGames(PlayerOptions player1,
                                       PlayerOptions player2,
                                       const std::vector<Opening>& openings,
                                       SyzygyTablebase* syzygy_tb,
                                       bool use_value)
    : options_{player1, player2}, syzygy_tb_(syzygy_tb) {
  eval_ = use_value
              ? std::unique_ptr<Evaluator>(std::make_unique<ValueEvaluator>())
              : std::unique_ptr<Evaluator>(std::make_unique<PolicyEvaluator>());
  trees_.reserve(openings.size());
  for (auto opening : openings) {
    trees_.push_back(std::make_shared<classic::NodeTree>());
    trees_.back()->ResetToPosition(opening.start_fen, {});
    results_.push_back(GameResult::UNDECIDED);

    for (Move m : opening.moves) {
      if (trees_.back()->IsBlackToMove()) m.Flip();
      trees_.back()->MakeMove(m);
    }
  }
}

void MultiSelfPlayGames::Abort() {
  std::lock_guard<std::mutex> lock(mutex_);
  abort_ = true;
}

void MultiSelfPlayGames::Play() {
  while (true) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (abort_) break;
    }
    bool all_done = true;
    bool blacks_move = false;
    for (size_t i = 0; i < trees_.size(); i++) {
      const auto& tree = trees_[i];
      if (results_[i] == GameResult::UNDECIDED) {
        if (tree->GetPositionHistory().ComputeGameResult() !=
            GameResult::UNDECIDED) {
          results_[i] = tree->GetPositionHistory().ComputeGameResult();
          continue;
        }
        if (syzygy_tb_ != nullptr) {
          auto board = tree->GetPositionHistory().Last().GetBoard();
          if (board.castlings().no_legal_castle() &&
              (board.ours() | board.theirs()).count() <=
                  syzygy_tb_->max_cardinality()) {
            auto tb_side_black = tree->IsBlackToMove();
            ProbeState state;
            const WDLScore wdl = syzygy_tb_->probe_wdl(
                tree->GetPositionHistory().Last(), &state);
            // Only fail state means the WDL is wrong, probe_wdl may produce
            // correct result with a stat other than OK.
            if (state != FAIL) {
              if (wdl == WDL_WIN) {
                results_[i] = tb_side_black ? GameResult::BLACK_WON
                                            : GameResult::WHITE_WON;
              } else if (wdl == WDL_LOSS) {
                results_[i] = tb_side_black ? GameResult::WHITE_WON
                                            : GameResult::BLACK_WON;
              } else {  // Cursed wins and blessed losses count as draws.
                results_[i] = GameResult::DRAW;
              }
              continue;
            }
          }
        }
        if (all_done) {
          all_done = false;
          blacks_move = tree->IsBlackToMove();
          // Don't break as we need to update result state for everything.
        }
      }
    }
    if (all_done) break;
    const int idx = blacks_move ? 1 : 0;
    eval_->Reset(options_[idx]);
    for (size_t i = 0; i < trees_.size(); i++) {
      const auto& tree = trees_[i];
      if (results_[i] != GameResult::UNDECIDED) {
        continue;
      }
      if (tree->IsBlackToMove() != blacks_move) continue;
      const auto& board = tree->GetPositionHistory().Last().GetBoard();
      auto legal_moves = board.GenerateLegalMoves();
      tree->GetCurrentHead()->CreateEdges(legal_moves);
      eval_->Gather(tree.get(), std::move(legal_moves));
    }
    eval_->Run();
    for (size_t i = 0; i < trees_.size(); i++) {
      const auto& tree = trees_[i];
      if (results_[i] != GameResult::UNDECIDED) {
        continue;
      }
      if (tree->IsBlackToMove() != blacks_move) continue;
      eval_->MakeBestMove(tree.get());
    }
  }
}

}  // namespace lczero
