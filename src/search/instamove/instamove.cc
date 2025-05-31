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

#include <algorithm>
#include <cmath>
#include <vector>

#include "chess/gamestate.h"
#include "chess/uciloop.h"
#include "neural/backend.h"
#include "neural/batchsplit.h"
#include "search/register.h"
#include "search/search.h"

namespace lczero {
namespace {

class InstamoveSearch : public SearchBase {
 public:
  using SearchBase::SearchBase;

 private:
  virtual Move GetBestMove(const GameState& game_state) = 0;

  void SetPosition(const GameState& game_state) final {
    game_state_ = game_state;
  }

  void StartSearch(const GoParams& go_params) final {
    responded_bestmove_.store(false, std::memory_order_relaxed);
    bestmove_ = GetBestMove(game_state_);
    if (!go_params.infinite && !go_params.ponder) RespondBestMove();
  }
  void WaitSearch() final {
    while (!responded_bestmove_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  void StopSearch() final { RespondBestMove(); }
  void AbortSearch() final { responded_bestmove_.store(true); }
  void RespondBestMove() {
    if (responded_bestmove_.exchange(true)) return;
    BestMoveInfo info{bestmove_};
    // TODO Remove this when move will be encoded from white perspective.
    if (game_state_.CurrentPosition().IsBlackToMove()) {
      info.bestmove.Flip();
    } else if (!info.ponder.is_null()) {
      info.ponder.Flip();
    }
    uci_responder_->OutputBestMove(&info);
  }

  void SetBackend(Backend* backend) override {
    batchsplit_backend_ = CreateBatchSplitingBackend(backend);
    backend_ = batchsplit_backend_.get();
  }
  void StartClock() final {}

  Move bestmove_;
  std::atomic<bool> responded_bestmove_{false};
  std::unique_ptr<Backend> batchsplit_backend_;
  GameState game_state_;
};

class PolicyHeadSearch : public InstamoveSearch {
 public:
  using InstamoveSearch::InstamoveSearch;

  Move GetBestMove(const GameState& game_state) final {
    const std::vector<Position> positions = game_state.GetPositions();
    MoveList legal_moves = positions.back().GetBoard().GenerateLegalMoves();
    std::vector<EvalResult> res = backend_->EvaluateBatch(
        std::vector<EvalPosition>{EvalPosition{positions, legal_moves}});
    const size_t best_move_idx =
        std::max_element(res[0].p.begin(), res[0].p.end()) - res[0].p.begin();
    Move best_move = legal_moves[best_move_idx];
    return best_move;
  }
};

class ValueHeadSearch : public InstamoveSearch {
 public:
  using InstamoveSearch::InstamoveSearch;
  Move GetBestMove(const GameState& game_state) final {
    std::unique_ptr<BackendComputation> computation =
        backend_->CreateComputation();

    PositionHistory history(game_state.GetPositions());
    const ChessBoard& board = history.Last().GetBoard();
    const std::vector<Move> legal_moves = board.GenerateLegalMoves();
    std::vector<EvalResult> results(legal_moves.size());

    for (size_t i = 0; i < legal_moves.size(); i++) {
      Move move = legal_moves[i];
      history.Append(move);
      switch (history.ComputeGameResult()) {
        case GameResult::UNDECIDED:
          computation->AddInput(
              EvalPosition{history.GetPositions(), {}},
              EvalResultPtr{.q = &results[i].q, .d = &results[i].d});
          break;
        case GameResult::DRAW:
          results[i].q = 0;
          results[i].d = 1;
          break;
        default:
          // A legal move to a non-drawn terminal without tablebases must be a
          // win.
          results[i].q = -1;
          results[i].d = 0;
      }
      history.Pop();
    }

    computation->ComputeBlocking();

    const size_t best_idx =
        std::min_element(results.begin(), results.end(),
                         [](const EvalResult& a, const EvalResult& b) {
                           return a.q < b.q;
                         }) -
        results.begin();

    std::vector<ThinkingInfo> infos = {{
        .depth = 1,
        .seldepth = 1,
        .nodes = static_cast<int64_t>(legal_moves.size()),
        .score = 90 * std::tan(1.5637541897 * results[best_idx].q),
        .wdl =
            ThinkingInfo::WDL{
                static_cast<int>(std::round(
                    500 * (1 + results[best_idx].q - results[best_idx].d))),
                static_cast<int>(std::round(1000 * results[best_idx].d)),
                static_cast<int>(std::round(
                    500 * (1 - results[best_idx].q - results[best_idx].d)))},
    }};
    uci_responder_->OutputThinkingInfo(&infos);
    Move best_move = legal_moves[best_idx];
    return best_move;
  }
};

class PolicyHeadFactory : public SearchFactory {
  std::string_view GetName() const override { return "policyhead"; }
  std::unique_ptr<SearchBase> CreateSearch(UciResponder* responder,
                                           const OptionsDict*) const override {
    return std::make_unique<PolicyHeadSearch>(responder);
  }
};

class ValueHeadFactory : public SearchFactory {
  std::string_view GetName() const override { return "valuehead"; }
  std::unique_ptr<SearchBase> CreateSearch(UciResponder* responder,
                                           const OptionsDict*) const override {
    return std::make_unique<ValueHeadSearch>(responder);
  }
};

REGISTER_SEARCH(PolicyHeadFactory)
REGISTER_SEARCH(ValueHeadFactory)

}  // namespace
}  // namespace lczero