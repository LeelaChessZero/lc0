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
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

#include "chess/gamestate.h"
#include "chess/uciloop.h"
#include "neural/backend.h"
#include "neural/batchsplit.h"
#include "search/classic/params.h"
#include "search/common/temperature.h"
#include "search/register.h"
#include "search/search.h"
#include "utils/random.h"

namespace lczero {
namespace {

class InstamoveSearch : public SearchBase {
 public:
  using SearchBase::SearchBase;

 private:
  virtual Move GetBestMove(const GameState& game_state) = 0;

  void SetPosition(const GameState& game_state) final { game_state_ = game_state; }

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

class PolicyHeadTempSearch : public InstamoveSearch {
 public:
  PolicyHeadTempSearch(UciResponder* responder, TempParams params)
      : InstamoveSearch(responder), params_(params) {}

 private:
  Move GetBestMove(const GameState& game_state) final {
    const std::vector<Position> positions = game_state.GetPositions();
    MoveList legal_moves = positions.back().GetBoard().GenerateLegalMoves();
    std::vector<EvalResult> res = backend_->EvaluateBatch(
        std::vector<EvalPosition>{EvalPosition{positions, legal_moves}});
    const size_t best_move_idx =
        std::max_element(res[0].p.begin(), res[0].p.end()) - res[0].p.begin();

    std::vector<ThinkingInfo> infos = {{
        .depth = 1,
        .seldepth = 1,
        .nodes = 1,
        .score = 90 * std::tan(1.5637541897 * res[0].q),
        .wdl = ThinkingInfo::WDL{
            static_cast<int>(std::round(500 * (1 + res[0].q - res[0].d))),
            static_cast<int>(std::round(1000 * res[0].d)),
            static_cast<int>(std::round(500 * (1 - res[0].q - res[0].d)))}},
    }};
    uci_responder_->OutputThinkingInfo(&infos);

    Move best_move;
    const int fullmove = positions.back().GetGamePly() / 2 + 1;
    const double tau = EffectiveTau(params_, fullmove);
    if (tau > 0.0 && legal_moves.size() > 1) {
      std::vector<double> policy(res[0].p.begin(), res[0].p.end());
      TempParams p = params_;
      p.visit_offset = 0.0;
      int idx = SampleWithTemperature(policy, std::span<const double>(), p, tau,
                                      Random::Get(),
                                      static_cast<int>(best_move_idx));
      best_move = legal_moves[idx];
    } else {
      best_move = legal_moves[best_move_idx];
    }
    return best_move;
  }

  TempParams params_;
};

class PolicyHeadTempFactory : public SearchFactory {
  std::string_view GetName() const override { return "policyhead_temp"; }
  void PopulateParams(OptionsParser* parser) const override {
    classic::BaseSearchParams::Populate(parser);
  }
  std::unique_ptr<SearchBase> CreateSearch(UciResponder* responder,
                                           const OptionsDict* options) const override {
    classic::BaseSearchParams base_params(*options);
    TempParams params{
        .temperature = base_params.GetTemperature(),
        .temp_decay_moves = base_params.GetTempDecayMoves(),
        .temp_cutoff_move = base_params.GetTemperatureCutoffMove(),
        .temp_endgame = base_params.GetTemperatureEndgame(),
        .value_cutoff = base_params.GetTemperatureWinpctCutoff(),
        .visit_offset = base_params.GetTemperatureVisitOffset(),
    };
    return std::make_unique<PolicyHeadTempSearch>(responder, params);
  }
};

REGISTER_SEARCH(PolicyHeadTempFactory)

}  // namespace
}  // namespace lczero

