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

#include "search/instamove/instamove.h"

#include <algorithm>
#include <vector>

#include "chess/gamestate.h"
#include "neural/backend.h"
#include "search/register.h"
#include "search/search.h"

namespace lczero {
namespace instamove {

class PolicyHeadSearch : public InstamoveSearch {
 public:
  PolicyHeadSearch(const SearchContext& context, const GameState& game_state)
      : InstamoveSearch(context), game_state_(game_state) {}

  Move GetBestMove() final {
    const std::vector<Position> positions = game_state_.GetPositions();
    MoveList legal_moves = positions.back().GetBoard().GenerateLegalMoves();
    std::vector<EvalResult> res = backend()->EvaluateBatch(
        std::vector<EvalPosition>{EvalPosition{positions, legal_moves}});
    const size_t best_move_idx =
        std::max_element(res[0].p.begin(), res[0].p.end()) - res[0].p.begin();
    return legal_moves[best_move_idx];
  }

 private:
  const GameState game_state_;
};

class PolicyHeadFactory : public SearchFactory {
  std::string_view GetName() const override { return "policyhead"; }
  std::unique_ptr<SearchEnvironment> CreateEnvironment(
      UciResponder* responder, const OptionsDict* options) const override {
    return std::make_unique<InstamoveEnvironment<PolicyHeadSearch>>(responder,
                                                                    options);
  }
};

REGISTER_SEARCH(PolicyHeadFactory);

}  // namespace instamove
}  // namespace lczero