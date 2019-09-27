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

#include "selfplay/game.h"
#include <algorithm>

#include "neural/writer.h"

namespace lczero {

namespace {
const OptionId kReuseTreeId{"reuse-tree", "ReuseTree",
                            "Reuse the search tree between moves."};
const OptionId kResignPercentageId{
    "resign-percentage", "ResignPercentage",
    "Resign when win percentage drops below specified value."};
const OptionId kResignWDLStyleId{
    "resign-wdlstyle", "ResignWDLStyle",
    "If set, resign percentage applies to any output state being above "
    "100% minus the percentage instead of winrate being below."};
const OptionId kResignEarliestMoveId{"resign-earliest-move",
                                     "ResignEarliestMove",
                                     "Earliest move that resign is allowed."};
}  // namespace

void SelfPlayGame::PopulateUciParams(OptionsParser* options) {
  options->Add<BoolOption>(kReuseTreeId) = false;
  options->Add<BoolOption>(kResignWDLStyleId) = false;
  options->Add<FloatOption>(kResignPercentageId, 0.0f, 100.0f) = 0.0f;
  options->Add<IntOption>(kResignEarliestMoveId, 0, 1000) = 0;
}

ValueSelfPlayGames::ValueSelfPlayGames(PlayerOptions player1,
                                         PlayerOptions player2,
                                         const std::vector<MoveList>& openings,
                                         SyzygyTablebase* syzygy_tb)
    : options_{player1, player2}, syzygy_tb_(syzygy_tb) {
  trees_.reserve(openings.size());
  for (auto opening : openings) {
    trees_.push_back(std::make_shared<NodeTree>());
    trees_.back()->ResetToPosition(ChessBoard::kStartposFen, {});
    results_.push_back(GameResult::UNDECIDED);

    for (Move m : opening) {
      trees_.back()->MakeMove(m);
    }
  }
}

void ValueSelfPlayGames::Abort() {
  std::lock_guard<std::mutex> lock(mutex_);
  abort_ = true;
}

void ValueSelfPlayGames::Play() {
  while (true) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (abort_) break;
    }
    bool all_done = true;
    bool blacks_move = false;
    for (int i = 0; i < trees_.size(); i++) {
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
            auto tb_side_black = (tree->GetPlyCount() % 2) == 1;
            ProbeState state;
            const WDLScore wdl = syzygy_tb_->probe_wdl(
                tree->GetPositionHistory().Last(), &state);
            // Only fail state means the WDL is wrong, probe_wdl may produce
            // correct result with a stat other than OK.
            if (state != FAIL) {
              if (wdl == WDL_WIN) {
                results_[i] =
                    tb_side_black ? GameResult::BLACK_WON : GameResult::WHITE_WON;
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
          blacks_move = (tree->GetPlyCount() % 2) == 1;
          // Don't break as we need to update result state for everything.
        }
      }
    }
    if (all_done) break;
    const int idx = blacks_move ? 1 : 0;
    auto comp = options_[idx].network->NewComputation();
    for (int i = 0; i < trees_.size(); i++) {
      const auto& tree = trees_[i];
      if (results_[i] != GameResult::UNDECIDED) {
        continue;
      }
      if (((tree->GetPlyCount() % 2) == 1) != blacks_move) continue;
      const auto& board = tree->GetPositionHistory().Last().GetBoard();
      auto legal_moves = board.GenerateLegalMoves();
      tree->GetCurrentHead()->CreateEdges(legal_moves);
      PositionHistory history = tree->GetPositionHistory();
      for (auto edge : tree->GetCurrentHead()->Edges()) {
        history.Append(edge.GetMove());
        if (history.ComputeGameResult() == GameResult::UNDECIDED) {
          auto planes = EncodePositionForNN(history, 8,
                                            FillEmptyHistory::FEN_ONLY);
          comp->AddInput(std::move(planes));
        }
        history.Pop();
      }
    }
    comp->ComputeBlocking();
    int comp_idx = 0;
    for (int i = 0; i < trees_.size(); i++) {
      const auto& tree = trees_[i];
      if (results_[i] != GameResult::UNDECIDED) {
        continue;
      }
      if (((tree->GetPlyCount() % 2) == 1) != blacks_move) continue;
      Move best;
      float max_q = std::numeric_limits<float>::lowest();
      PositionHistory history = tree->GetPositionHistory();
      for (auto edge : tree->GetCurrentHead()->Edges()) {
        history.Append(edge.GetMove());
        auto result = history.ComputeGameResult();
        float q = -1;
        if (result == GameResult::UNDECIDED) {
          // NN eval is for side to move perspective - so if its good, its bad for us.
          q = -comp->GetQVal(comp_idx);
          comp_idx++;
        } else if (result == GameResult::DRAW) {
          q = 0;
        } else {
          // A legal move to a non-drawn terminal without tablebases must be a win.
          q = 1;
        }
        if (q >= max_q) {
          max_q = q;
          best = edge.GetMove(tree->GetPositionHistory().IsBlackToMove());
        }
        history.Pop();
      }
      tree->MakeMove(best);
    }
  }
}

SelfPlayGame::SelfPlayGame(PlayerOptions player1, PlayerOptions player2,
                           bool shared_tree, const MoveList& opening)
    : options_{player1, player2} {
  tree_[0] = std::make_shared<NodeTree>();
  tree_[0]->ResetToPosition(ChessBoard::kStartposFen, {});

  if (shared_tree) {
    tree_[1] = tree_[0];
  } else {
    tree_[1] = std::make_shared<NodeTree>();
    tree_[1]->ResetToPosition(ChessBoard::kStartposFen, {});
  }
  for (Move m : opening) {
    tree_[0]->MakeMove(m);
    if (tree_[0] != tree_[1]) tree_[1]->MakeMove(m);
  }
}

void SelfPlayGame::Play(int white_threads, int black_threads, bool training,
                        bool enable_resign) {
  bool blacks_move = (tree_[0]->GetPlyCount() % 2) == 1;

  // Do moves while not end of the game. (And while not abort_)
  while (!abort_) {
    game_result_ = tree_[0]->GetPositionHistory().ComputeGameResult();

    // If endgame, stop.
    if (game_result_ != GameResult::UNDECIDED) break;

    // Initialize search.
    const int idx = blacks_move ? 1 : 0;
    if (!options_[idx].uci_options->Get<bool>(kReuseTreeId.GetId())) {
      tree_[idx]->TrimTreeAtHead();
    }
    if (options_[idx].search_limits.movetime > -1) {
      options_[idx].search_limits.search_deadline =
          std::chrono::steady_clock::now() +
          std::chrono::milliseconds(options_[idx].search_limits.movetime);
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (abort_) break;
      search_ = std::make_unique<Search>(
          *tree_[idx], options_[idx].network, options_[idx].best_move_callback,
          options_[idx].info_callback, options_[idx].search_limits,
          *options_[idx].uci_options, options_[idx].cache, nullptr);
      // TODO: add Syzygy option for selfplay.
    }

    // Do search.
    search_->RunBlocking(blacks_move ? black_threads : white_threads);
    move_count_++;
    nodes_total_ += search_->GetTotalPlayouts();
    if (abort_) break;

    auto best_eval = search_->GetBestEval();
    if (training) {
      // Append training data. The GameResult is later overwritten.
      auto best_q = best_eval.first;
      auto best_d = best_eval.second;
      training_data_.push_back(tree_[idx]->GetCurrentHead()->GetV4TrainingData(
          GameResult::UNDECIDED, tree_[idx]->GetPositionHistory(),
          search_->GetParams().GetHistoryFill(), best_q, best_d));
    }

    float eval = best_eval.first;
    eval = (eval + 1) / 2;
    if (eval < min_eval_[idx]) min_eval_[idx] = eval;
    const int move_number = tree_[0]->GetPositionHistory().GetLength() / 2 + 1;
    auto best_w = (best_eval.first + 1.0f - best_eval.second) / 2.0f;
    auto best_d = best_eval.second;
    auto best_l = best_w - best_eval.first;
    max_eval_[0] = std::max(max_eval_[0], blacks_move ? best_l : best_w);
    max_eval_[1] = std::max(max_eval_[1], best_d);
    max_eval_[2] = std::max(max_eval_[2], blacks_move ? best_w : best_l);
    if (enable_resign && move_number >= options_[idx].uci_options->Get<int>(
                                            kResignEarliestMoveId.GetId())) {
      const float resignpct =
          options_[idx].uci_options->Get<float>(kResignPercentageId.GetId()) /
          100;
      if (options_[idx].uci_options->Get<bool>(kResignWDLStyleId.GetId())) {
        auto threshold = 1.0f - resignpct;
        if (best_w > threshold) {
          game_result_ =
              blacks_move ? GameResult::BLACK_WON : GameResult::WHITE_WON;
          break;
        }
        if (best_l > threshold) {
          game_result_ =
              blacks_move ? GameResult::WHITE_WON : GameResult::BLACK_WON;
          break;
        }
        if (best_d > threshold) {
          game_result_ = GameResult::DRAW;
          break;
        }
      } else {
        if (eval < resignpct) {  // always false when resignpct == 0
          game_result_ =
              blacks_move ? GameResult::WHITE_WON : GameResult::BLACK_WON;
          break;
        }
      }
    }

    // Add best move to the tree.
    const Move move = search_->GetBestMove().first;
    tree_[0]->MakeMove(move);
    if (tree_[0] != tree_[1]) tree_[1]->MakeMove(move);
    blacks_move = !blacks_move;
  }
}

std::vector<Move> SelfPlayGame::GetMoves() const {
  std::vector<Move> moves;
  bool flip = !tree_[0]->IsBlackToMove();
  for (Node* node = tree_[0]->GetCurrentHead();
       node != tree_[0]->GetGameBeginNode(); node = node->GetParent()) {
    moves.push_back(node->GetParent()->GetEdgeToNode(node)->GetMove(flip));
    flip = !flip;
  }
  std::reverse(moves.begin(), moves.end());
  return moves;
}

float SelfPlayGame::GetWorstEvalForWinnerOrDraw() const {
  // TODO: This assumes both players have the same resign style.
  // Supporting otherwise involves mixing the meaning of worst.
  if (options_[0].uci_options->Get<bool>(kResignWDLStyleId.GetId())) {
    if (game_result_ == GameResult::WHITE_WON) {
      return std::max(max_eval_[1], max_eval_[2]);
    } else if (game_result_ == GameResult::BLACK_WON) {
      return std::max(max_eval_[1], max_eval_[0]);
    } else {
      return std::max(max_eval_[2], max_eval_[0]);
    }
  }
  if (game_result_ == GameResult::WHITE_WON) return min_eval_[0];
  if (game_result_ == GameResult::BLACK_WON) return min_eval_[1];
  return std::min(min_eval_[0], min_eval_[1]);
}

void SelfPlayGame::Abort() {
  std::lock_guard<std::mutex> lock(mutex_);
  abort_ = true;
  if (search_) search_->Abort();
}

void SelfPlayGame::WriteTrainingData(TrainingDataWriter* writer) const {
  assert(!training_data_.empty());
  bool black_to_move =
      tree_[0]->GetPositionHistory().Starting().IsBlackToMove();
  for (auto chunk : training_data_) {
    if (game_result_ == GameResult::WHITE_WON) {
      chunk.result = black_to_move ? -1 : 1;
    } else if (game_result_ == GameResult::BLACK_WON) {
      chunk.result = black_to_move ? 1 : -1;
    } else {
      chunk.result = 0;
    }
    writer->WriteChunk(chunk);
    black_to_move = !black_to_move;
  }
}

}  // namespace lczero
