/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

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

#include "mcts/search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <thread>

#include "mcts/node.h"
#include "neural/cache.h"
#include "neural/encoder.h"
#include "utils/fastmath.h"
#include "utils/random.h"

namespace lczero {

namespace {
// Maximum delay between outputting "uci info" when nothing interesting happens.
const int kUciInfoMinimumFrequencyMs = 5000;
}  // namespace

Search::Search(const NodeTree& tree, Network* network,
               std::unique_ptr<UciResponder> uci_responder,
               const MoveList& searchmoves,
               std::chrono::steady_clock::time_point start_time,
               std::unique_ptr<SearchStopper> stopper, bool infinite,
               const OptionsDict& options, NNCache* cache,
               SyzygyTablebase* syzygy_tb)
    : ok_to_respond_bestmove_(!infinite),
      stopper_(std::move(stopper)),
      root_node_(tree.GetCurrentHead()),
      cache_(cache),
      syzygy_tb_(syzygy_tb),
      played_history_(tree.GetPositionHistory()),
      network_(network),
      searchmoves_(searchmoves),
      start_time_(start_time),
      initial_visits_(root_node_->GetN()),
      uci_responder_(std::move(uci_responder)),
      params_(options) {
  if (params_.GetMaxConcurrentSearchers() != 0) {
    pending_searchers_.store(params_.GetMaxConcurrentSearchers(),
                             std::memory_order_release);
  }
}

namespace {
void ApplyDirichletNoise(Node* node, float eps, double alpha) {
  float total = 0;
  std::vector<float> noise;

  for (int i = 0; i < node->GetNumEdges(); ++i) {
    float eta = Random::Get().GetGamma(alpha, 1.0);
    noise.emplace_back(eta);
    total += eta;
  }

  if (total < std::numeric_limits<float>::min()) return;

  int noise_idx = 0;
  for (const auto& child : node->Edges()) {
    auto* edge = child.edge();
    edge->SetP(edge->GetP() * (1 - eps) + eps * noise[noise_idx++] / total);
  }
}
}  // namespace

void Search::SendUciInfo() REQUIRES(nodes_mutex_) {
  const auto max_pv = params_.GetMultiPv();
  const auto edges = GetBestChildrenNoTemperature(root_node_, max_pv, 0);
  const auto score_type = params_.GetScoreType();
  const auto per_pv_counters = params_.GetPerPvCounters();
  const auto display_cache_usage = params_.GetDisplayCacheUsage();
  const auto draw_score = GetDrawScore(false);

  std::vector<ThinkingInfo> uci_infos;

  // Info common for all multipv variants.
  ThinkingInfo common_info;
  common_info.depth = cum_depth_ / (total_playouts_ ? total_playouts_ : 1);
  common_info.seldepth = max_depth_;
  common_info.time = GetTimeSinceStart();
  if (!per_pv_counters) {
    common_info.nodes = total_playouts_ + initial_visits_;
  }
  if (display_cache_usage) {
    common_info.hashfull =
        cache_->GetSize() * 1000LL / std::max(cache_->GetCapacity(), 1);
  }
  if (nps_start_time_) {
    const auto time_since_first_batch_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - *nps_start_time_)
            .count();
    if (time_since_first_batch_ms > 0) {
      common_info.nps = total_playouts_ * 1000 / time_since_first_batch_ms;
    }
  }
  common_info.tb_hits = tb_hits_.load(std::memory_order_acquire);

  int multipv = 0;
  const auto default_q = -root_node_->GetQ(-draw_score);
  const auto default_wl = -root_node_->GetWL();
  const auto default_d = root_node_->GetD();
  for (const auto& edge : edges) {
    ++multipv;
    uci_infos.emplace_back(common_info);
    auto& uci_info = uci_infos.back();
    const auto wl = edge.GetWL(default_wl);
    const auto d = edge.GetD(default_d);
    const int w = static_cast<int>(std::round(500.0 * (1.0 + wl - d)));
    const auto q = edge.GetQ(default_q, draw_score, /* logit_q= */ false);
    if (edge.IsTerminal() && wl != 0.0f) {
      uci_info.mate = std::copysign(
          std::round(edge.GetM(0.0f)) / 2 + (edge.IsTbTerminal() ? 101 : 1),
          wl);
    } else if (score_type == "centipawn_with_drawscore") {
      uci_info.score = 90 * tan(1.5637541897 * q);
    } else if (score_type == "centipawn") {
      uci_info.score = 90 * tan(1.5637541897 * wl);
    } else if (score_type == "centipawn_2019") {
      uci_info.score = 295 * wl / (1 - 0.976953126 * std::pow(wl, 14));
    } else if (score_type == "centipawn_2018") {
      uci_info.score = 290.680623072 * tan(1.548090806 * wl);
    } else if (score_type == "win_percentage") {
      uci_info.score = wl * 5000 + 5000;
    } else if (score_type == "Q") {
      uci_info.score = q * 10000;
    } else if (score_type == "W-L") {
      uci_info.score = wl * 10000;
    }
    const int l = static_cast<int>(std::round(500.0 * (1.0 - wl - d)));
    // Using 1000-w-l instead of 1000*d for D score so that W+D+L add up to
    // 1000.0.
    uci_info.wdl = ThinkingInfo::WDL{w, 1000 - w - l, l};
    if (max_pv > 1) uci_info.multipv = multipv;
    if (per_pv_counters) uci_info.nodes = edge.GetN();
    bool flip = played_history_.IsBlackToMove();
    int depth = 0;
    for (auto iter = edge; iter;
         iter = GetBestChildNoTemperature(iter.node(), depth), flip = !flip) {
      uci_info.pv.push_back(iter.GetMove(flip));
      if (!iter.node()) break;  // Last edge was dangling, cannot continue.
      depth += 1;
    }
  }

  if (!uci_infos.empty()) last_outputted_uci_info_ = uci_infos.front();
  if (current_best_edge_ && !edges.empty()) {
    last_outputted_info_edge_ = current_best_edge_.edge();
  }

  uci_responder_->OutputThinkingInfo(&uci_infos);
}

// Decides whether anything important changed in stats and new info should be
// shown to a user.
void Search::MaybeOutputInfo() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  if (!bestmove_is_sent_ && current_best_edge_ &&
      (current_best_edge_.edge() != last_outputted_info_edge_ ||
       last_outputted_uci_info_.depth !=
           static_cast<int>(cum_depth_ /
                            (total_playouts_ ? total_playouts_ : 1)) ||
       last_outputted_uci_info_.seldepth != max_depth_ ||
       last_outputted_uci_info_.time + kUciInfoMinimumFrequencyMs <
           GetTimeSinceStart())) {
    SendUciInfo();
    if (params_.GetLogLiveStats()) {
      SendMovesStats();
    }
    if (stop_.load(std::memory_order_acquire) && !ok_to_respond_bestmove_) {
      std::vector<ThinkingInfo> info(1);
      info.back().comment =
          "WARNING: Search has reached limit and does not make any progress.";
      uci_responder_->OutputThinkingInfo(&info);
    }
  }
}

int64_t Search::GetTimeSinceStart() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_time_)
      .count();
}

int64_t Search::GetTimeSinceFirstBatch() const {
  if (!nps_start_time_) return 0;
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - *nps_start_time_)
      .count();
}

// Root is depth 0, i.e. even depth.
float Search::GetDrawScore(bool is_odd_depth) const {
  return (is_odd_depth ? params_.GetOpponentDrawScore()
                       : params_.GetSidetomoveDrawScore()) +
         (is_odd_depth == played_history_.IsBlackToMove()
              ? params_.GetWhiteDrawDelta()
              : params_.GetBlackDrawDelta());
}

namespace {
inline float GetFpu(const SearchParams& params, Node* node, bool is_root_node,
                    float draw_score) {
  const auto value = params.GetFpuValue(is_root_node);
  return params.GetFpuAbsolute(is_root_node)
             ? value
             : -node->GetQ(-draw_score) -
                   value * std::sqrt(node->GetVisitedPolicy());
}

inline float ComputeCpuct(const SearchParams& params, uint32_t N,
                          bool is_root_node) {
  const float init = params.GetCpuct(is_root_node);
  const float k = params.GetCpuctFactor(is_root_node);
  const float base = params.GetCpuctBase(is_root_node);
  return init + (k ? k * FastLog((N + base) / base) : 0.0f);
}
}  // namespace

std::vector<std::string> Search::GetVerboseStats(Node* node) const {
  assert(node == root_node_ || node->GetParent() == root_node_);
  const bool is_root = (node == root_node_);
  const bool is_odd_depth = !is_root;
  const bool is_black_to_move = (played_history_.IsBlackToMove() == is_root);
  const float draw_score = GetDrawScore(is_odd_depth);
  const float fpu = GetFpu(params_, node, is_root, draw_score);
  const float cpuct = ComputeCpuct(params_, node->GetN(), is_root);
  const float U_coeff =
      cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
  const float childVisits = node->GetChildrenVisits();
  const float shift = params_.GetPolicyDecayShift();
  const float slope = params_.GetPolicyDecaySlope();
  const bool logit_q = params_.GetLogitQ();
  const float m_slope = params_.GetMovesLeftSlope();
  const float m_cap = params_.GetMovesLeftMaxEffect();
  const float a = params_.GetMovesLeftConstantFactor();
  const float b = params_.GetMovesLeftScaledFactor();
  const float c = params_.GetMovesLeftQuadraticFactor();
  const bool do_moves_left_adjustment =
      network_->GetCapabilities().moves_left !=
          pblczero::NetworkFormat::MOVES_LEFT_NONE &&
      (std::abs(node->GetQ(0.0f)) > params_.GetMovesLeftThreshold());
  std::vector<EdgeAndNode> edges;
  for (const auto& edge : node->Edges()) edges.push_back(edge);

  std::sort(
      edges.begin(), edges.end(),
      [&fpu, &U_coeff, &childVisits, &shift, &slope, &logit_q,
       &draw_score](EdgeAndNode a, EdgeAndNode b) {
        return std::forward_as_tuple(
                   a.GetN(),
                   a.GetQ(fpu, draw_score, logit_q) +
                   a.GetU(U_coeff, childVisits, shift, slope)) <
               std::forward_as_tuple(
                   b.GetN(),
                   b.GetQ(fpu, draw_score, logit_q) +
                   b.GetU(U_coeff, childVisits, shift, slope));
      });

  std::vector<std::string> infos;
  for (const auto& edge : edges) {
    std::ostringstream oss;
    oss << std::fixed;

    oss << std::left << std::setw(5)
        << edge.GetMove(is_black_to_move).as_string();
    
    float Q = edge.GetQ(fpu, draw_score, logit_q);
    float M_effect = do_moves_left_adjustment 
        ? (std::clamp(m_slope * edge.GetM(0.0f), -m_cap, m_cap) *
            std::copysign(1.0f, -Q) * (a + b * std::abs(Q) + c * Q * Q)) 
        : 0.0f;

    // TODO: should this be displaying transformed index?
    oss << " (" << std::setw(4) << edge.GetMove().as_nn_index(0) << ")";

    oss << " N: " << std::right << std::setw(7) << edge.GetN() << " (+"
        << std::setw(2) << edge.GetNInFlight() << ") ";

    oss << "(P: " << std::setw(5) << std::setprecision(2) << edge.GetP() * 100
        << "%) ";

    // Default value here assumes user knows to ignore this field when N is 0.
    oss << "(WL: " << std::setw(8) << std::setprecision(5) << edge.GetWL(0.0f)
        << ") ";

    // Default value here assumes user knows to ignore this field when N is 0.
    oss << "(D: " << std::setw(6) << std::setprecision(3) << edge.GetD(0.0f)
        << ") ";

    // Default value here assumes user knows to ignore this field when N is 0.
    oss << "(M: " << std::setw(4) << std::setprecision(1) << edge.GetM(0.0f)
        << ") ";

    oss << "(Q: " << std::setw(8) << std::setprecision(5)
        << edge.GetQ(fpu, draw_score, /* logit_q= */ false) << ") ";

    oss << "(U: " << std::setw(6) << std::setprecision(5)
        << edge.GetU(U_coeff, childVisits, shift, slope) << ") ";

    oss << "(S: " << std::setw(8) << std::setprecision(5)
        << Q + edge.GetU(U_coeff, childVisits, shift, slope) + M_effect << ") ";

    oss << "(V: ";
    std::optional<float> v;
    if (edge.IsTerminal()) {
      v = edge.node()->GetQ(draw_score);
    } else {
      NNCacheLock nneval = GetCachedNNEval(edge.node());
      if (nneval) v = -nneval->q;
    }
    if (v) {
      oss << std::setw(7) << std::setprecision(4) << *v;
    } else {
      oss << " -.----";
    }
    oss << ") ";

    const auto [edge_lower, edge_upper] = edge.GetBounds();
    oss << (edge_lower == edge_upper
                ? "(T) "
                : edge_lower == GameResult::DRAW &&
                          edge_upper == GameResult::WHITE_WON
                      ? "(W) "
                      : edge_lower == GameResult::BLACK_WON &&
                                edge_upper == GameResult::DRAW
                            ? "(L) "
                            : "");
    infos.emplace_back(oss.str());
  }
  return infos;
}

void Search::SendMovesStats() const REQUIRES(counters_mutex_) {
  auto move_stats = GetVerboseStats(root_node_);

  if (params_.GetVerboseStats()) {
    std::vector<ThinkingInfo> infos;
    std::transform(move_stats.begin(), move_stats.end(),
                   std::back_inserter(infos), [](const std::string& line) {
                     ThinkingInfo info;
                     info.comment = line;
                     return info;
                   });
    uci_responder_->OutputThinkingInfo(&infos);
  } else {
    LOGFILE << "=== Move stats:";
    for (const auto& line : move_stats) LOGFILE << line;
  }
  if (final_bestmove_.HasNode()) {
    LOGFILE
        << "--- Opponent moves after: "
        << final_bestmove_.GetMove(played_history_.IsBlackToMove()).as_string();
    for (const auto& line : GetVerboseStats(final_bestmove_.node())) {
      LOGFILE << line;
    }
  }
}

NNCacheLock Search::GetCachedNNEval(Node* node) const {
  if (!node) return {};

  std::vector<Move> moves;
  for (; node != root_node_; node = node->GetParent()) {
    moves.push_back(node->GetOwnEdge()->GetMove());
  }
  PositionHistory history(played_history_);
  for (auto iter = moves.rbegin(), end = moves.rend(); iter != end; ++iter) {
    history.Append(*iter);
  }
  const auto hash = history.HashLast(params_.GetCacheHistoryLength() + 1);
  NNCacheLock nneval(cache_, hash);
  return nneval;
}

void Search::MaybeTriggerStop(const IterationStats& stats,
                              StoppersHints* hints) {
  hints->Reset();
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  // Already responded bestmove, nothing to do here.
  if (bestmove_is_sent_) return;
  // Don't stop when the root node is not yet expanded.
  if (total_playouts_ + initial_visits_ == 0) return;

  if (!stop_.load(std::memory_order_acquire) || !ok_to_respond_bestmove_) {
    if (stopper_->ShouldStop(stats, hints)) FireStopInternal();
  }

  // If we are the first to see that stop is needed.
  if (stop_.load(std::memory_order_acquire) && ok_to_respond_bestmove_ &&
      !bestmove_is_sent_) {
    SendUciInfo();
    EnsureBestMoveKnown();
    SendMovesStats();
    BestMoveInfo info(
        final_bestmove_.GetMove(played_history_.IsBlackToMove()),
        final_pondermove_.GetMove(!played_history_.IsBlackToMove()));
    uci_responder_->OutputBestMove(&info);
    stopper_->OnSearchDone(stats);
    bestmove_is_sent_ = true;
    current_best_edge_ = EdgeAndNode();
  }

  // Use a 0 visit cancel score update to clear out any cached best edge, as
  // at the next iteration remaining playouts may be different.
  // TODO(crem) Is it really needed?
  root_node_->CancelScoreUpdate(0);
}

// Return the evaluation of the actual best child, regardless of temperature
// settings. This differs from GetBestMove, which does obey any temperature
// settings. So, somethimes, they may return results of different moves.
Search::BestEval Search::GetBestEval() const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  float parent_wl = -root_node_->GetWL();
  float parent_d = root_node_->GetD();
  float parent_m = root_node_->GetM();
  if (!root_node_->HasChildren()) return {parent_wl, parent_d, parent_m};
  EdgeAndNode best_edge = GetBestChildNoTemperature(root_node_, 0);
  return {best_edge.GetWL(parent_wl), best_edge.GetD(parent_d),
          best_edge.GetM(parent_m - 1) + 1};
}

std::pair<Move, Move> Search::GetBestMove() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  EnsureBestMoveKnown();
  return {final_bestmove_.GetMove(played_history_.IsBlackToMove()),
          final_pondermove_.GetMove(!played_history_.IsBlackToMove())};
}

std::int64_t Search::GetTotalPlayouts() const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  return total_playouts_;
}

bool Search::PopulateRootMoveLimit(MoveList* root_moves) const {
  // Search moves overrides tablebase.
  if (!searchmoves_.empty()) {
    *root_moves = searchmoves_;
    return false;
  }
  auto board = played_history_.Last().GetBoard();
  if (!syzygy_tb_ || !board.castlings().no_legal_castle() ||
      (board.ours() | board.theirs()).count() > syzygy_tb_->max_cardinality()) {
    return false;
  }
  return syzygy_tb_->root_probe(
             played_history_.Last(),
             params_.GetSyzygyFastPlay() ||
                 played_history_.DidRepeatSinceLastZeroingMove(),
             root_moves) ||
         syzygy_tb_->root_probe_wdl(played_history_.Last(), root_moves);
}

void Search::ResetBestMove() {
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  bool old_sent = bestmove_is_sent_;
  bestmove_is_sent_ = false;
  EnsureBestMoveKnown();
  bestmove_is_sent_ = old_sent;
}

// Computes the best move, maybe with temperature (according to the settings).
void Search::EnsureBestMoveKnown() REQUIRES(nodes_mutex_)
    REQUIRES(counters_mutex_) {
  if (bestmove_is_sent_) return;
  if (!root_node_->HasChildren()) return;

  float temperature = params_.GetTemperature();
  const int cutoff_move = params_.GetTemperatureCutoffMove();
  const int decay_delay_moves = params_.GetTempDecayDelayMoves();
  const int decay_moves = params_.GetTempDecayMoves();
  const int moves = played_history_.Last().GetGamePly() / 2;

  if (cutoff_move && (moves + 1) >= cutoff_move) {
    temperature = params_.GetTemperatureEndgame();
  } else if (temperature && decay_moves) {
    if (moves >= decay_delay_moves + decay_moves) {
      temperature = 0.0;
    } else if (moves >= decay_delay_moves) {
      temperature *= static_cast<float>
        (decay_delay_moves + decay_moves - moves) / decay_moves;
    }
    // don't allow temperature to decay below endgame temperature
    if (temperature < params_.GetTemperatureEndgame()) {
      temperature = params_.GetTemperatureEndgame();
    }
  }

  final_bestmove_ = temperature ? GetBestRootChildWithTemperature(temperature)
                                : GetBestChildNoTemperature(root_node_, 0);

  if (final_bestmove_.HasNode() && final_bestmove_.node()->HasChildren()) {
    final_pondermove_ = GetBestChildNoTemperature(final_bestmove_.node(), 1);
  }
}

// Returns @count children with most visits.
std::vector<EdgeAndNode> Search::GetBestChildrenNoTemperature(Node* parent,
                                                              int count,
                                                              int depth) const {
  MoveList root_limit;
  if (parent == root_node_) {
    PopulateRootMoveLimit(&root_limit);
  }
  const bool is_odd_depth = (depth % 2) == 1;
  const float draw_score = GetDrawScore(is_odd_depth);
  // Best child is selected using the following criteria:
  // * Prefer shorter terminal wins / avoid shorter terminal losses.
  // * Largest number of playouts.
  // * If two nodes have equal number:
  //   * If that number is 0, the one with larger prior wins.
  //   * If that number is larger than 0, the one with larger eval wins.
  std::vector<EdgeAndNode> edges;
  for (auto edge : parent->Edges()) {
    if (parent == root_node_ && !root_limit.empty() &&
        std::find(root_limit.begin(), root_limit.end(), edge.GetMove()) ==
            root_limit.end()) {
      continue;
    }
    edges.push_back(edge);
  }
  const auto middle = (static_cast<int>(edges.size()) > count)
                          ? edges.begin() + count
                          : edges.end();
  std::partial_sort(
      edges.begin(), middle, edges.end(),
      [draw_score](const auto& a, const auto& b) {
        // The function returns "true" when a is preferred to b.

        // Lists edge types from less desirable to more desirable.
        enum EdgeRank {
          kTerminalLoss,
          kTablebaseLoss,
          kNonTerminal,  // Non terminal or terminal draw.
          kTablebaseWin,
          kTerminalWin,
        };

        auto GetEdgeRank = [](const EdgeAndNode& edge) {
          // This default isn't used as wl only checked for case edge is
          // terminal.
          const auto wl = edge.GetWL(0.0f);
          if (!edge.IsTerminal() || !wl) return kNonTerminal;
          if (edge.IsTbTerminal()) {
            return wl < 0.0 ? kTablebaseLoss : kTablebaseWin;
          }
          return wl < 0.0 ? kTerminalLoss : kTerminalWin;
        };

        // If moves have different outcomes, prefer better outcome.
        const auto a_rank = GetEdgeRank(a);
        const auto b_rank = GetEdgeRank(b);
        if (a_rank != b_rank) return a_rank > b_rank;

        // If both are terminal draws, try to make it shorter.
        if (a_rank == kNonTerminal && a.IsTerminal() && b.IsTerminal()) {
          if (a.IsTbTerminal() != b.IsTbTerminal()) {
            // Prefer non-tablebase draws.
            return a.IsTbTerminal() < b.IsTbTerminal();
          }
          // Prefer shorter draws.
          return a.GetM(0.0f) < b.GetM(0.0f);
        }

        // Neither is terminal, use standard rule.
        if (a_rank == kNonTerminal) {
          // Prefer largest playouts then eval then prior.
          if (a.GetN() != b.GetN()) return a.GetN() > b.GetN();
          // Default doesn't matter here so long as they are the same as either
          // both are N==0 (thus we're comparing equal defaults) or N!=0 and
          // default isn't used.
          if (a.GetQ(0.0f, draw_score, false) !=
              b.GetQ(0.0f, draw_score, false)) {
            return a.GetQ(0.0f, draw_score, false) >
                   b.GetQ(0.0f, draw_score, false);
          }
          return a.GetP() > b.GetP();
        }

        // Both variants are winning, prefer shortest win.
        if (a_rank > kNonTerminal) {
          return a.GetM(0.0f) < b.GetM(0.0f);
        }

        // Both variants are losing, prefer longest losses.
        return a.GetM(0.0f) > b.GetM(0.0f);
      });

  if (count < static_cast<int>(edges.size())) {
    edges.resize(count);
  }
  return edges;
}

// Returns a child with most visits.
EdgeAndNode Search::GetBestChildNoTemperature(Node* parent, int depth) const {
  auto res = GetBestChildrenNoTemperature(parent, 1, depth);
  return res.empty() ? EdgeAndNode() : res.front();
}

// Returns a child of a root chosen according to weighted-by-temperature visit
// count.
EdgeAndNode Search::GetBestRootChildWithTemperature(float temperature) const {
  // Root is at even depth.
  const float draw_score = GetDrawScore(/* is_odd_depth= */ false);
  MoveList root_limit;
  PopulateRootMoveLimit(&root_limit);

  std::vector<float> cumulative_sums;
  float sum = 0.0;
  float max_n = 0.0;
  const float offset = params_.GetTemperatureVisitOffset();
  float max_eval = -1.0f;
  const float fpu =
      GetFpu(params_, root_node_, /* is_root= */ true, draw_score);

  for (auto edge : root_node_->Edges()) {
    if (!root_limit.empty() && std::find(root_limit.begin(), root_limit.end(),
                                         edge.GetMove()) == root_limit.end()) {
      continue;
    }
    if (edge.GetN() + offset > max_n) {
      max_n = edge.GetN() + offset;
      max_eval = edge.GetQ(fpu, draw_score, /* logit_q= */ false);
    }
  }

  // No move had enough visits for temperature, so use default child criteria
  if (max_n <= 0.0f) return GetBestChildNoTemperature(root_node_, 0);

  // TODO(crem) Simplify this code when samplers.h is merged.
  const float min_eval =
      max_eval - params_.GetTemperatureWinpctCutoff() / 50.0f;
  for (auto edge : root_node_->Edges()) {
    if (!root_limit.empty() && std::find(root_limit.begin(), root_limit.end(),
                                         edge.GetMove()) == root_limit.end()) {
      continue;
    }
    if (edge.GetQ(fpu, draw_score, /* logit_q= */ false) < min_eval) continue;
    sum += std::pow(
        std::max(0.0f, (static_cast<float>(edge.GetN()) + offset) / max_n),
        1 / temperature);
    cumulative_sums.push_back(sum);
  }
  assert(sum);

  const float toss = Random::Get().GetFloat(cumulative_sums.back());
  int idx =
      std::lower_bound(cumulative_sums.begin(), cumulative_sums.end(), toss) -
      cumulative_sums.begin();

  for (auto edge : root_node_->Edges()) {
    if (!root_limit.empty() && std::find(root_limit.begin(), root_limit.end(),
                                         edge.GetMove()) == root_limit.end()) {
      continue;
    }
    if (edge.GetQ(fpu, draw_score, /* logit_q= */ false) < min_eval) continue;
    if (idx-- == 0) return edge;
  }
  assert(false);
  return {};
}

void Search::StartThreads(size_t how_many) {
  Mutex::Lock lock(threads_mutex_);
  // First thread is a watchdog thread.
  if (threads_.size() == 0) {
    threads_.emplace_back([this]() { WatchdogThread(); });
  }
  // Start working threads.
  while (threads_.size() <= how_many) {
    threads_.emplace_back([this]() {
      SearchWorker worker(this, params_);
      worker.RunBlocking();
    });
  }
  LOGFILE << "Search started. "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start_time_)
                 .count()
          << "ms already passed.";
}

void Search::RunBlocking(size_t threads) {
  StartThreads(threads);
  Wait();
}

bool Search::IsSearchActive() const {
  return !stop_.load(std::memory_order_acquire);
}

void Search::PopulateCommonIterationStats(IterationStats* stats) {
  stats->time_since_movestart = GetTimeSinceStart();

  SharedMutex::SharedLock nodes_lock(nodes_mutex_);
  stats->time_since_first_batch = GetTimeSinceFirstBatch();
  if (!nps_start_time_ && total_playouts_ > 0) {
    nps_start_time_ = std::chrono::steady_clock::now();
  }
  stats->total_nodes = total_playouts_ + initial_visits_;
  stats->nodes_since_movestart = total_playouts_;
  stats->batches_since_movestart = total_batches_;
  stats->average_depth = cum_depth_ / (total_playouts_ ? total_playouts_ : 1);
  stats->edge_n.clear();
  for (const auto& edge : root_node_->Edges()) {
    stats->edge_n.push_back(edge.GetN());
  }
}

void Search::WatchdogThread() {
  LOGFILE << "Start a watchdog thread.";
  StoppersHints hints;
  IterationStats stats;
  while (true) {
    hints.Reset();
    PopulateCommonIterationStats(&stats);
    MaybeTriggerStop(stats, &hints);
    MaybeOutputInfo();

    using namespace std::chrono_literals;
    constexpr auto kMaxWaitTimeMs = 100;
    constexpr auto kMinWaitTimeMs = 1;

    Mutex::Lock lock(counters_mutex_);
    // Only exit when bestmove is responded. It may happen that search threads
    // already all exited, and we need at least one thread that can do that.
    if (bestmove_is_sent_) break;

    auto remaining_time = hints.GetEstimatedRemainingTimeMs();
    if (remaining_time > kMaxWaitTimeMs) remaining_time = kMaxWaitTimeMs;
    if (remaining_time < kMinWaitTimeMs) remaining_time = kMinWaitTimeMs;
    // There is no real need to have max wait time, and sometimes it's fine
    // to wait without timeout at all (e.g. in `go nodes` mode), but we
    // still limit wait time for exotic cases like when pc goes to sleep
    // mode during thinking.
    // Minimum wait time is there to prevent busy wait and other threads
    // starvation.
    watchdog_cv_.wait_for(
        lock.get_raw(), std::chrono::milliseconds(remaining_time),
        [this]() { return stop_.load(std::memory_order_acquire); });
  }
  LOGFILE << "End a watchdog thread.";
}

void Search::FireStopInternal() {
  stop_.store(true, std::memory_order_release);
  watchdog_cv_.notify_all();
}

void Search::Stop() {
  Mutex::Lock lock(counters_mutex_);
  ok_to_respond_bestmove_ = true;
  FireStopInternal();
  LOGFILE << "Stopping search due to `stop` uci command.";
}

void Search::Abort() {
  Mutex::Lock lock(counters_mutex_);
  if (!stop_.load(std::memory_order_acquire) ||
      (!bestmove_is_sent_ && !ok_to_respond_bestmove_)) {
    bestmove_is_sent_ = true;
    FireStopInternal();
  }
  LOGFILE << "Aborting search, if it is still active.";
}

void Search::Wait() {
  Mutex::Lock lock(threads_mutex_);
  while (!threads_.empty()) {
    threads_.back().join();
    threads_.pop_back();
  }
}

void Search::CancelSharedCollisions() REQUIRES(nodes_mutex_) {
  for (auto& entry : shared_collisions_) {
    Node* node = entry.first;
    for (node = node->GetParent(); node != root_node_->GetParent();
         node = node->GetParent()) {
      node->CancelScoreUpdate(entry.second);
    }
  }
  shared_collisions_.clear();
}

Search::~Search() {
  Abort();
  Wait();
  {
    SharedMutex::Lock lock(nodes_mutex_);
    CancelSharedCollisions();
  }
  LOGFILE << "Search destroyed.";
}

//////////////////////////////////////////////////////////////////////////////
// SearchWorker
//////////////////////////////////////////////////////////////////////////////

void SearchWorker::ExecuteOneIteration() {
  // 1. Initialize internal structures.
  InitializeIteration(search_->network_->NewComputation());

  if (params_.GetMaxConcurrentSearchers() != 0) {
    while (true) {
      // If search is stop, we've not gathered or done anything and we don't
      // want to, so we can safely skip all below. But make sure we have done
      // at least one iteration.
      if (search_->stop_.load(std::memory_order_acquire) &&
          search_->GetTotalPlayouts() + search_->initial_visits_ > 0) {
        return;
      }
      int available =
          search_->pending_searchers_.load(std::memory_order_acquire);
      if (available > 0 &&
          search_->pending_searchers_.compare_exchange_weak(
              available, available - 1, std::memory_order_acq_rel)) {
        break;
      }
      // This is a hard spin lock to reduce latency but at the expense of busy
      // wait cpu usage. If search worker count is large, this is probably a bad
      // idea.
    }
  }

  // 2. Gather minibatch.
  GatherMinibatch();

  // 2b. Collect collisions.
  CollectCollisions();

  // 3. Prefetch into cache.
  MaybePrefetchIntoCache();

  if (params_.GetMaxConcurrentSearchers() != 0) {
    search_->pending_searchers_.fetch_add(1, std::memory_order_acq_rel);
  }

  // 4. Run NN computation.
  RunNNComputation();

  // 5. Retrieve NN computations (and terminal values) into nodes.
  FetchMinibatchResults();

  // 6. Propagate the new nodes' information to all their parents in the tree.
  DoBackupUpdate();

  // 7. Update the Search's status and progress information.
  UpdateCounters();
}

// 1. Initialize internal structures.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::InitializeIteration(
    std::unique_ptr<NetworkComputation> computation) {
  computation_ = std::make_unique<CachingComputation>(std::move(computation),
                                                      search_->cache_);
  minibatch_.clear();

  if (!root_move_filter_populated_) {
    root_move_filter_populated_ = true;
    if (search_->PopulateRootMoveLimit(&root_move_filter_)) {
      search_->tb_hits_.fetch_add(1, std::memory_order_acq_rel);
    }
  }
}

// 2. Gather minibatch.
// ~~~~~~~~~~~~~~~~~~~~
void SearchWorker::GatherMinibatch() {
  // Total number of nodes to process.
  int minibatch_size = 0;
  int collision_events_left = params_.GetMaxCollisionEvents();
  int collisions_left = params_.GetMaxCollisionVisitsId();

  // Number of nodes processed out of order.
  number_out_of_order_ = 0;

  // Gather nodes to process in the current batch.
  // If we had too many nodes out of order, also interrupt the iteration so
  // that search can exit.
  while (minibatch_size < params_.GetMiniBatchSize() &&
         number_out_of_order_ < params_.GetMaxOutOfOrderEvals()) {
    // If there's something to process without touching slow neural net, do it.
    if (minibatch_size > 0 && computation_->GetCacheMisses() == 0) return;
    // Pick next node to extend.
    minibatch_.emplace_back(PickNodeToExtend(collisions_left));
    auto& picked_node = minibatch_.back();
    auto* node = picked_node.node;

    // There was a collision. If limit has been reached, return, otherwise
    // just start search of another node.
    if (picked_node.IsCollision()) {
      if (--collision_events_left <= 0) return;
      if ((collisions_left -= picked_node.multivisit) <= 0) return;
      if (search_->stop_.load(std::memory_order_acquire)) return;
      continue;
    }
    ++minibatch_size;

    // If node is already known as terminal (win/loss/draw according to rules
    // of the game), it means that we already visited this node before.
    if (picked_node.IsExtendable()) {
      // Node was never visited, extend it.
      ExtendNode(node);

      // Only send non-terminal nodes to a neural network.
      if (!node->IsTerminal()) {
        picked_node.nn_queried = true;
        int transform;
        picked_node.is_cache_hit = AddNodeToComputation(node, true, &transform);
        picked_node.probability_transform = transform;
      }
    }

    // If out of order eval is enabled and the node to compute we added last
    // doesn't require NN eval (i.e. it's a cache hit or terminal node), do
    // out of order eval for it.
    if (params_.GetOutOfOrderEval() && picked_node.CanEvalOutOfOrder()) {
      // Perform out of order eval for the last entry in minibatch_.
      FetchSingleNodeResult(&picked_node, computation_->GetBatchSize() - 1);
      {
        // Nodes mutex for doing node updates.
        SharedMutex::Lock lock(search_->nodes_mutex_);
        DoBackupUpdateSingleNode(picked_node);
      }

      // Remove last entry in minibatch_, as it has just been
      // processed.
      // If NN eval was already processed out of order, remove it.
      if (picked_node.nn_queried) computation_->PopCacheHit();
      minibatch_.pop_back();
      --minibatch_size;
      ++number_out_of_order_;
    }
    // Check for stop at the end so we have at least one node.
    if (search_->stop_.load(std::memory_order_acquire)) return;
  }
}

namespace {
void IncrementNInFlight(Node* node, Node* root, int amount) {
  if (amount == 0) return;
  while (true) {
    node->IncrementNInFlight(amount);
    if (node == root) break;
    node = node->GetParent();
  }
}
}  // namespace

// Returns node and whether there's been a search collision on the node.
SearchWorker::NodeToProcess SearchWorker::PickNodeToExtend(
    int collision_limit) {
  // Starting from search_->root_node_, generate a playout, choosing a
  // node at each level according to the MCTS formula. n_in_flight_ is
  // incremented for each node in the playout (via TryStartScoreUpdate()).

  Node* node = search_->root_node_;
  Node::Iterator best_edge;
  Node::Iterator second_best_edge;

  // Precache a newly constructed node to avoid memory allocations being
  // performed while the mutex is held.
  if (!precached_node_) {
    precached_node_ = std::make_unique<Node>(nullptr, 0);
  }

  SharedMutex::Lock lock(search_->nodes_mutex_);

  // Fetch the current best root node visits for possible smart pruning.
  const int64_t best_node_n = search_->current_best_edge_.GetN();

  // True on first iteration, false as we dive deeper.
  bool is_root_node = true;
  const float even_draw_score = search_->GetDrawScore(false);
  const float odd_draw_score = search_->GetDrawScore(true);
  uint16_t depth = 0;
  bool node_already_updated = true;

  while (true) {
    // First, terminate if we find collisions or leaf nodes.
    // Set 'node' to point to the node that was picked on previous iteration,
    // possibly spawning it.
    // TODO(crem) This statement has to be in the end of the loop rather than
    //            in the beginning (and there would be no need for "if
    //            (!is_root_node)"), but that would mean extra mutex lock.
    //            Will revisit that after rethinking locking strategy.
    if (!node_already_updated) {
      node = best_edge.GetOrSpawnNode(/* parent */ node, &precached_node_);
    }
    best_edge.Reset();
    depth++;

    // n_in_flight_ is incremented. If the method returns false, then there is
    // a search collision, and this node is already being expanded.
    if (!node->TryStartScoreUpdate()) {
      if (!is_root_node) {
        IncrementNInFlight(node->GetParent(), search_->root_node_,
                           collision_limit - 1);
      }
      return NodeToProcess::Collision(node, depth, collision_limit);
    }
    // Either terminal or unexamined leaf node -- the end of this playout.
    if (node->IsTerminal() || !node->HasChildren()) {
      return NodeToProcess::Visit(node, depth);
    }
    Node* possible_shortcut_child = node->GetCachedBestChild();
    if (possible_shortcut_child) {
      // Add two here to reverse the conservatism that goes into calculating the
      // remaining cache visits.
      collision_limit =
          std::min(collision_limit, node->GetRemainingCacheVisits() + 2);
      is_root_node = false;
      node = possible_shortcut_child;
      node_already_updated = true;
      continue;
    }
    node_already_updated = false;

    // If we fall through, then n_in_flight_ has been incremented but this
    // playout remains incomplete; we must go deeper.
    const float cpuct = ComputeCpuct(params_, node->GetN(), is_root_node);
    const float puct_mult =
        cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
    const float childVisits = node->GetChildrenVisits();
    const float shift = params_.GetPolicyDecayShift();
    const float slope = params_.GetPolicyDecaySlope();
    float best = std::numeric_limits<float>::lowest();
    float best_without_u = std::numeric_limits<float>::lowest();
    float second_best = std::numeric_limits<float>::lowest();
    // Root depth is 1 here, while for GetDrawScore() it's 0-based, that's why
    // the weirdness.
    const float draw_score =
        (depth % 2 == 0) ? odd_draw_score : even_draw_score;
    const float fpu = GetFpu(params_, node, is_root_node, draw_score);

    const float node_q = node->GetQ(0.0f);
    const bool do_moves_left_adjustment =
        moves_left_support_ &&
        (std::abs(node_q) > params_.GetMovesLeftThreshold());

    for (auto child : node->Edges()) {
      if (is_root_node) {
        // If there's no chance to catch up to the current best node with
        // remaining playouts, don't consider it.
        // best_move_node_ could have changed since best_node_n was retrieved.
        // To ensure we have at least one node to expand, always include
        // current best node.
        if (child != search_->current_best_edge_ &&
            latest_time_manager_hints_.GetEstimatedRemainingPlayouts() <
                best_node_n - child.GetN()) {
          continue;
        }
        // If root move filter exists, make sure move is in the list.
        if (!root_move_filter_.empty() &&
            std::find(root_move_filter_.begin(), root_move_filter_.end(),
                      child.GetMove()) == root_move_filter_.end()) {
          continue;
        }
      }

      const float Q = child.GetQ(fpu, draw_score, params_.GetLogitQ());
      float M = 0.0f;
      if (do_moves_left_adjustment) {
        const float m_slope = params_.GetMovesLeftSlope();
        const float m_cap = params_.GetMovesLeftMaxEffect();
        const float parent_m = node->GetM();
        const float child_m = child.GetM(parent_m);
        M = std::clamp(m_slope * (child_m - parent_m), -m_cap, m_cap) *
            std::copysign(1.0f, -Q);
        const float a = params_.GetMovesLeftConstantFactor();
        const float b = params_.GetMovesLeftScaledFactor();
        const float c = params_.GetMovesLeftQuadraticFactor();
        M *= a + b * std::abs(Q) + c * Q * Q;
      }

      const float score = child.GetU(puct_mult, childVisits, shift, slope) + Q + M;
      if (score > best) {
        second_best = best;
        second_best_edge = best_edge;
        best = score;
        best_without_u = Q + M;
        best_edge = child;
      } else if (score > second_best) {
        second_best = score;
        second_best_edge = child;
      }
    }

    if (second_best_edge) {
      int estimated_visits_to_change_best =
          best_edge.GetVisitsToReachU(second_best, puct_mult, best_without_u);
      // Only cache for n-2 steps as the estimate created by GetVisitsToReachU
      // has potential rounding errors and some conservative logic that can push
      // it up to 2 away from the real value.
      node->UpdateBestChild(best_edge,
                            std::max(0, estimated_visits_to_change_best - 2));
      collision_limit =
          std::min(collision_limit, estimated_visits_to_change_best);
      assert(collision_limit >= 1);
      second_best_edge.Reset();
    }

    is_root_node = false;
  }
}

void SearchWorker::ExtendNode(Node* node) {
  // Initialize position sequence with pre-move position.
  history_.Trim(search_->played_history_.GetLength());
  std::vector<Move> to_add;
  // Could instead reserve one more than the difference between history_.size()
  // and history_.capacity().
  to_add.reserve(60);
  Node* cur = node;
  while (cur != search_->root_node_) {
    Node* prev = cur->GetParent();
    to_add.push_back(prev->GetEdgeToNode(cur)->GetMove());
    cur = prev;
  }
  for (int i = to_add.size() - 1; i >= 0; i--) {
    history_.Append(to_add[i]);
  }

  // We don't need the mutex because other threads will see that N=0 and
  // N-in-flight=1 and will not touch this node.
  const auto& board = history_.Last().GetBoard();
  auto legal_moves = board.GenerateLegalMoves();

  // Check whether it's a draw/lose by position. Importantly, we must check
  // these before doing the by-rule checks below.
  if (legal_moves.empty()) {
    // Could be a checkmate or a stalemate
    if (board.IsUnderCheck()) {
      node->MakeTerminal(GameResult::WHITE_WON);
    } else {
      node->MakeTerminal(GameResult::DRAW);
    }
    return;
  }

  // We can shortcircuit these draws-by-rule only if they aren't root;
  // if they are root, then thinking about them is the point.
  if (node != search_->root_node_) {
    if (!board.HasMatingMaterial()) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    }

    if (history_.Last().GetRule50Ply() >= 100) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    }

    if (history_.Last().GetRepetitions() >= 2) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    }

    // Neither by-position or by-rule termination, but maybe it's a TB position.
    if (search_->syzygy_tb_ && board.castlings().no_legal_castle() &&
        history_.Last().GetRule50Ply() == 0 &&
        (board.ours() | board.theirs()).count() <=
            search_->syzygy_tb_->max_cardinality()) {
      ProbeState state;
      const WDLScore wdl =
          search_->syzygy_tb_->probe_wdl(history_.Last(), &state);
      // Only fail state means the WDL is wrong, probe_wdl may produce correct
      // result with a stat other than OK.
      if (state != FAIL) {
        // TB nodes don't have NN evaluation, assign M from parent node.
        float m = 0.0f;
        auto parent = node->GetParent();
        if (parent) {
          m = std::max(0.0f, parent->GetM() - 1.0f);
        }
        // If the colors seem backwards, check the checkmate check above.
        if (wdl == WDL_WIN) {
          node->MakeTerminal(GameResult::BLACK_WON, m,
                             Node::Terminal::Tablebase);
        } else if (wdl == WDL_LOSS) {
          node->MakeTerminal(GameResult::WHITE_WON, m,
                             Node::Terminal::Tablebase);
        } else {  // Cursed wins and blessed losses count as draws.
          node->MakeTerminal(GameResult::DRAW, m, Node::Terminal::Tablebase);
        }
        search_->tb_hits_.fetch_add(1, std::memory_order_acq_rel);
        return;
      }
    }
  }

  // Add legal moves as edges of this node.
  node->CreateEdges(legal_moves);
}

// Returns whether node was already in cache.
bool SearchWorker::AddNodeToComputation(Node* node, bool add_if_cached,
                                        int* transform_out) {
  const auto hash = history_.HashLast(params_.GetCacheHistoryLength() + 1);
  // If already in cache, no need to do anything.
  if (add_if_cached) {
    if (computation_->AddInputByHash(hash)) {
      if (transform_out) {
        *transform_out = TransformForPosition(
            search_->network_->GetCapabilities().input_format, history_);
      }
      return true;
    }
  } else {
    if (search_->cache_->ContainsKey(hash)) {
      if (transform_out) {
        *transform_out = TransformForPosition(
            search_->network_->GetCapabilities().input_format, history_);
      }
      return true;
    }
  }
  int transform;
  auto planes =
      EncodePositionForNN(search_->network_->GetCapabilities().input_format,
                          history_, 8, params_.GetHistoryFill(), &transform);

  std::vector<uint16_t> moves;

  if (node && node->HasChildren()) {
    // Legal moves are known, use them.
    moves.reserve(node->GetNumEdges());
    for (const auto& edge : node->Edges()) {
      moves.emplace_back(edge.GetMove().as_nn_index(transform));
    }
  } else {
    // Cache pseudolegal moves. A bit of a waste, but faster.
    const auto& pseudolegal_moves =
        history_.Last().GetBoard().GeneratePseudolegalMoves();
    moves.reserve(pseudolegal_moves.size());
    for (auto iter = pseudolegal_moves.begin(), end = pseudolegal_moves.end();
         iter != end; ++iter) {
      moves.emplace_back(iter->as_nn_index(transform));
    }
  }

  computation_->AddInput(hash, std::move(planes), std::move(moves));
  if (transform_out) *transform_out = transform;
  return false;
}

// 2b. Copy collisions into shared collisions.
void SearchWorker::CollectCollisions() {
  SharedMutex::Lock lock(search_->nodes_mutex_);

  for (const NodeToProcess& node_to_process : minibatch_) {
    if (node_to_process.IsCollision()) {
      search_->shared_collisions_.emplace_back(node_to_process.node,
                                               node_to_process.multivisit);
    }
  }
}

// 3. Prefetch into cache.
// ~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::MaybePrefetchIntoCache() {
  // TODO(mooskagh) Remove prefetch into cache if node collisions work well.
  // If there are requests to NN, but the batch is not full, try to prefetch
  // nodes which are likely useful in future.
  if (search_->stop_.load(std::memory_order_acquire)) return;
  if (computation_->GetCacheMisses() > 0 &&
      computation_->GetCacheMisses() < params_.GetMaxPrefetchBatch()) {
    history_.Trim(search_->played_history_.GetLength());
    SharedMutex::SharedLock lock(search_->nodes_mutex_);
    PrefetchIntoCache(
        search_->root_node_,
        params_.GetMaxPrefetchBatch() - computation_->GetCacheMisses(), false);
  }
}

// Prefetches up to @budget nodes into cache. Returns number of nodes
// prefetched.
int SearchWorker::PrefetchIntoCache(Node* node, int budget, bool is_odd_depth) {
  const float draw_score = search_->GetDrawScore(is_odd_depth);
  if (budget <= 0) return 0;

  // We are in a leaf, which is not yet being processed.
  if (!node || node->GetNStarted() == 0) {
    if (AddNodeToComputation(node, false, nullptr)) {
      // Make it return 0 to make it not use the slot, so that the function
      // tries hard to find something to cache even among unpopular moves.
      // In practice that slows things down a lot though, as it's not always
      // easy to find what to cache.
      return 1;
    }
    return 1;
  }

  assert(node);
  // n = 0 and n_in_flight_ > 0, that means the node is being extended.
  if (node->GetN() == 0) return 0;
  // The node is terminal; don't prefetch it.
  if (node->IsTerminal()) return 0;

  // Populate all subnodes and their scores.
  typedef std::pair<float, EdgeAndNode> ScoredEdge;
  std::vector<ScoredEdge> scores;
  const float cpuct =
      ComputeCpuct(params_, node->GetN(), node == search_->root_node_);
  const float childVisits = node->GetChildrenVisits();
  const float shift = params_.GetPolicyDecayShift();
  const float slope = params_.GetPolicyDecaySlope();
  const float puct_mult =
      cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
  const float fpu =
      GetFpu(params_, node, node == search_->root_node_, draw_score);
  for (auto edge : node->Edges()) {
    if (edge.GetP() == 0.0f) continue;
    // Flip the sign of a score to be able to easily sort.
    // TODO: should this use logit_q if set??
    scores.emplace_back(-edge.GetU(puct_mult, childVisits, shift, slope) -
                            edge.GetQ(fpu, draw_score, /* logit_q= */ false),
                        edge);
  }

  size_t first_unsorted_index = 0;
  int total_budget_spent = 0;
  int budget_to_spend = budget;  // Initialize for the case where there's only
                                 // one child.
  for (size_t i = 0; i < scores.size(); ++i) {
    if (search_->stop_.load(std::memory_order_acquire)) break;
    if (budget <= 0) break;

    // Sort next chunk of a vector. 3 at a time. Most of the time it's fine.
    if (first_unsorted_index != scores.size() &&
        i + 2 >= first_unsorted_index) {
      const int new_unsorted_index =
          std::min(scores.size(), budget < 2 ? first_unsorted_index + 2
                                             : first_unsorted_index + 3);
      std::partial_sort(scores.begin() + first_unsorted_index,
                        scores.begin() + new_unsorted_index, scores.end(),
                        [](const ScoredEdge& a, const ScoredEdge& b) {
                          return a.first < b.first;
                        });
      first_unsorted_index = new_unsorted_index;
    }

    auto edge = scores[i].second;
    // Last node gets the same budget as prev-to-last node.
    if (i != scores.size() - 1) {
      // Sign of the score was flipped for sorting, so flip it back.
      const float next_score = -scores[i + 1].first;
      // TODO: As above - should this use logit_q if set?
      const float q = edge.GetQ(-fpu, draw_score, /* logit_q= */ false);
      if (next_score > q) {
        budget_to_spend =
            std::min(budget, int(edge.GetP() * puct_mult / (next_score - q) -
                                 edge.GetNStarted()) +
                                 1);
      } else {
        budget_to_spend = budget;
      }
    }
    history_.Append(edge.GetMove());
    const int budget_spent =
        PrefetchIntoCache(edge.node(), budget_to_spend, !is_odd_depth);
    history_.Pop();
    budget -= budget_spent;
    total_budget_spent += budget_spent;
  }
  return total_budget_spent;
}

// 4. Run NN computation.
// ~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::RunNNComputation() { computation_->ComputeBlocking(); }

// 5. Retrieve NN computations (and terminal values) into nodes.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::FetchMinibatchResults() {
  // Populate NN/cached results, or terminal results, into nodes.
  int idx_in_computation = 0;
  for (auto& node_to_process : minibatch_) {
    FetchSingleNodeResult(&node_to_process, idx_in_computation);
    if (node_to_process.nn_queried) ++idx_in_computation;
  }
}

void SearchWorker::FetchSingleNodeResult(NodeToProcess* node_to_process,
                                         int idx_in_computation) {
  Node* node = node_to_process->node;
  if (!node_to_process->nn_queried) {
    // Terminal nodes don't involve the neural NetworkComputation, nor do
    // they require any further processing after value retrieval.
    node_to_process->v = node->GetWL();
    node_to_process->d = node->GetD();
    node_to_process->m = node->GetM();
    return;
  }
  // For NN results, we need to populate policy as well as value.
  // First the value...
  node_to_process->v = -computation_->GetQVal(idx_in_computation);
  node_to_process->d = computation_->GetDVal(idx_in_computation);
  node_to_process->m = computation_->GetMVal(idx_in_computation);
  // ...and secondly, the policy data.
  // Calculate maximum first.
  float max_p = -std::numeric_limits<float>::infinity();
  for (auto edge : node->Edges()) {
    max_p = std::max(max_p, computation_->GetPVal(
                                idx_in_computation,
                                edge.GetMove().as_nn_index(
                                    node_to_process->probability_transform)));
  }
  float total = 0.0;
  for (auto edge : node->Edges()) {
    float p = computation_->GetPVal(
        idx_in_computation,
        edge.GetMove().as_nn_index(node_to_process->probability_transform));
    // Perform softmax and take into account policy softmax temperature T.
    // Note that we want to calculate (exp(p-max_p))^(1/T) = exp((p-max_p)/T).
    p = FastExp((p - max_p) / params_.GetPolicySoftmaxTemp());

    // Note that p now lies in [0, 1], so it is safe to store it in compressed
    // format. Normalization happens later.
    edge.edge()->SetP(p);
    // Edge::SetP does some rounding, so only add to the total after rounding.
    total += edge.edge()->GetP();
  }
  // Normalize P values to add up to 1.0.
  if (total > 0.0f) {
    const float scale = 1.0f / total;
    for (auto edge : node->Edges()) edge.edge()->SetP(edge.GetP() * scale);
  }
  // Add Dirichlet noise if enabled and at root.
  if (params_.GetNoiseEpsilon() && node == search_->root_node_) {
    ApplyDirichletNoise(node, params_.GetNoiseEpsilon(),
                        params_.GetNoiseAlpha());
  }
}

// 6. Propagate the new nodes' information to all their parents in the tree.
// ~~~~~~~~~~~~~~
void SearchWorker::DoBackupUpdate() {
  // Nodes mutex for doing node updates.
  SharedMutex::Lock lock(search_->nodes_mutex_);

  bool work_done = number_out_of_order_ > 0;
  for (const NodeToProcess& node_to_process : minibatch_) {
    DoBackupUpdateSingleNode(node_to_process);
    if (!node_to_process.IsCollision()) {
      work_done = true;
    }
  }
  if (!work_done) return;
  search_->CancelSharedCollisions();
  search_->total_batches_ += 1;
}

void SearchWorker::DoBackupUpdateSingleNode(
    const NodeToProcess& node_to_process) REQUIRES(search_->nodes_mutex_) {
  Node* node = node_to_process.node;
  if (node_to_process.IsCollision()) {
    // Collisions are handled via shared_collisions instead.
    return;
  }

  // For the first visit to a terminal, maybe update parent bounds too.
  auto update_parent_bounds =
      params_.GetStickyEndgames() && node->IsTerminal() && !node->GetN();

  // Backup V value up to a root. After 1 visit, V = Q.
  float v = node_to_process.v;
  float d = node_to_process.d;
  float m = node_to_process.m;
  int depth = 0;
  for (Node *n = node, *p; n != search_->root_node_->GetParent(); n = p) {
    p = n->GetParent();

    // Current node might have become terminal from some other descendant, so
    // backup the rest of the way with more accurate values.
    if (n->IsTerminal()) {
      v = n->GetWL();
      d = n->GetD();
      m = n->GetM();
    }
    n->FinalizeScoreUpdate(v / (1.0f + params_.GetShortSightedness() * depth),
                           d, m, node_to_process.multivisit);

    // Nothing left to do without ancestors to update.
    if (!p) break;

    // Try setting parent bounds except the root or those already terminal.
    update_parent_bounds = update_parent_bounds && p != search_->root_node_ &&
                           !p->IsTerminal() && MaybeSetBounds(p, m);

    // Q will be flipped for opponent.
    v = -v;
    depth++;
    m++;

    // Update the stats.
    // Best move.
    if (p == search_->root_node_ &&
        search_->current_best_edge_.GetN() <= n->GetN()) {
      search_->current_best_edge_ =
          search_->GetBestChildNoTemperature(search_->root_node_, 0);
    }
  }
  search_->total_playouts_ += node_to_process.multivisit;
  search_->cum_depth_ += node_to_process.depth * node_to_process.multivisit;
  search_->max_depth_ = std::max(search_->max_depth_, node_to_process.depth);
}

bool SearchWorker::MaybeSetBounds(Node* p, float m) const {
  auto losing_m = 0.0f;
  auto prefer_tb = false;

  // Determine the maximum (lower, upper) bounds across all children.
  // (-1,-1) Loss (initial and lowest bounds)
  // (-1, 0) Can't Win
  // (-1, 1) Regular node
  // ( 0, 0) Draw
  // ( 0, 1) Can't Lose
  // ( 1, 1) Win (highest bounds)
  auto lower = GameResult::BLACK_WON;
  auto upper = GameResult::BLACK_WON;
  for (const auto& edge : p->Edges()) {
    const auto [edge_lower, edge_upper] = edge.GetBounds();
    lower = std::max(edge_lower, lower);
    upper = std::max(edge_upper, upper);

    // Checkmate is the best, so short-circuit.
    const auto is_tb = edge.IsTbTerminal();
    if (edge_lower == GameResult::WHITE_WON && !is_tb) {
      prefer_tb = false;
      break;
    } else if (edge_upper == GameResult::BLACK_WON) {
      // Track the longest loss.
      losing_m = std::max(losing_m, edge.GetM(0.0f));
    }
    prefer_tb = prefer_tb || is_tb;
  }

  // The parent's bounds are flipped from the children (-max(U), -max(L))
  // aggregated as if it was a single child (forced move) of the same bound.
  //       Loss (-1,-1) -> ( 1, 1) Win
  //  Can't Win (-1, 0) -> ( 0, 1) Can't Lose
  //    Regular (-1, 1) -> (-1, 1) Regular
  //       Draw ( 0, 0) -> ( 0, 0) Draw
  // Can't Lose ( 0, 1) -> (-1, 0) Can't Win
  //        Win ( 1, 1) -> (-1,-1) Loss

  // Nothing left to do for ancestors if the parent would be a regular node.
  if (lower == GameResult::BLACK_WON && upper == GameResult::WHITE_WON) {
    return false;
  } else if (lower == upper) {
    // Search can stop at the parent if the bounds can't change anymore, so make
    // it terminal preferring shorter wins and longer losses.
    p->MakeTerminal(
        -upper,
        (upper == GameResult::BLACK_WON ? std::max(losing_m, m) : m) + 1.0f,
        prefer_tb ? Node::Terminal::Tablebase : Node::Terminal::EndOfGame);
  } else {
    p->SetBounds(-upper, -lower);
  }

  // Bounds were set, so indicate we should check the parent too.
  return true;
}

// 7. Update the Search's status and progress information.
//~~~~~~~~~~~~~~~~~~~~
void SearchWorker::UpdateCounters() {
  search_->PopulateCommonIterationStats(&iteration_stats_);
  search_->MaybeTriggerStop(iteration_stats_, &latest_time_manager_hints_);
  search_->MaybeOutputInfo();

  // If this thread had no work, not even out of order, then sleep for some
  // milliseconds. Collisions don't count as work, so have to enumerate to find
  // out if there was anything done.
  bool work_done = number_out_of_order_ > 0;
  if (!work_done) {
    for (NodeToProcess& node_to_process : minibatch_) {
      if (!node_to_process.IsCollision()) {
        work_done = true;
        break;
      }
    }
  }
  if (!work_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

}  // namespace lczero
