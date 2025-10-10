/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2023 The LCZero Authors

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

#include "search/dag_classic/search.h"

#include <absl/cleanup/cleanup.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <thread>

#include "search/dag_classic/node.h"
#include "utils/fastmath.h"
#include "utils/random.h"
#include "utils/spinhelper.h"

namespace lczero {
namespace dag_classic {

namespace {
// Maximum delay between outputting "uci info" when nothing interesting happens.
const int kUciInfoMinimumFrequencyMs = 5000;

MoveList MakeRootMoveFilter(const MoveList& searchmoves,
                            SyzygyTablebase* syzygy_tb,
                            const PositionHistory& history, bool fast_play,
                            std::atomic<int>* tb_hits, bool* dtz_success) {
  assert(tb_hits);
  assert(dtz_success);
  // Search moves overrides tablebase.
  if (!searchmoves.empty()) return searchmoves;
  const auto& board = history.Last().GetBoard();
  MoveList root_moves;
  if (!syzygy_tb || !board.castlings().no_legal_castle() ||
      (board.ours() | board.theirs()).count() > syzygy_tb->max_cardinality()) {
    return root_moves;
  }
  if (syzygy_tb->root_probe(
          history.Last(), fast_play || history.DidRepeatSinceLastZeroingMove(),
          false, &root_moves)) {
    *dtz_success = true;
    tb_hits->fetch_add(1, std::memory_order_acq_rel);
  } else if (syzygy_tb->root_probe_wdl(history.Last(), &root_moves)) {
    tb_hits->fetch_add(1, std::memory_order_acq_rel);
  }
  return root_moves;
}

class MEvaluator {
 public:
  MEvaluator()
      : enabled_{false},
        m_slope_{0.0f},
        m_cap_{0.0f},
        a_constant_{0.0f},
        a_linear_{0.0f},
        a_square_{0.0f},
        q_threshold_{0.0f},
        parent_m_{0.0f} {}

  MEvaluator(const SearchParams& params, const Node* parent = nullptr)
      : enabled_{true},
        m_slope_{params.GetMovesLeftSlope()},
        m_cap_{params.GetMovesLeftMaxEffect()},
        a_constant_{params.GetMovesLeftConstantFactor()},
        a_linear_{params.GetMovesLeftScaledFactor()},
        a_square_{params.GetMovesLeftQuadraticFactor()},
        q_threshold_{params.GetMovesLeftThreshold()},
        parent_m_{parent ? parent->GetM() : 0.0f},
        parent_within_threshold_{parent ? WithinThreshold(parent, q_threshold_)
                                        : false} {}

  void SetParent(const Node* parent) {
    assert(parent);
    if (enabled_) {
      parent_m_ = parent->GetM();
      parent_within_threshold_ = WithinThreshold(parent, q_threshold_);
    }
  }

  // Calculates the utility for favoring shorter wins and longer losses.
  float GetMUtility(Node* child, float q) const {
    if (!enabled_ || !parent_within_threshold_) return 0.0f;
    const float child_m = child->GetM();
    float m = std::clamp(m_slope_ * (child_m - parent_m_), -m_cap_, m_cap_);
    m *= FastSign(-q);
    if (q_threshold_ > 0.0f && q_threshold_ < 1.0f) {
      // This allows a smooth M effect with higher q thresholds, which is
      // necessary for using MLH together with contempt.
      q = std::max(0.0f, (std::abs(q) - q_threshold_)) / (1.0f - q_threshold_);
    }
    m *= a_constant_ + a_linear_ * std::abs(q) + a_square_ * q * q;
    return m;
  }

  float GetMUtility(const EdgeAndNode& child, float q) const {
    if (!enabled_ || !parent_within_threshold_) return 0.0f;
    if (child.GetN() == 0) return GetDefaultMUtility();
    return GetMUtility(child.node(), q);
  }

  // The M utility to use for unvisited nodes.
  float GetDefaultMUtility() const { return 0.0f; }

 private:
  static bool WithinThreshold(const Node* parent, float q_threshold) {
    return std::abs(parent->GetQ(0.0f)) > q_threshold;
  }

  const bool enabled_;
  const float m_slope_;
  const float m_cap_;
  const float a_constant_;
  const float a_linear_;
  const float a_square_;
  const float q_threshold_;
  float parent_m_ = 0.0f;
  bool parent_within_threshold_ = false;
};

// Unpack task_count_ atomic which holds both task_count_ and tasks_taken_. It
// can unpack a value from an already read value or load it from the atomic
// variable.
// Variables are packed together because there is a potential race between task
// workers and ResetTasks. A task worker can read tasks_taken_ and task_count
// to a local register. A task worker can be suspended by kernel before tries
// to acquire work. Other threads can process all tasks and main thread resets
// tasks before the suspended thread resumes. The suspended thread now manages
// to acquire work based on stale values if the stale tasks_taken was zero.
// Packed values avoid the race because compare exchange is checking both when
// incrementing tasks_taken_.
template<typename T>
std::tuple<int, int, int> ReadTaskCount(T& task_count) {
  int packed;
  if constexpr(std::is_same_v<T, std::atomic<int>>) {
    packed = task_count.load(std::memory_order_acquire);
  } else {
    packed = task_count;
  }
  // The top half is tasks taken.
  const int shift = SearchWorker::kTasksTakenShift;
  int tasks_taken = packed >> shift;
  // The bottom is task count. The first shift moves the sign bit from the lower
  // half to the hardware sign bit. The second shift lowers bits back to the
  // original positions and duplicates the sign bit if it is set.
  int tc = (packed << shift) >> shift;
  return {packed, tasks_taken, tc};
}

[[maybe_unused]]
bool IsTasksCompleted(const std::atomic<int>& task_count,
                      const std::atomic<int>& completed_tasks) {
  int tc = 0, nta = 0;
  std::tie(std::ignore, nta, tc) = ReadTaskCount(task_count);
  int ct = completed_tasks.load(std::memory_order_acquire);
  return tc == ct || (nta == ct && tc == -1);
}

}  // namespace

Search::Search(const NodeTree& tree, Backend* backend,
               std::unique_ptr<UciResponder> uci_responder,
               const MoveList& searchmoves,
               std::chrono::steady_clock::time_point start_time,
               std::unique_ptr<classic::SearchStopper> stopper, bool infinite,
               bool ponder, const OptionsDict& options, TranspositionTable* tt,
               SyzygyTablebase* syzygy_tb)
    : ok_to_respond_bestmove_(!infinite && !ponder),
      stopper_(std::move(stopper)),
      root_node_(tree.GetCurrentHead()),
      tt_(tt),
      syzygy_tb_(syzygy_tb),
      played_history_(tree.GetPositionHistory()),
      backend_(backend),
      backend_attributes_(backend->GetAttributes()),
      params_(options),
      searchmoves_(searchmoves),
      start_time_(start_time),
      initial_visits_(root_node_->GetN()),
      root_move_filter_(MakeRootMoveFilter(
          searchmoves_, syzygy_tb_, played_history_,
          params_.GetSyzygyFastPlay(), &tb_hits_, &root_is_in_dtz_)),
      uci_responder_(std::move(uci_responder)) {
  // Evict expired entries from the transposition table.
  // Garbage collection may lead to expiration at any time so this is not
  // enough to prevent expired entries later during the search.
  absl::erase_if(*tt_, [](const auto& item) { return item.second.expired(); });

  LOGFILE << "Transposition table garbage collection done.";

  if (params_.GetMaxConcurrentSearchers() != 0) {
    pending_searchers_.store(params_.GetMaxConcurrentSearchers(),
                             std::memory_order_release);
  }
  contempt_mode_ = params_.GetContemptMode();
  // Make sure the contempt mode is never "play" beyond this point.
  if (contempt_mode_ == ContemptMode::PLAY) {
    if (infinite) {
      // For infinite search disable contempt, only "white"/"black" make sense.
      contempt_mode_ = ContemptMode::NONE;
      // Issue a warning only if contempt mode would have an effect.
      if (params_.GetWDLRescaleDiff() != 0.0f) {
        std::vector<ThinkingInfo> info(1);
        info.back().comment =
            "WARNING: Contempt mode set to 'disable' as 'play' not supported "
            "for infinite search.";
        uci_responder_->OutputThinkingInfo(&info);
      }
    } else {
      // Otherwise set it to the root move's side, unless pondering.
      contempt_mode_ = played_history_.IsBlackToMove() != ponder
                           ? ContemptMode::BLACK
                           : ContemptMode::WHITE;
    }
  }
}

namespace {
void ApplyDirichletNoise(LowNode* node, float eps, double alpha) {
  float total = 0;
  std::vector<float> noise;

  for (int i = 0; i < node->GetNumEdges(); ++i) {
    float eta = Random::Get().GetGamma(alpha, 1.0);
    noise.emplace_back(eta);
    total += eta;
  }

  if (total < std::numeric_limits<float>::min()) return;

  int noise_idx = 0;
  auto edges = node->GetEdges();
  std::transform(edges, edges + node->GetNumEdges(), edges,
      [&](auto edge) {
        edge.SetP(edge.GetP() * (1 - eps) + eps * noise[noise_idx++] / total);
        return edge;
      });
}
}  // namespace

namespace {
// WDL conversion formula based on random walk model.
inline double WDLRescale(float& v, float& d, float wdl_rescale_ratio,
                         float wdl_rescale_diff, float sign, bool invert,
                         float max_reasonable_s) {
  if (invert) {
    wdl_rescale_diff = -wdl_rescale_diff;
    wdl_rescale_ratio = 1.0f / wdl_rescale_ratio;
  }
  auto w = (1 + v - d) / 2;
  auto l = (1 - v - d) / 2;
  // Safeguard against numerical issues; skip WDL transformation if WDL is too
  // extreme.
  const float eps = 0.0001f;
  if (w > eps && d > eps && l > eps && w < (1.0f - eps) && d < (1.0f - eps) &&
      l < (1.0f - eps)) {
    auto a = FastLog(1 / l - 1);
    auto b = FastLog(1 / w - 1);
    auto s = 2 / (a + b);
    // Safeguard against unrealistically broad WDL distributions coming from
    // the NN. Originally hardcoded, made into a parameter for piece odds.
    if (!invert) s = std::min(max_reasonable_s, s);
    auto mu = (a - b) / (a + b);
    auto s_new = s * wdl_rescale_ratio;
    if (invert) {
      std::swap(s, s_new);
      s = std::min(max_reasonable_s, s);
    }
    auto mu_new = mu + sign * s * s * wdl_rescale_diff;
    auto w_new = FastLogistic((-1.0f + mu_new) / s_new);
    auto l_new = FastLogistic((-1.0f - mu_new) / s_new);
    v = w_new - l_new;
    d = std::max(0.0f, 1.0f - w_new - l_new);
    return mu_new;
  }
  return 0;
}
}  // namespace

void Search::SendUciInfo(const classic::IterationStats& stats)
                         REQUIRES(nodes_mutex_) REQUIRES(counters_mutex_) {
  const auto max_pv = params_.GetMultiPv();
  const auto edges = GetBestChildrenNoTemperature(root_node_, max_pv, 0);
  const auto score_type = params_.GetScoreType();
  const auto per_pv_counters = params_.GetPerPvCounters();
  const auto draw_score = GetDrawScore(false);

  std::vector<ThinkingInfo> uci_infos;

  // Info common for all multipv variants.
  ThinkingInfo common_info;
  common_info.depth = cum_depth_ / (total_playouts_ ? total_playouts_ : 1);
  common_info.seldepth = max_depth_;
  common_info.time = stats.time_since_movestart;
  if (!per_pv_counters) {
    common_info.nodes = total_playouts_ + initial_visits_;
  }
  if (stats.time_since_first_batch) {
    const auto time_since_first_batch_ms = stats.time_since_first_batch;
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
    auto wl = edge.GetWL(default_wl);
    auto d = edge.GetD(default_d);
    float mu_uci = 0.0f;
    if (score_type == "WDL_mu" || (params_.GetWDLRescaleDiff() != 0.0f &&
                                   contempt_mode_ != ContemptMode::NONE)) {
      auto sign = ((contempt_mode_ == ContemptMode::BLACK) ==
                   played_history_.IsBlackToMove())
                      ? 1.0f
                      : -1.0f;
      mu_uci = WDLRescale(
          wl, d, params_.GetWDLRescaleRatio(),
          contempt_mode_ == ContemptMode::NONE
              ? 0
              : params_.GetWDLRescaleDiff() * params_.GetWDLEvalObjectivity(),
          sign, true, params_.GetWDLMaxS());
    }
    const auto q = edge.GetQ(default_q, draw_score);
    if (edge.IsTerminal() && wl != 0.0f) {
      uci_info.mate = std::copysign(
          std::round(edge.GetM(0.0f) + 1) / 2 + (edge.IsTbTerminal() ? 100 : 0),
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
    } else if (score_type == "WDL_mu") {
      // Reports the WDL mu value whenever it is reasonable, and defaults to
      // centipawn otherwise.
      const float centipawn_fallback_threshold = 0.996f;
      float centipawn_score = 45 * tan(1.56728071628 * wl);
      uci_info.score =
          backend_attributes_.has_wdl && mu_uci != 0.0f &&
                  std::abs(wl) + d < centipawn_fallback_threshold &&
                  (std::abs(mu_uci) < 1.0f ||
                   std::abs(centipawn_score) < std::abs(100 * mu_uci))
              ? 100 * mu_uci
              : centipawn_score;
    }

    auto wdl_w =
        std::max(0, static_cast<int>(std::round(500.0 * (1.0 + wl - d))));
    auto wdl_l =
        std::max(0, static_cast<int>(std::round(500.0 * (1.0 - wl - d))));
    // Using 1000-w-l so that W+D+L add up to 1000.0.
    auto wdl_d = 1000 - wdl_w - wdl_l;
    if (wdl_d < 0) {
      wdl_w = std::min(1000, std::max(0, wdl_w + wdl_d / 2));
      wdl_l = 1000 - wdl_w;
      wdl_d = 0;
    }
    uci_info.wdl = ThinkingInfo::WDL{wdl_w, wdl_d, wdl_l};
    if (backend_attributes_.has_mlh) {
      uci_info.moves_left = static_cast<int>(
          (1.0f + edge.GetM(1.0f + root_node_->GetM())) / 2.0f);
    }
    if (max_pv > 1) uci_info.multipv = multipv;
    if (per_pv_counters) uci_info.nodes = edge.GetN();
    bool flip = played_history_.IsBlackToMove();
    int depth = 0;
    auto history = played_history_;
    for (auto iter = edge; iter;
         iter = GetBestChildNoTemperature(iter.node(), depth), flip = !flip) {
      uci_info.pv.push_back(iter.GetMove(flip));
      history.Append(iter.GetMove());
      // Last edge was dangling or a draw by repetition, cannot continue.
      if (!iter.node() || history.Last().GetRepetitions() >= 2) break;
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
void Search::MaybeOutputInfo(const classic::IterationStats& stats) {
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
    SendUciInfo(stats);
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

void Search::RecordNPSStartTime() {
  if (nps_start_time_) return;
  nps_start_time_ = std::chrono::steady_clock::now();
}

// Root is depth 0, i.e. even depth.
float Search::GetDrawScore(bool is_odd_depth) const {
  return (is_odd_depth == played_history_.IsBlackToMove()
              ? params_.GetDrawScore()
              : -params_.GetDrawScore());
}

namespace {
inline float GetFpu(const SearchParams& params, const Node* node, bool is_root_node,
                    float draw_score) {
  const auto value = params.GetFpuValue(is_root_node);
  return params.GetFpuAbsolute(is_root_node)
             ? value
             : -node->GetQ(-draw_score) -
                   value * std::sqrt(node->GetVisitedPolicy());
}

// Faster version for if visited_policy is readily available already.
inline float GetFpu(const SearchParams& params, const Node* node, bool is_root_node,
                    float draw_score, float visited_pol) {
  const auto value = params.GetFpuValue(is_root_node);
  return params.GetFpuAbsolute(is_root_node)
             ? value
             : -node->GetQ(-draw_score) - value * std::sqrt(visited_pol);
}

inline float ComputeCpuct(const SearchParams& params, uint32_t N,
                          bool is_root_node) {
  const float init = params.GetCpuct(is_root_node);
  const float k = params.GetCpuctFactor(is_root_node);
  const float base = params.GetCpuctBase(is_root_node);
  return init + (k ? k * FastLog((N + base) / base) : 0.0f);
}
}  // namespace

// Ignore the last tuple element when sorting in GetVerboseStats
static bool operator<(const EdgeAndNode&, const EdgeAndNode&) { return false; }

std::vector<std::string> Search::GetVerboseStats(
    const Node* node, std::optional<Move> move_to_node) const {
  const bool is_root = (node == root_node_);
  const bool is_odd_depth = !is_root;
  const bool is_black_to_move = (played_history_.IsBlackToMove() == is_root);
  const float draw_score = GetDrawScore(is_odd_depth);
  const float fpu = GetFpu(params_, node, is_root, draw_score);
  const float cpuct = ComputeCpuct(params_, node->GetTotalVisits(), is_root);
  const float U_coeff =
      cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
  std::vector<std::tuple<uint32_t, float, EdgeAndNode>> edges;
  edges.reserve(node->GetNumEdges());
  for (const auto& edge : node->Edges()) {
    edges.emplace_back(edge.GetN(),
                       edge.GetQ(fpu, draw_score) + edge.GetU(U_coeff),
                       edge);
  }
  std::sort(edges.begin(), edges.end());

  auto print = [](auto* oss, auto pre, auto v, auto post, auto w, int p = 0) {
    *oss << pre << std::setw(w) << std::setprecision(p) << v << post;
  };
  auto print_head = [&](auto* oss, auto label, int i, auto n, auto f, auto p) {
    *oss << std::fixed;
    print(oss, "", label, " ", 5);
    print(oss, "(", i, ") ", 4);
    *oss << std::right;
    print(oss, "N: ", n, " ", 7);
    print(oss, "(+", f, ") ", 2);
    print(oss, "(P: ", p * 100, "%) ", 5, p >= 0.99995f ? 1 : 2);
  };
  auto print_stats = [&](auto* oss, const auto* n) {
    const auto sign = n == node ? -1 : 1;
    if (n) {
      auto wl = sign * n->GetWL();
      auto d = n->GetD();
      auto is_perspective = ((contempt_mode_ == ContemptMode::BLACK) ==
                             played_history_.IsBlackToMove())
                                ? 1.0f
                                : -1.0f;
      WDLRescale(
          wl, d, params_.GetWDLRescaleRatio(),
          contempt_mode_ == ContemptMode::NONE
              ? 0
              : params_.GetWDLRescaleDiff() * params_.GetWDLEvalObjectivity(),
          is_perspective, true, params_.GetWDLMaxS());
      print(oss, "(WL: ", wl, ") ", 8, 5);
      print(oss, "(D: ", d, ") ", 5, 3);
      print(oss, "(M: ", n->GetM(), ") ", 4, 1);
      print(oss, "(Q: ", wl + draw_score * d, ") ", 8, 5);
    } else {
      *oss << "(WL:  -.-----) (D: -.---) (M:  -.-) ";
      print(oss, "(Q: ", fpu, ") ", 8, 5);
    }
  };
  auto print_tail = [&](auto* oss, const auto* n, bool is_edge) {
    const auto sign = n == node ? -1 : 1;
    std::optional<float> v;
    if (n && n->IsTerminal()) {
      v = n->GetQ(sign * draw_score);
    } else if (n) {
      auto history = played_history_;
      if (move_to_node) {
        history.Append(*move_to_node);
      }
      if (is_edge) {
        history.Append(n->GetMove());
      }
      std::optional<EvalResult> nneval = backend_->GetCachedEvaluation(
          EvalPosition{history.GetPositions(), {}});
      if (nneval) v = -nneval->q;
    }
    if (v) {
      print(oss, "(V: ", sign * *v, ") ", 7, 4);
    } else {
      *oss << "(V:  -.----) ";
    }

    if (n) {
      auto [lo, up] = n->GetBounds();
      if (sign == -1) {
        lo = -lo;
        up = -up;
        std::swap(lo, up);
      }
      *oss << (lo == up                                                ? "(T) "
               : lo == GameResult::DRAW && up == GameResult::WHITE_WON ? "(W) "
               : lo == GameResult::BLACK_WON && up == GameResult::DRAW ? "(L) "
                                                                       : "");
    }
  };

  std::vector<std::string> infos;
  const auto m_evaluator =
      backend_attributes_.has_mlh ? MEvaluator(params_, node) : MEvaluator();
  for (const auto& edge_tuple : edges) {
    const auto& edge = std::get<2>(edge_tuple);
    float Q = edge.GetQ(fpu, draw_score);
    float M = m_evaluator.GetMUtility(edge, Q);
    std::ostringstream oss;
    oss << std::left;
    // TODO: should this be displaying transformed index?
    print_head(&oss, edge.GetMove(is_black_to_move).ToString(true),
               MoveToNNIndex(edge.GetMove(), 0), edge.GetN(),
               edge.GetNInFlight(), edge.GetP());
    print_stats(&oss, edge.node());
    print(&oss, "(U: ", edge.GetU(U_coeff), ") ", 6, 5);
    print(&oss, "(S: ", Q + edge.GetU(U_coeff) + M, ") ", 8, 5);
    print_tail(&oss, edge.node(), true);
    infos.emplace_back(oss.str());
  }

  // Include stats about the node in similar format to its children above.
  std::ostringstream oss;
  print_head(&oss, "node ", node->GetNumEdges(), node->GetN(),
             node->GetNInFlight(), node->GetVisitedPolicy());
  print_stats(&oss, node);
  print_tail(&oss, node, false);
  infos.emplace_back(oss.str());
  return infos;
}

void Search::SendMovesStats() const REQUIRES(counters_mutex_) {
  auto move_stats = GetVerboseStats(root_node_, std::nullopt);

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
  for (auto& edge : root_node_->Edges()) {
    if (!(edge.GetMove(played_history_.IsBlackToMove()) == final_bestmove_)) {
      continue;
    }
    if (edge.HasNode()) {
      LOGFILE << "--- Opponent moves after: " << final_bestmove_.ToString(true);
      for (const auto& line : GetVerboseStats(edge.node(), edge.GetMove())) {
        LOGFILE << line;
      }
    }
  }
}

void Search::MaybeTriggerStop(const classic::IterationStats& stats,
                              classic::StoppersHints* hints) {
  hints->Reset();
  if (params_.GetNpsLimit() > 0) {
    hints->UpdateEstimatedNps(params_.GetNpsLimit());
  }
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  // Already responded bestmove, nothing to do here.
  if (bestmove_is_sent_) return;
  // Don't stop when the root node is not yet expanded.
  if (stats.total_nodes == 0) return;

  if (!stop_.load(std::memory_order_acquire)) {
    const float delay = params_.GetGarbageCollectionDelay() / 100.0f;
    if (stopper_->ShouldStop(stats, hints)) {
      FireStopInternal();
    } else if (!gc_started_ &&
        stats.time_since_movestart > delay *
        (stats.time_since_movestart + hints->GetEstimatedRemainingTimeMs())) {
      NodeGarbageCollector::Instance().Start();
      gc_started_ = true;
    }
  }

  // If we are the first to see that stop is needed.
  if (stop_.load(std::memory_order_acquire) && ok_to_respond_bestmove_ &&
      !bestmove_is_sent_) {
    SendUciInfo(stats);
    EnsureBestMoveKnown();
    SendMovesStats();
    BestMoveInfo info(final_bestmove_, final_pondermove_);
    uci_responder_->OutputBestMove(&info);
    stopper_->OnSearchDone(stats);
    bestmove_is_sent_ = true;
    current_best_edge_ = EdgeAndNode();
    NodeGarbageCollector::Instance().Stop();
  }
}

// Return the evaluation of the actual best child, regardless of temperature
// settings. This differs from GetBestMove, which does obey any temperature
// settings. So, somethimes, they may return results of different moves.
Eval Search::GetBestEval(Move* move, bool* is_terminal) const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  float parent_wl = -root_node_->GetWL();
  float parent_d = root_node_->GetD();
  float parent_m = root_node_->GetM();
  if (!root_node_->HasChildren()) return {parent_wl, parent_d, parent_m};
  EdgeAndNode best_edge = GetBestChildNoTemperature(root_node_, 0);
  if (move) *move = best_edge.GetMove(played_history_.IsBlackToMove());
  if (is_terminal) *is_terminal = best_edge.IsTerminal();
  return {best_edge.GetWL(parent_wl), best_edge.GetD(parent_d),
          best_edge.GetM(parent_m - 1) + 1};
}

std::pair<Move, Move> Search::GetBestMove() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  EnsureBestMoveKnown();
  return {final_bestmove_, final_pondermove_};
}

std::int64_t Search::GetTotalPlayouts() const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  return total_playouts_;
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
  if (root_node_->GetN() == 0) return;
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
      temperature *=
          static_cast<float>(decay_delay_moves + decay_moves - moves) /
          decay_moves;
    }
    // don't allow temperature to decay below endgame temperature
    if (temperature < params_.GetTemperatureEndgame()) {
      temperature = params_.GetTemperatureEndgame();
    }
  }

  auto bestmove_edge = temperature
                           ? GetBestRootChildWithTemperature(temperature)
                           : GetBestChildNoTemperature(root_node_, 0);
  final_bestmove_ = bestmove_edge.GetMove(played_history_.IsBlackToMove());

  if (bestmove_edge.GetN() > 0 && bestmove_edge.node()->HasChildren()) {
    final_pondermove_ = GetBestChildNoTemperature(bestmove_edge.node(), 1)
                            .GetMove(!played_history_.IsBlackToMove());
  }
}

// Returns @count children with most visits.
std::vector<EdgeAndNode> Search::GetBestChildrenNoTemperature(Node* parent,
                                                              int count,
                                                              int depth) const {
  // Even if Edges is populated at this point, its a race condition to access
  // the node, so exit quickly.
  if (parent->GetN() == 0) return {};
  const bool is_odd_depth = (depth % 2) == 1;
  const float draw_score = GetDrawScore(is_odd_depth);
  // Best child is selected using the following criteria:
  // * Prefer shorter terminal wins / avoid shorter terminal losses.
  // * Largest number of playouts.
  // * If two nodes have equal number:
  //   * If that number is 0, the one with larger prior wins.
  //   * If that number is larger than 0, the one with larger eval wins.
  std::vector<EdgeAndNode> edges;
  for (auto& edge : parent->Edges()) {
    if (parent == root_node_ && !root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
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
          // Not safe to access IsTerminal if GetN is 0.
          if (edge.GetN() == 0 || !edge.IsTerminal() || !wl) {
            return kNonTerminal;
          }
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
        // Not safe to access IsTerminal if GetN is 0.
        if (a_rank == kNonTerminal && a.GetN() != 0 && b.GetN() != 0 &&
            a.IsTerminal() && b.IsTerminal()) {
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
          if (a.GetQ(0.0f, draw_score) != b.GetQ(0.0f, draw_score)) {
            return a.GetQ(0.0f, draw_score) > b.GetQ(0.0f, draw_score);
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

  std::vector<float> cumulative_sums;
  float sum = 0.0;
  float max_n = 0.0;
  const float offset = params_.GetTemperatureVisitOffset();
  float max_eval = -1.0f;
  const float fpu =
      GetFpu(params_, root_node_, /* is_root= */ true, draw_score);

  for (auto& edge : root_node_->Edges()) {
    if (!root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      continue;
    }
    if (edge.GetN() + offset > max_n) {
      max_n = edge.GetN() + offset;
      max_eval = edge.GetQ(fpu, draw_score);
    }
  }

  // TODO(crem) Simplify this code when samplers.h is merged.
  const float min_eval =
      max_eval - params_.GetTemperatureWinpctCutoff() / 50.0f;
  for (auto& edge : root_node_->Edges()) {
    if (!root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      continue;
    }
    if (edge.GetQ(fpu, draw_score) < min_eval) continue;
    sum += std::pow(
        std::max(0.0f,
                 (max_n <= 0.0f
                      ? edge.GetP()
                      : ((static_cast<float>(edge.GetN()) + offset) / max_n))),
        1 / temperature);
    cumulative_sums.push_back(sum);
  }
  assert(sum);

  const float toss = Random::Get().GetFloat(cumulative_sums.back());
  int idx =
      std::lower_bound(cumulative_sums.begin(), cumulative_sums.end(), toss) -
      cumulative_sums.begin();

  for (auto& edge : root_node_->Edges()) {
    if (!root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      continue;
    }
    if (edge.GetQ(fpu, draw_score) < min_eval) continue;
    if (idx-- == 0) return edge;
  }
  assert(false);
  return {};
}

void Search::StartThreads(size_t how_many) {
  Mutex::Lock lock(threads_mutex_);
  if (how_many == 0 && threads_.size() == 0) {
    how_many = backend_attributes_.suggested_num_search_threads +
               !backend_attributes_.runs_on_cpu;
  }
  thread_count_.store(how_many, std::memory_order_release);
  // First thread is a watchdog thread.
  if (threads_.size() == 0) {
    threads_.emplace_back([this]() { WatchdogThread(); });
  }
  // Start working threads.
  for (size_t i = 0; i < how_many; i++) {
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

void Search::PopulateCommonIterationStats(classic::IterationStats* stats) {
  stats->time_since_movestart = GetTimeSinceStart();

  SharedMutex::SharedLock nodes_lock(nodes_mutex_);
  stats->time_since_first_batch = GetTimeSinceFirstBatch();
  stats->total_nodes = total_playouts_ + initial_visits_;
  stats->nodes_since_movestart = total_playouts_;
  stats->batches_since_movestart = total_batches_;
  stats->average_depth = cum_depth_ / (total_playouts_ ? total_playouts_ : 1);
  stats->edge_n.clear();
  stats->win_found = false;
  stats->may_resign = true;
  stats->num_losing_edges = 0;
  stats->time_usage_hint_ = classic::IterationStats::TimeUsageHint::kNormal;
  stats->mate_depth = std::numeric_limits<int>::max();

  // If root node hasn't finished first visit, none of this code is safe.
  if (root_node_->GetN() > 0) {
    const auto draw_score = GetDrawScore(true);
    const float fpu =
        GetFpu(params_, root_node_, /* is_root_node */ true, draw_score);
    float max_q_plus_m = -1000;
    uint64_t max_n = 0;
    bool max_n_has_max_q_plus_m = true;
    const auto m_evaluator = backend_attributes_.has_mlh
                                 ? MEvaluator(params_, root_node_)
                                 : MEvaluator();
    for (const auto& edge : root_node_->Edges()) {
      const auto n = edge.GetN();
      const auto q = edge.GetQ(fpu, draw_score);
      const auto m = m_evaluator.GetMUtility(edge, q);
      const auto q_plus_m = q + m;
      stats->edge_n.push_back(n);
      if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) > 0.0f) {
        stats->win_found = true;
      }
      if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) < 0.0f) {
        stats->num_losing_edges += 1;
      }
      if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) == 1.0f &&
          !edge.IsTbTerminal()) {
        stats->mate_depth =
            std::min(stats->mate_depth,
                     static_cast<int>(std::round(edge.GetM(0.0f))) / 2 + 1);
      }

      // If game is resignable, no need for moving quicker. This allows
      // proving mate when losing anyway for better score output.
      // Hardcoded resign threshold, because there is no available parameter.
      if (n > 0 && q > -0.98f) {
        stats->may_resign = false;
      }
      if (max_n < n) {
        max_n = n;
        max_n_has_max_q_plus_m = false;
      }
      if (max_q_plus_m <= q_plus_m) {
        max_n_has_max_q_plus_m = (max_n == n);
        max_q_plus_m = q_plus_m;
      }
    }
    if (!max_n_has_max_q_plus_m) {
      stats->time_usage_hint_ =
          classic::IterationStats::TimeUsageHint::kNeedMoreTime;
    }
  }
}

void Search::WatchdogThread() {
  LOGFILE << "Start a watchdog thread.";
  classic::StoppersHints hints;
  classic::IterationStats stats;
  while (true) {
    PopulateCommonIterationStats(&stats);
    MaybeTriggerStop(stats, &hints);
    MaybeOutputInfo(stats);

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
  NodeGarbageCollector::Instance().Stop();
  Mutex::Lock lock(counters_mutex_);
  ok_to_respond_bestmove_ = true;
  FireStopInternal();
  LOGFILE << "Stopping search due to `stop` uci command.";
}

void Search::Abort() {
  NodeGarbageCollector::Instance().Abort();
  Mutex::Lock lock(counters_mutex_);
  if (!stop_.load(std::memory_order_acquire) ||
      (!bestmove_is_sent_ && !ok_to_respond_bestmove_)) {
    bestmove_is_sent_ = true;
    FireStopInternal();
  }
  LOGFILE << "Aborting search, if it is still active.";
}

void Search::Wait() {
  NodeGarbageCollector::Instance().Wait();
  Mutex::Lock lock(threads_mutex_);
  bool active_threads = !threads_.empty();
  while (!threads_.empty()) {
    threads_.back().join();
    threads_.pop_back();
  }
  if (active_threads) {
    SharedMutex::Lock lock(nodes_mutex_);

    assert(root_node_->ZeroNInFlight());
  }
  LOGFILE << "Search threads cleaned.";
}

void SearchWorker::CancelCollisions() {
  for (auto& entry : minibatch_) {
    if (!entry.IsCollision()) continue;
    auto path = entry.path;
    for (auto it = ++(path.crbegin()); it != path.crend(); ++it) {
      std::get<0>(*it)->CancelScoreUpdate(entry.multivisit);
    }
  }
}

Search::~Search() {
  Abort();
  Wait();
  LOGFILE << "Search destroyed.";
}

//////////////////////////////////////////////////////////////////////////////
// SearchWorker
//////////////////////////////////////////////////////////////////////////////

SearchWorker::~SearchWorker()
{
  {
    // Tasks must be completed before destructor. If a gather tasks is running,
    // it can increment task_count_ which would break the exit state.
    assert(IsTasksCompleted(task_count_, completed_tasks_));
    task_count_.fetch_or(kTaskCountSuspend, std::memory_order_release);
    Mutex::Lock lock(picking_tasks_mutex_);
    exiting_ = true;
    task_added_.notify_all();
  }
  for (size_t i = 0; i < task_threads_.size(); i++) {
    task_threads_[i].join();
  }
  LOGFILE << "Search worker destroyed.";
}

std::tuple<SearchWorker::PickTask*, int, int> SearchWorker::PickTaskToProcess() {
  auto [packed_value, nta, tc] = ReadTaskCount(task_count_);

  // Check if tasks are queued and try increment taken count.
  while (nta < tc &&
      !task_count_.compare_exchange_weak(packed_value, packed_value + kTasksTakenOne,
                                         std::memory_order_acq_rel)) {
    // Queue had tasks but another worker increment taken. We check
    // if new work was added to the queue. Then we try to increment
    // taken again.
    std::tie(packed_value, nta, tc) = ReadTaskCount(packed_value);
  }
  // We incremented taken if nta and tc are different
  if (nta < tc) {
    return {picking_tasks_.data() + nta, nta, tc};
  }
  return {nullptr, nta, tc};
}

void SearchWorker::ProcessTask(PickTask* task, int id,
                               std::vector<NodeToProcess>* receiver,
                               TaskWorkspace* workspace) {
  switch (task->task_type) {
    case PickTask::kGathering: {
      PickNodesToExtendTask(task->start_path, task->collision_limit,
                            task->history, receiver,
                            workspace);
      break;
    }
    case PickTask::kProcessing: {
      ProcessPickedTask(task->start_idx, task->end_idx);
      break;
    }
  }
  picking_tasks_.data()[id].complete = true;
  completed_tasks_.fetch_add(1, std::memory_order_acq_rel);
}

void SearchWorker::RunTasks(int tid) {
  while (true) {
    PickTask* task = nullptr;
    int id = 0;
    int tc = 0;
    {
      int spins = 0;
      while (true) {
        std::tie(task, id, tc) = PickTaskToProcess();
        if (task) {
          break;
        } else if (tc != -1) {
          spins++;
          if (spins >= 512) {
            std::this_thread::yield();
            spins = 0;
          } else {
            SpinloopPause();
          }
          continue;
        }
        spins = 0;
        // Looks like sleep time.
        Mutex::Lock lock(picking_tasks_mutex_);
        // Refresh them now we have the lock.
        int tc, nta;
        std::tie(std::ignore, std::ignore, tc) = ReadTaskCount(task_count_);
        if (tc != -1) continue;
        if (exiting_) return;
        task_added_.wait(lock.get_raw());
        std::tie(std::ignore, nta, tc) = ReadTaskCount(task_count_);
        if (nta >= tc && exiting_) return;
      }
    }
    if (task != nullptr) {
      ProcessTask(task, id, &(task->results), &(task_workspaces_[tid]));
    }
  }
}

void SearchWorker::ExecuteOneIteration() {
  // 1. Initialize internal structures.
  InitializeIteration();

  if (params_.GetMaxConcurrentSearchers() != 0) {
    std::unique_ptr<SpinHelper> spin_helper;
    if (params_.GetSearchSpinBackoff()) {
      spin_helper = std::make_unique<ExponentialBackoffSpinHelper>();
    } else {
      // This is a hard spin lock to reduce latency but at the expense of busy
      // wait cpu usage. If search worker count is large, this is probably a
      // bad idea.
      spin_helper = std::make_unique<SpinHelper>();
    }

    while (true) {
      // If search is stopped, we've not gathered or done anything and we don't
      // want to, so we can safely skip all below. But make sure we have done
      // at least one iteration.
      if (search_->stop_.load(std::memory_order_acquire) &&
          search_->GetTotalPlayouts() + search_->initial_visits_ > 0) {
        return;
      }

      int available =
          search_->pending_searchers_.load(std::memory_order_acquire);
      if (available == 0) {
        spin_helper->Wait();
        continue;
      }

      if (search_->pending_searchers_.compare_exchange_weak(
              available, available - 1, std::memory_order_acq_rel)) {
        break;
      } else {
        spin_helper->Backoff();
      }
    }
  }

  // 2. Gather minibatch.
  GatherMinibatch();
  assert(IsTasksCompleted(task_count_, completed_tasks_));
  task_count_.fetch_or(kTaskCountSuspend, std::memory_order_release);
  search_->backend_waiting_counter_.fetch_add(1, std::memory_order_relaxed);

  if (params_.GetMaxConcurrentSearchers() != 0) {
    search_->pending_searchers_.fetch_add(1, std::memory_order_acq_rel);
  }

  // 4. Run NN computation.
  RunNNComputation();
  search_->backend_waiting_counter_.fetch_add(-1, std::memory_order_relaxed);

  // 5. Retrieve NN computations (and terminal values) into nodes.
  FetchMinibatchResults();

  // 6. Propagate the new nodes' information to all their parents in the tree.
  DoBackupUpdate();

  // 7. Update the Search's status and progress information.
  UpdateCounters();

  // If required, waste time to limit nps.
  if (params_.GetNpsLimit() > 0 && iteration_stats_.time_since_first_batch) {
    while (search_->IsSearchActive()) {
      // GetTimeSinceFirstBatch is set only once. We check iteration_stats_ to
      // know if it was set and later read inside nodes_mutex_.
      int64_t time_since_first_batch_ms = search_->GetTimeSinceFirstBatch();
      auto nps = search_->GetTotalPlayouts() * 1e3f / time_since_first_batch_ms;
      if (nps > params_.GetNpsLimit()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } else {
        break;
      }
    }
  }
}

// 1. Initialize internal structures.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::InitializeIteration() {
  // Free the old computation before allocating a new one. This works better
  // when backend caches buffer allocations between computations.
  computation_.reset();
  computation_ = search_->backend_->CreateComputation();
  minibatch_.clear();
  minibatch_.reserve(2 * target_minibatch_size_);
}

// 2. Gather minibatch.
// ~~~~~~~~~~~~~~~~~~~~
namespace {
int Mix(int high, int low, float ratio) {
  return static_cast<int>(std::round(static_cast<float>(low) +
                                     static_cast<float>(high - low) * ratio));
}

int CalculateCollisionsLeft(int64_t nodes, const SearchParams& params) {
  // End checked first
  if (nodes >= params.GetMaxCollisionVisitsScalingEnd()) {
    return params.GetMaxCollisionVisits();
  }
  if (nodes <= params.GetMaxCollisionVisitsScalingStart()) {
    return 1;
  }
  return Mix(params.GetMaxCollisionVisits(), 1,
             std::pow((static_cast<float>(nodes) -
                       params.GetMaxCollisionVisitsScalingStart()) /
                          (params.GetMaxCollisionVisitsScalingEnd() -
                           params.GetMaxCollisionVisitsScalingStart()),
                      params.GetMaxCollisionVisitsScalingPower()));
}
}  // namespace

void SearchWorker::GatherMinibatch() {
  // Total number of nodes to process.
  int minibatch_size = 0;
  int cur_n = 0;

  // Collision use atomic operations. We can cancel them outside the lock.
  struct CollisionsManager {
    SearchWorker& worker;
    CollisionsManager(SearchWorker& worker) : worker(worker) {
    }
    ~CollisionsManager() {
      worker.CancelCollisions();
    }
  } cancel_collisions_object(*this);
  // We take the nodes_mutex_ only once to avoid bouncing between this thread
  // and a thread returning from RunNNComputation.
  SharedMutex::Lock lock(search_->nodes_mutex_);
  cur_n = search_->root_node_->GetN();
  // TODO: GetEstimatedRemainingPlayouts has already had smart pruning factor
  // applied, which doesn't clearly make sense to include here...
  int64_t remaining_n =
      latest_time_manager_hints_.GetEstimatedRemainingPlayouts();
  int collisions_left = CalculateCollisionsLeft(
      std::min(static_cast<int64_t>(cur_n), remaining_n), params_);

  // Number of nodes processed out of order.
  number_out_of_order_ = 0;

  int thread_count = search_->thread_count_.load(std::memory_order_acquire);

  absl::Cleanup record_batch_start_time = [&] {
    if (minibatch_size) search_->RecordNPSStartTime();
  };

  // Gather nodes to process in the current batch.
  // If we had too many nodes out of order, also interrupt the iteration so
  // that search can exit.
  while (minibatch_size < target_minibatch_size_ &&
         number_out_of_order_ < max_out_of_order_) {
    // If there's something to process without touching slow neural net, do it.
    if (minibatch_size > 0 && computation_->UsedBatchSize() == 0) return;

    // If there is backend work to be done, and the backend is idle - exit
    // immediately.
    // Only do this fancy work if there are multiple threads as otherwise we
    // early exit from every batch since there is never another search thread to
    // be keeping the backend busy. Which would mean that threads=1 has a
    // massive nps drop.
    if (thread_count > 1 && minibatch_size > 0 &&
        static_cast<int>(computation_->UsedBatchSize()) >
            params_.GetIdlingMinimumWork() &&
        thread_count - search_->backend_waiting_counter_.load(
                           std::memory_order_relaxed) >
            params_.GetThreadIdlingThreshold()) {
      return;
    }

    int new_start = static_cast<int>(minibatch_.size());

    PickNodesToExtend(
        std::min({collisions_left, target_minibatch_size_ - minibatch_size,
                  max_out_of_order_ - number_out_of_order_}));

    // Count the non-collisions.
    int non_collisions = 0;
    for (int i = new_start; i < static_cast<int>(minibatch_.size()); i++) {
      auto& picked_node = minibatch_[i];
      if (picked_node.IsCollision()) {
        continue;
      }
      ++non_collisions;
      ++minibatch_size;
    }

    {

      bool needs_wait = false;
      int ppt_start = new_start;
      if (task_workers_ > 0 &&
          non_collisions >= params_.GetMinimumWorkSizeForProcessing()) {
        const int num_tasks = std::clamp(
            non_collisions / params_.GetMinimumWorkPerTaskForProcessing(), 2,
            task_workers_ + 1);
        // Round down, left overs can go to main thread so it waits less.
        int per_worker = non_collisions / num_tasks;
        needs_wait = true;
        ResetTasks();
        int found = 0;
        for (int i = new_start; i < static_cast<int>(minibatch_.size()); i++) {
          auto& picked_node = minibatch_[i];
          if (picked_node.IsCollision()) {
            continue;
          }
          ++found;
          if (found == per_worker) {
            picking_tasks_.emplace_back(ppt_start, i + 1);
            task_count_.fetch_add(1, std::memory_order_acq_rel);
            ppt_start = i + 1;
            found = 0;
            if (picking_tasks_.size() == static_cast<size_t>(num_tasks - 1)) {
              break;
            }
          }
        }
      }
      ProcessPickedTask(ppt_start, static_cast<int>(minibatch_.size()));
      if (needs_wait) {
        WaitForTasks();
      }
    }
    bool some_ooo = false;
    for (int i = static_cast<int>(minibatch_.size()) - 1; i >= new_start; i--) {
      if (minibatch_[i].ooo_completed) {
        some_ooo = true;
        break;
      }
    }
    if (some_ooo) {
      for (int i = static_cast<int>(minibatch_.size()) - 1; i >= new_start;
           i--) {
        // If there was any OOO, revert 'all' new collisions - it isn't possible
        // to identify exactly which ones are afterwards and only prune those.
        // This may remove too many items, but hopefully most of the time they
        // will just be added back in the same in the next gather.
        if (minibatch_[i].IsCollision()) {
          for (auto it = ++(minibatch_[i].path.crbegin());
               it != minibatch_[i].path.crend(); ++it) {
            std::get<0>(*it)->CancelScoreUpdate(minibatch_[i].multivisit);
          }
          minibatch_.erase(minibatch_.begin() + i);
        } else if (minibatch_[i].ooo_completed) {
          FetchSingleNodeResult(&minibatch_[i]);
          DoBackupUpdateSingleNode(minibatch_[i]);
          minibatch_.erase(minibatch_.begin() + i);
          --minibatch_size;
          ++number_out_of_order_;
        }
      }
    }

    // Check for stop at the end so we have at least one node.
    for (size_t i = new_start; i < minibatch_.size(); i++) {
      auto& picked_node = minibatch_[i];

      if (picked_node.IsCollision()) {
        // Check to see if we can upsize the collision to exit sooner.
        if (picked_node.maxvisit > 0 &&
            collisions_left > picked_node.multivisit) {
          int extra = std::min(picked_node.maxvisit, collisions_left) -
                      picked_node.multivisit;
          picked_node.multivisit += extra;
          for (auto it = ++(picked_node.path.crbegin());
               it != picked_node.path.crend(); ++it) {
            std::get<0>(*it)->IncrementNInFlight(extra);
          }
        }
        if ((collisions_left -= picked_node.multivisit) <= 0) return;
        if (search_->stop_.load(std::memory_order_acquire)) return;
      }
    }
  }
}

void SearchWorker::ProcessPickedTask(int start_idx, int end_idx)
    REQUIRES(search_->nodes_mutex_) {
  for (int i = start_idx; i < end_idx; i++) {
    auto& picked_node = minibatch_[i];
    if (picked_node.IsCollision()) continue;
    // If node is a collision, known as a terminal (win/loss/draw according to
    // the rules of the game) or has a low node, it means that we have already
    // visited this node before and can't extend it.
    if (picked_node.IsExtendable()) {
      // Node was never visited, extend it.
      ExtendNode(picked_node);
    }

    picked_node.ooo_completed =
        params_.GetOutOfOrderEval() && picked_node.CanEvalOutOfOrder();
  }
}

#define MAX_TASKS 256

void SearchWorker::ResetTasks() {
  // Tasks must be completed before reset.
  assert(IsTasksCompleted(task_count_, completed_tasks_));
  task_count_.store(0, std::memory_order_release);
  completed_tasks_.store(0, std::memory_order_release);
  picking_tasks_.clear();
  // Reserve because resizing breaks pointers held by the task threads.
  picking_tasks_.reserve(MAX_TASKS);
}

int SearchWorker::WaitForTasks() REQUIRES(search_->nodes_mutex_) {
  // Process any outstanding tasks before checking if compelted. This avoids a
  // long polling loop when PickNodesToExtend scheduled many tasks.
  while (true) {
    PickTask* task = nullptr;
    int id = 0;
    std::tie(task, id, std::ignore) = PickTaskToProcess();
    if (task == nullptr) {
      break;
    }
    ProcessTask(task, id, &minibatch_, &main_workspace_);
  }
  // Spin lock, other tasks should be done soon.
  while (true) {
    int completed = completed_tasks_.load(std::memory_order_acquire);
    int todo, nta;
    std::tie(std::ignore, nta, todo) = ReadTaskCount(task_count_);
    std::ignore = nta;
    assert(nta <= todo);
    if (todo == completed) return completed;
    SpinloopPause();
  }
}

void SearchWorker::PickNodesToExtend(int collision_limit)
    REQUIRES(search_->nodes_mutex_) {
  ResetTasks();
  if (task_workers_ > 0 && !search_->backend_attributes_.runs_on_cpu) {
    // While nothing is ready yet - wake the task runners so they are ready to
    // receive quickly.
    Mutex::Lock lock(picking_tasks_mutex_);
    task_added_.notify_all();
  }
  std::vector<Move> empty_movelist;
  history_.Trim(search_->played_history_.GetLength());
  PickNodesToExtendTask({std::make_tuple(search_->root_node_, 0, 0)},
                        collision_limit, history_, &minibatch_,
                        &main_workspace_);

  WaitForTasks();
  for (int i = 0; i < static_cast<int>(picking_tasks_.size()); i++) {
    for (int j = 0; j < static_cast<int>(picking_tasks_[i].results.size());
         j++) {
      minibatch_.emplace_back(std::move(picking_tasks_[i].results[j]));
    }
  }
}

// Check if the situation described by @depth under root and @position is a
// safe two-fold or a draw by repetition and return the number of safe
// repetitions and moves_left.
// Depth starts with 0 at root, so number of plies in PV equals depth.
std::pair<int, int> SearchWorker::GetRepetitions(int depth,
                                                 const Position& position) {
  const auto repetitions = position.GetRepetitions();

  if (repetitions == 0) return {0, 0};

  if (repetitions >= 2) return {repetitions, 0};

  const auto plies = position.GetPliesSincePrevRepetition();
  if (params_.GetTwoFoldDraws() && /*repetitions == 1 &&*/ depth >= 4 &&
      depth >= plies) {
    return {1, plies};
  }

  return {0, 0};
}

// Check if PickNodesToExtendTask should stop picking at this @node.
bool SearchWorker::ShouldStopPickingHere(Node* node, bool is_root_node,
                                         int repetitions) {
  constexpr double wl_diff_limit = 0.01f;
  constexpr float d_diff_limit = 0.01f;
  constexpr float m_diff_limit = 2.0f;

  if (node->GetN() == 0 || node->IsTerminal()) return true;

  // Only stop at root when there is no other option.
  assert(!is_root_node || node == search_->root_node_);
  if (is_root_node) return false;

  // Stop at draws by repetition.
  if (repetitions >= 2) return true;

  // Check if Node and LowNode differ significantly.
  auto low_node = node->GetLowNode().get();
  assert(low_node);

  // Only known transpositions can differ.
  if (!low_node->IsTransposition()) return false;

  // LowNode is terminal when Node is not.
  if (low_node->IsTerminal()) return true;

  // Bounds differ (swap).
  auto [low_node_lower, low_node_upper] = low_node->GetBounds();
  auto [node_lower, node_upper] = node->GetBounds();
  if (low_node_lower != -node_upper || low_node_upper != -node_lower)
    return true;

  // WL differs significantly (flip).
  auto wl_diff = std::abs(low_node->GetWL() + node->GetWL());
  if (wl_diff >= wl_diff_limit) return true;

  // D differs significantly.
  auto d_diff = std::abs(low_node->GetD() - node->GetD());
  if (d_diff >= d_diff_limit) return true;

  // M differs significantly (increment).
  auto m_diff = std::abs(low_node->GetM() + 1 - node->GetM());
  if (m_diff >= m_diff_limit) return true;

  return false;
}

void SearchWorker::PickNodesToExtendTask(
    const BackupPath& path, int collision_limit, PositionHistory& history,
    std::vector<NodeToProcess>* receiver,
    TaskWorkspace* workspace) NO_THREAD_SAFETY_ANALYSIS {
  assert(path.size() == (size_t)history.GetLength() -
                            search_->played_history_.GetLength() + 1);

  // TODO: Bring back pre-cached nodes created outside locks in a way that works
  // with tasks.
  // TODO: pre-reserve visits_to_perform for expected depth and likely maximum
  // width. Maybe even do so outside of lock scope.
  auto& vtp_buffer = workspace->vtp_buffer;
  auto& visits_to_perform = workspace->visits_to_perform;
  visits_to_perform.clear();
  auto& vtp_last_filled = workspace->vtp_last_filled;
  vtp_last_filled.clear();
  auto& current_path = workspace->current_path;
  current_path.clear();
  auto& full_path = workspace->full_path;
  full_path = path;
  assert(full_path.size() > 0);
  auto [node, repetitions, moves_left] = full_path.back();
  // Sometimes receiver is reused, othertimes not, so only jump start if small.
  if (receiver->capacity() < 30) {
    receiver->reserve(receiver->size() + 30);
  }

  // This 1 is 'filled pre-emptively'.
  std::array<float, 256> current_util;

  // These 3 are 'filled on demand'.
  std::array<float, 256> current_score;
  std::array<int, 256> current_nstarted;
  auto& cur_iters = workspace->cur_iters;

  Node::Iterator best_edge;
  Node::Iterator second_best_edge;
  // Fetch the current best root node visits for possible smart pruning.
  const int64_t best_node_n = search_->current_best_edge_.GetN();

  int passed_off = 0;
  int completed_visits = 0;

  bool is_root_node = node == search_->root_node_;
  const float even_draw_score = search_->GetDrawScore(false);
  const float odd_draw_score = search_->GetDrawScore(true);
  const auto& root_move_filter = search_->root_move_filter_;
  auto m_evaluator = moves_left_support_ ? MEvaluator(params_) : MEvaluator();

  int max_limit = std::numeric_limits<int>::max();

  current_path.push_back(-1);
  while (current_path.size() > 0) {
    assert(full_path.size() >= path.size());
    // First prepare visits_to_perform.
    if (current_path.back() == -1) {
      // Need to do n visits, where n is either collision_limit, or comes from
      // visits_to_perform for the current path.
      int cur_limit = collision_limit;
      if (current_path.size() > 1) {
        cur_limit =
            (*visits_to_perform.back())[current_path[current_path.size() - 2]];
      }
      // First check if node is terminal or not-expanded.  If either than create
      // a collision of appropriate size and pop current_path.
      if (ShouldStopPickingHere(node, is_root_node, repetitions)) {
        if (is_root_node) {
          // Root node is special - since its not reached from anywhere else, so
          // it needs its own logic. Still need to create the collision to
          // ensure the outer gather loop gives up.
          if (node->TryStartScoreUpdate()) {
            cur_limit -= 1;
            minibatch_.push_back(
                NodeToProcess::Visit(full_path, search_->played_history_));
            completed_visits++;
          }
        }
        // Visits are created elsewhere, just need the collisions here.
        if (cur_limit > 0) {
          int max_count = 0;
          if (cur_limit == collision_limit && path.size() == 1 &&
              max_limit > cur_limit) {
            max_count = max_limit;
          }
          receiver->push_back(
              NodeToProcess::Collision(full_path, cur_limit, max_count));
          completed_visits += cur_limit;
        }
        history.Pop();
        full_path.pop_back();
        if (full_path.size() > 0) {
          std::tie(node, repetitions, moves_left) = full_path.back();
        } else {
          node = nullptr;
          repetitions = 0;
        }
        current_path.pop_back();
        continue;
      }
      if (is_root_node) {
        // Root node is again special - needs its n in flight updated separately
        // as its not handled on the path to it, since there isn't one.
        node->IncrementNInFlight(cur_limit);
      }

      // Create visits_to_perform new back entry for this level.
      if (vtp_buffer.size() > 0) {
        visits_to_perform.push_back(std::move(vtp_buffer.back()));
        vtp_buffer.pop_back();
      } else {
        visits_to_perform.push_back(std::make_unique<std::array<int, 256>>());
      }
      vtp_last_filled.push_back(-1);

      // Cache all constant UCT parameters.

      int max_needed = node->GetNumEdges();
      for (int i = 0; i < max_needed; i++) {
        current_util[i] = std::numeric_limits<float>::lowest();
      }
      // Root depth is 1 here, while for GetDrawScore() it's 0-based, that's why
      // the weirdness.
      const float draw_score =
          (full_path.size() % 2 == 0) ? odd_draw_score : even_draw_score;
      m_evaluator.SetParent(node);
      float visited_pol = 0.0f;
      for (Node* child : node->VisitedNodes()) {
        int index = child->Index();
        visited_pol += child->GetP();
        float q = child->GetQ(draw_score);
        current_util[index] = q + m_evaluator.GetMUtility(child, q);
      }
      const float fpu =
          GetFpu(params_, node, is_root_node, draw_score, visited_pol);
      for (int i = 0; i < max_needed; i++) {
        if (current_util[i] == std::numeric_limits<float>::lowest()) {
          current_util[i] = fpu + m_evaluator.GetDefaultMUtility();
        }
      }

      const float cpuct =
          ComputeCpuct(params_, node->GetTotalVisits(), is_root_node);
      const float puct_mult =
          cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
      int cache_filled_idx = -1;
      while (cur_limit > 0) {
        // Perform UCT for current node.
        float best = std::numeric_limits<float>::lowest();
        int best_idx = -1;
        float best_without_u = std::numeric_limits<float>::lowest();
        float second_best = std::numeric_limits<float>::lowest();
        bool can_exit = false;
        best_edge.Reset();
        for (int idx = 0; idx < max_needed; ++idx) {
          if (idx > cache_filled_idx) {
            if (idx == 0) {
              cur_iters[idx] = node->Edges();
            } else {
              cur_iters[idx] = cur_iters[idx - 1];
              ++cur_iters[idx];
            }
            current_nstarted[idx] = cur_iters[idx].GetNStarted();
          }
          int nstarted = current_nstarted[idx];
          const float util = current_util[idx];
          if (idx > cache_filled_idx) {
            current_score[idx] =
                cur_iters[idx].GetP() * puct_mult / (1 + nstarted) + util;
            cache_filled_idx++;
          }
          if (is_root_node) {
            // If there's no chance to catch up to the current best node with
            // remaining playouts, don't consider it.
            // best_move_node_ could have changed since best_node_n was
            // retrieved. To ensure we have at least one node to expand, always
            // include current best node.
            if (cur_iters[idx] != search_->current_best_edge_ &&
                latest_time_manager_hints_.GetEstimatedRemainingPlayouts() <
                    best_node_n - cur_iters[idx].GetN()) {
              continue;
            }
            // If root move filter exists, make sure move is in the list.
            if (!root_move_filter.empty() &&
                std::find(root_move_filter.begin(), root_move_filter.end(),
                          cur_iters[idx].GetMove()) == root_move_filter.end()) {
              continue;
            }
          }

          float score = current_score[idx];
          if (score > best) {
            second_best = best;
            second_best_edge = best_edge;
            best = score;
            best_idx = idx;
            best_without_u = util;
            best_edge = cur_iters[idx];
          } else if (score > second_best) {
            second_best = score;
            second_best_edge = cur_iters[idx];
          }
          if (can_exit) break;
          if (nstarted == 0) {
            // One more loop will get 2 unvisited nodes, which is sufficient to
            // ensure second best is correct. This relies upon the fact that
            // edges are sorted in policy decreasing order.
            can_exit = true;
          }
        }
        int new_visits = 0;
        if (second_best_edge) {
          int estimated_visits_to_change_best = std::numeric_limits<int>::max();
          if (best_without_u < second_best) {
            const auto n1 = current_nstarted[best_idx] + 1;
            estimated_visits_to_change_best = static_cast<int>(
                std::max(1.0f, std::min(cur_iters[best_idx].GetP() * puct_mult /
                                                (second_best - best_without_u) -
                                            n1 + 1,
                                        1e9f)));
          }
          second_best_edge.Reset();
          max_limit = std::min(max_limit, estimated_visits_to_change_best);
          new_visits = std::min(cur_limit, estimated_visits_to_change_best);
        } else {
          // No second best - only one edge, so everything goes in here.
          new_visits = cur_limit;
        }
        if (best_idx >= vtp_last_filled.back()) {
          auto* vtp_array = visits_to_perform.back().get()->data();
          std::fill(vtp_array + (vtp_last_filled.back() + 1),
                    vtp_array + best_idx + 1, 0);
        }
        (*visits_to_perform.back())[best_idx] += new_visits;
        cur_limit -= new_visits;

        Node* child_node = best_edge.GetOrSpawnNode(/* parent */ node);
        history.Append(best_edge.GetMove());
        auto [child_repetitions, child_moves_left] =
            GetRepetitions(full_path.size(), history.Last());
        full_path.push_back({child_node, child_repetitions, child_moves_left});
        if (child_node->TryStartScoreUpdate()) {
          current_nstarted[best_idx]++;
          new_visits -= 1;
          if (ShouldStopPickingHere(child_node, false, child_repetitions)) {
            // Reduce 1 for the visits_to_perform to ensure the collision
            // created doesn't include this visit.
            (*visits_to_perform.back())[best_idx] -= 1;
            receiver->push_back(NodeToProcess::Visit(full_path, history));
            completed_visits++;
          } else {
            child_node->IncrementNInFlight(new_visits);
            current_nstarted[best_idx] += new_visits;
          }
          current_score[best_idx] = cur_iters[best_idx].GetP() * puct_mult /
                                        (1 + current_nstarted[best_idx]) +
                                    current_util[best_idx];
        }
        if (best_idx > vtp_last_filled.back() &&
            (*visits_to_perform.back())[best_idx] > 0) {
          vtp_last_filled.back() = best_idx;
        }
        history.Pop();
        full_path.pop_back();
      }
      is_root_node = false;
      // Actively do any splits now rather than waiting for potentially long
      // tree walk to get there.
      for (int i = 0; i <= vtp_last_filled.back(); i++) {
        int child_limit = (*visits_to_perform.back())[i];
        if (task_workers_ > 0 &&
            child_limit > params_.GetMinimumWorkSizeForPicking() &&
            child_limit <
                ((collision_limit - passed_off - completed_visits) * 2 / 3) &&
            child_limit + passed_off + completed_visits <
                collision_limit -
                    params_.GetMinimumRemainingWorkSizeForPicking()) {
          Node* child_node = cur_iters[i].GetOrSpawnNode(/* parent */ node);
          history.Append(cur_iters[i].GetMove());
          auto [child_repetitions, child_moves_left] =
              GetRepetitions(full_path.size(), history.Last());
          full_path.push_back(
              {child_node, child_repetitions, child_moves_left});
          // Don't split if not expanded or terminal.
          if (!ShouldStopPickingHere(child_node, false, child_repetitions)) {
            bool passed = false;
            {
              // Multiple writers, so need mutex here.
              Mutex::Lock lock(picking_tasks_mutex_);
              // Ensure not to exceed size of reservation.
              if (picking_tasks_.size() < MAX_TASKS) {
                picking_tasks_.emplace_back(full_path, history, child_limit);
                task_count_.fetch_add(1, std::memory_order_acq_rel);
                task_added_.notify_all();
                passed = true;
                passed_off += child_limit;
              }
            }
            if (passed) {
              (*visits_to_perform.back())[i] = 0;
            }
          }
          history.Pop();
          full_path.pop_back();
        }
      }
      // Fall through to select the first child.
    }
    int min_idx = current_path.back();
    bool found_child = false;
    if (vtp_last_filled.back() > min_idx) {
      int idx = -1;
      for (auto& child : node->Edges()) {
        idx++;
        if (idx > min_idx && (*visits_to_perform.back())[idx] > 0) {
          current_path.back() = idx;
          current_path.push_back(-1);
          node = child.GetOrSpawnNode(/* parent */ node);
          history.Append(child.GetMove());
          std::tie(repetitions, moves_left) =
              GetRepetitions(full_path.size(), history.Last());
          full_path.push_back({node, repetitions, moves_left});
          found_child = true;
          break;
        }
        if (idx >= vtp_last_filled.back()) break;
      }
    }
    if (!found_child) {
      history.Pop();
      full_path.pop_back();
      if (full_path.size() > 0) {
        std::tie(node, repetitions, moves_left) = full_path.back();
      } else {
        node = nullptr;
        repetitions = 0;
      }
      current_path.pop_back();
      vtp_buffer.push_back(std::move(visits_to_perform.back()));
      visits_to_perform.pop_back();
      vtp_last_filled.pop_back();
    }
  }
}

void SearchWorker::ExtendNode(NodeToProcess& picked_node) {
  const auto path = picked_node.path;
  assert(!std::get<0>(path.back())->GetLowNode());

  const PositionHistory& history = picked_node.history;

  // We don't need the mutex because other threads will see that N=0 and
  // N-in-flight=1 and will not touch this node.
  const auto& board = history.Last().GetBoard();
  std::vector<Move> legal_moves = board.GenerateLegalMoves();

  // Check whether it's a draw/lose by position. Importantly, we must check
  // these before doing the by-rule checks below.
  auto node = picked_node.node;
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

    if (history.Last().GetRule50Ply() >= 100) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    }

    // Handle repetition draws as pseudo-terminals.
    if (picked_node.repetitions >= 2) {
      // Not a real terminal, set low node.
    }
    // Neither by-position or by-rule termination, but maybe it's a TB
    // position.
    else if (search_->syzygy_tb_ && !search_->root_is_in_dtz_ &&
             board.castlings().no_legal_castle() &&
             history.Last().GetRule50Ply() == 0 &&
             (board.ours() | board.theirs()).count() <=
                 search_->syzygy_tb_->max_cardinality()) {
      ProbeState state;
      const WDLScore wdl =
          search_->syzygy_tb_->probe_wdl(history.Last(), &state);
      // Only fail state means the WDL is wrong, probe_wdl may produce correct
      // result with a stat other than OK.
      if (state != FAIL) {
        // TB nodes don't have NN evaluation, assign M from parent node.
        float m = 0.0f;
        if (path.size() > 1) {
          auto parent = std::get<0>(path[path.size() - 2]);
          m = std::max(0.0f, parent->GetM() - 1.0f);
        }
        // If the colors seem backwards, check the checkmate check above.
        if (wdl == WDL_WIN) {
          node->MakeTerminal(GameResult::BLACK_WON, m, Terminal::Tablebase);
        } else if (wdl == WDL_LOSS) {
          node->MakeTerminal(GameResult::WHITE_WON, m, Terminal::Tablebase);
        } else {  // Cursed wins and blessed losses count as draws.
          node->MakeTerminal(GameResult::DRAW, m, Terminal::Tablebase);
        }
        search_->tb_hits_.fetch_add(1, std::memory_order_acq_rel);
        return;
      }
    }
  }

  // Check the transposition table first and NN cache second before asking for
  // NN evaluation.
  picked_node.hash = history.HashLast(params_.GetCacheHistoryLength() + 1);
  auto tt_iter = search_->tt_->find(picked_node.hash);
  // Transposition table entry might be expired.
  if (tt_iter != search_->tt_->end()) {
    picked_node.tt_low_node = tt_iter->second.lock();
  }
  if (picked_node.tt_low_node) {
    assert(!tt_iter->second.expired());
    picked_node.is_tt_hit = true;
  } else {
    picked_node.tt_low_node = std::make_shared<LowNode>(legal_moves);
    picked_node.nn_queried = true;
    picked_node.eval->p.resize(legal_moves.size());
    picked_node.is_cache_hit = computation_->AddInput(
                                   EvalPosition{
                                       .pos = history.GetPositions(),
                                       .legal_moves = legal_moves,
                                   },
                                   picked_node.eval->AsPtr()) ==
                               BackendComputation::FETCHED_IMMEDIATELY;
  }
}

// 4. Run NN computation.
// ~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::RunNNComputation() {
  if (computation_->UsedBatchSize() > 0) computation_->ComputeBlocking();
}

// 5. Retrieve NN computations (and terminal values) into nodes.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::FetchMinibatchResults() {
  // Populate NN/cached results, or terminal results, into nodes.
  for (auto& node_to_process : minibatch_) {
    FetchSingleNodeResult(&node_to_process);
  }
}

void SearchWorker::FetchSingleNodeResult(NodeToProcess* node_to_process) {
  if (!node_to_process->nn_queried) return;

  auto wdl_rescale = [&]() {
    if (params_.GetWDLRescaleRatio() != 1.0f ||
        (params_.GetWDLRescaleDiff() != 0.0f &&
         search_->contempt_mode_ != ContemptMode::NONE)) {
      // Check whether root moves are from the set perspective.
      bool root_stm = search_->contempt_mode_ == ContemptMode::WHITE;
      auto sign = (root_stm ^ node_to_process->history.IsBlackToMove())
                      ? 1.0f
                      : -1.0f;
      WDLRescale(node_to_process->eval->q, node_to_process->eval->d,
                 params_.GetWDLRescaleRatio(),
                 search_->contempt_mode_ == ContemptMode::NONE
                     ? 0
                     : params_.GetWDLRescaleDiff(),
                 sign, false, params_.GetWDLMaxS());
    }
  };
  wdl_rescale();
  node_to_process->tt_low_node->SetNNEval(node_to_process->eval.get());
  node_to_process->tt_low_node->SortEdges();

  // Add NN results to node.
  Node* node = node_to_process->node;
  // Add Dirichlet noise if enabled and at root.
  if (params_.GetNoiseEpsilon() && node == search_->root_node_) {
    ApplyDirichletNoise(node_to_process->tt_low_node.get(),
                        params_.GetNoiseEpsilon(), params_.GetNoiseAlpha());
    node_to_process->tt_low_node->SortEdges();
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
  search_->total_batches_ += 1;
}

bool SearchWorker::MaybeAdjustForTerminalOrTransposition(
    Node* n, const std::shared_ptr<LowNode>& nl, float& v, float& d, float& m,
    uint32_t& n_to_fix, float& v_delta, float& d_delta, float& m_delta,
    bool& update_parent_bounds) const {
  if (n->IsTerminal()) {
    v = n->GetWL();
    d = n->GetD();
    m = n->GetM();

    return true;
  }

  // Use information from transposition or a new terminal.
  if (nl->IsTransposition() || nl->IsTerminal() || n->GetN() < nl->GetN()) {
    // Adapt information from low node to node by flipping Q sign, bounds,
    // result and incrementing m.
    v = -nl->GetWL();
    d = nl->GetD();
    m = nl->GetM() + 1;
    // When starting at or going through a transposition/terminal, make sure to
    // use the information it has already acquired.
    n_to_fix = n->GetN();
    v_delta = v - n->GetWL();
    d_delta = d - n->GetD();
    m_delta = m - n->GetM();
    // Update bounds.
    if (params_.GetStickyEndgames()) {
      auto tt = nl->GetTerminalType();
      if (tt != Terminal::NonTerminal) {
        GameResult r;
        if (v == 1.0f) {
          r = GameResult::WHITE_WON;
        } else if (v == -1.0f) {
          r = GameResult::BLACK_WON;
        } else {
          r = GameResult::DRAW;
        }

        n->MakeTerminal(r, m, tt);
        update_parent_bounds = true;
      } else {
        auto [lower, upper] = nl->GetBounds();
        n->SetBounds(-upper, -lower);
      }
    }

    return true;
  }

  return false;
}

// Use information from terminal status or low node to update node and node's
// parent low node and so on until the root is reached. Low node may become a
// transposition and/or get more information even during this batch. Both low
// node and node may adjust bounds and become a terminal during this batch.
void SearchWorker::DoBackupUpdateSingleNode(
    const NodeToProcess& node_to_process) REQUIRES(search_->nodes_mutex_) {
  if (node_to_process.IsCollision()) {
    return;
  }

  auto path = node_to_process.path;

  if (node_to_process.nn_queried) {
    auto [tt_iter, is_tt_miss] = search_->tt_->try_emplace(
        node_to_process.hash, node_to_process.tt_low_node);
    if (is_tt_miss) {
      assert(!tt_iter->second.expired());
      node_to_process.node->SetLowNode(node_to_process.tt_low_node);
    } else {
      auto tt_low_node = tt_iter->second.lock();
      if (!tt_low_node) {
        tt_iter->second = node_to_process.tt_low_node;
        node_to_process.node->SetLowNode(node_to_process.tt_low_node);
      } else {
        assert(!tt_iter->second.expired());
        node_to_process.node->SetLowNode(tt_low_node);
      }
    }
  } else if (node_to_process.is_tt_hit) {
    node_to_process.node->SetLowNode(node_to_process.tt_low_node);
  }

  auto [n, nr, nm] = path.back();
  // For the first visit to a terminal, maybe update parent bounds too.
  auto update_parent_bounds =
      params_.GetStickyEndgames() && n->IsTerminal() && !n->GetN();
  const auto& nl = n->GetLowNode();
  float v = 0.0f;
  float d = 0.0f;
  float m = 0.0f;
  uint32_t n_to_fix = 0;
  float v_delta = 0.0f;
  float d_delta = 0.0f;
  float m_delta = 0.0f;

  // Update the low node at the start of the backup path first, but only visit
  // it the first time that backup sees it.
  if (nl && nl->GetN() == 0) {
    nl->FinalizeScoreUpdate(nl->GetWL(), nl->GetD(), nl->GetM(),
                            node_to_process.multivisit);
  }

  if (nr >= 2) {
    // Three-fold itself has to be handled as a terminal to produce relevant
    // results. Unlike two-folds that can keep updating their "real" values.
    n->SetRepetition();
    v = 0.0f;
    d = 1.0f;
    m = 1;
  } else if (!MaybeAdjustForTerminalOrTransposition(n, nl, v, d, m, n_to_fix,
                                                    v_delta, d_delta, m_delta,
                                                    update_parent_bounds)) {
    // If there is nothing better, use original NN values adjusted for node.
    v = -nl->GetWL();
    d = nl->GetD();
    m = nl->GetM() + 1;
  }

  // Backup V value up to a root. After 1 visit, V = Q.
  for (auto it = path.crbegin(); it != path.crend();
       /* ++it in the body */) {
    n->FinalizeScoreUpdate(v, d, m, node_to_process.multivisit);
    if (n_to_fix > 0 && !n->IsTerminal()) {
      n->AdjustForTerminal(v_delta, d_delta, m_delta, n_to_fix);
    }

    // Stop delta update on repetition "terminal" and propagate a draw above
    // repetitions valid on the current path.
    // Only do this after edge update to have good values if play goes here.
    if (nr == 1 && !n->IsTerminal()) {
      n->SetRepetition();
      v = 0.0f;
      d = 1.0f;
      m = nm + 1;
    }
    if (n->IsRepetition()) n_to_fix = 0;

    // Nothing left to do without ancestors to update.
    if (++it == path.crend()) break;
    auto [p, pr, pm] = *it;
    const auto& pl = p->GetLowNode();

    assert(!p->IsTerminal() ||
           (p->IsTerminal() && pl->IsTerminal() && p->GetWL() == -pl->GetWL() &&
            p->GetD() == pl->GetD()));
    // If parent low node is already a (new) terminal, then change propagated
    // values and stop terminal adjustment.
    if (pl->IsTerminal()) {
      v = pl->GetWL();
      d = pl->GetD();
      m = pl->GetM();
      n_to_fix = 0;
    }
    pl->FinalizeScoreUpdate(v, d, m, node_to_process.multivisit);
    if (n_to_fix > 0) {
      pl->AdjustForTerminal(v_delta, d_delta, m_delta, n_to_fix);
    }

    bool old_update_parent_bounds = update_parent_bounds;
    // Try setting parent bounds except the root or those already terminal.
    update_parent_bounds =
        update_parent_bounds && p != search_->root_node_ && !pl->IsTerminal() &&
        MaybeSetBounds(p, m, &n_to_fix, &v_delta, &d_delta, &m_delta);

    // Q will be flipped for opponent.
    v = -v;
    v_delta = -v_delta;
    m++;

    MaybeAdjustForTerminalOrTransposition(p, pl, v, d, m, n_to_fix, v_delta,
                                          d_delta, m_delta,
                                          update_parent_bounds);

    // Update the stats.
    // Best move.
    // If update_parent_bounds was set, we just adjusted bounds on the
    // previous loop or there was no previous loop, so if n is a terminal, it
    // just became that way and could be a candidate for changing the current
    // best edge. Otherwise a visit can only change best edge if its to an edge
    // that isn't already the best and the new n is equal or greater to the old
    // n.
    if (p == search_->root_node_ &&
        ((old_update_parent_bounds && n->IsTerminal()) ||
         (n != search_->current_best_edge_.node() &&
          search_->current_best_edge_.GetN() <= n->GetN()))) {
      search_->current_best_edge_ =
          search_->GetBestChildNoTemperature(search_->root_node_, 0);
    }

    n = p;
    nr = pr;
    nm = pm;
  }
  search_->total_playouts_ += node_to_process.multivisit;
  search_->cum_depth_ +=
      node_to_process.path.size() * node_to_process.multivisit;
  search_->max_depth_ =
      std::max(search_->max_depth_, (uint16_t)node_to_process.path.size());
}

bool SearchWorker::MaybeSetBounds(Node* p, float m, uint32_t* n_to_fix,
                                  float* v_delta, float* d_delta,
                                  float* m_delta) const {
  auto losing_m = 0.0f;
  auto prefer_tb = false;

  // Determine the maximum (lower, upper) bounds across all edges.
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
  const auto& pl = p->GetLowNode();
  if (lower == GameResult::BLACK_WON && upper == GameResult::WHITE_WON) {
    return false;
  } else if (lower == upper) {
    // Search can stop at the parent if the bounds can't change anymore, so make
    // it terminal preferring shorter wins and longer losses.
    *n_to_fix = p->GetN();
    assert(*n_to_fix > 0);
    pl->MakeTerminal(
        upper, (upper == GameResult::BLACK_WON ? std::max(losing_m, m) : m),
        prefer_tb ? Terminal::Tablebase : Terminal::EndOfGame);
    // v, d and m will be set in MaybeAdjustForTerminalOrTransposition.
    *v_delta = pl->GetWL() + p->GetWL();
    *d_delta = pl->GetD() - p->GetD();
    *m_delta = pl->GetM() + 1 - p->GetM();
    p->MakeTerminal(
        -upper,
        (upper == GameResult::BLACK_WON ? std::max(losing_m, m) : m) + 1.0f,
        prefer_tb ? Terminal::Tablebase : Terminal::EndOfGame);
  } else {
    pl->SetBounds(lower, upper);
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
  search_->MaybeOutputInfo(iteration_stats_);

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

}  // namespace dag_classic
}  // namespace lczero
