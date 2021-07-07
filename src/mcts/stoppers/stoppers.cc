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

#include "mcts/stoppers/stoppers.h"

#include "mcts/node.h"
#include "neural/cache.h"

namespace lczero {

///////////////////////////
// ChainedSearchStopper
///////////////////////////

bool ChainedSearchStopper::ShouldStop(const IterationStats& stats,
                                      StoppersHints* hints) {
  for (const auto& x : stoppers_) {
    if (x->ShouldStop(stats, hints)) return true;
  }
  return false;
}

void ChainedSearchStopper::AddStopper(std::unique_ptr<SearchStopper> stopper) {
  if (stopper) stoppers_.push_back(std::move(stopper));
}

void ChainedSearchStopper::OnSearchDone(const IterationStats& stats) {
  for (const auto& x : stoppers_) x->OnSearchDone(stats);
}

///////////////////////////
// VisitsStopper
///////////////////////////

bool VisitsStopper::ShouldStop(const IterationStats& stats,
                               StoppersHints* hints) {
  if (populate_remaining_playouts_) {
    hints->UpdateEstimatedRemainingPlayouts(nodes_limit_ - stats.total_nodes);
  }
  if (stats.total_nodes >= nodes_limit_) {
    LOGFILE << "Stopped search: Reached visits limit: " << stats.total_nodes
            << ">=" << nodes_limit_;
    return true;
  }
  return false;
}

///////////////////////////
// PlayoutsStopper
///////////////////////////

bool PlayoutsStopper::ShouldStop(const IterationStats& stats,
                                 StoppersHints* hints) {
  if (populate_remaining_playouts_) {
    hints->UpdateEstimatedRemainingPlayouts(nodes_limit_ -
                                            stats.nodes_since_movestart);
  }
  if (stats.nodes_since_movestart >= nodes_limit_) {
    LOGFILE << "Stopped search: Reached playouts limit: "
            << stats.nodes_since_movestart << ">=" << nodes_limit_;
    return true;
  }
  return false;
}

///////////////////////////
// MemoryWatchingStopper
///////////////////////////

namespace {
const size_t kAvgNodeSize =
    sizeof(Node) + MemoryWatchingStopper::kAvgMovesPerPosition * sizeof(Edge);
const size_t kAvgCacheItemSize =
    NNCache::GetItemStructSize() + sizeof(CachedNNRequest) +
    sizeof(CachedNNRequest::IdxAndProb) *
        MemoryWatchingStopper::kAvgMovesPerPosition;
}  // namespace

MemoryWatchingStopper::MemoryWatchingStopper(int cache_size, int ram_limit_mb,
                                             bool populate_remaining_playouts)
    : VisitsStopper(
          (ram_limit_mb * 1000000LL - cache_size * kAvgCacheItemSize) /
              kAvgNodeSize,
          populate_remaining_playouts) {
  LOGFILE << "RAM limit " << ram_limit_mb << "MB. Cache takes "
          << cache_size * kAvgCacheItemSize / 1000000
          << "MB. Remaining memory is enough for " << GetVisitsLimit()
          << " nodes.";
}

///////////////////////////
// TimelimitStopper
///////////////////////////

TimeLimitStopper::TimeLimitStopper(int64_t time_limit_ms)
    : time_limit_ms_(time_limit_ms) {}

bool TimeLimitStopper::ShouldStop(const IterationStats& stats,
                                  StoppersHints* hints) {
  hints->UpdateEstimatedRemainingTimeMs(time_limit_ms_ -
                                        stats.time_since_movestart);
  if (stats.time_since_movestart >= time_limit_ms_) {
    LOGFILE << "Stopping search: Ran out of time.";
    return true;
  }
  return false;
}

int64_t TimeLimitStopper::GetTimeLimitMs() const { return time_limit_ms_; }

///////////////////////////
// DepthStopper
///////////////////////////
bool DepthStopper::ShouldStop(const IterationStats& stats, StoppersHints*) {
  if (stats.average_depth >= depth_) {
    LOGFILE << "Stopped search: Reached depth.";
    return true;
  }
  return false;
}

///////////////////////////
// KldGainStopper
///////////////////////////

KldGainStopper::KldGainStopper(float min_gain, int average_interval)
    : min_gain_(min_gain), average_interval_(average_interval) {}

bool KldGainStopper::ShouldStop(const IterationStats& stats, StoppersHints*) {
  Mutex::Lock lock(mutex_);
  const auto new_child_nodes = stats.total_nodes - 1.0;
  if (new_child_nodes < prev_child_nodes_ + average_interval_) return false;

  const auto new_visits = stats.edge_n;
  if (!prev_visits_.empty()) {
    double kldgain = 0.0;
    for (decltype(new_visits)::size_type i = 0; i < new_visits.size(); i++) {
      double o_p = prev_visits_[i] / prev_child_nodes_;
      double n_p = new_visits[i] / new_child_nodes;
      if (prev_visits_[i] != 0) kldgain += o_p * log(o_p / n_p);
    }
    if (kldgain / (new_child_nodes - prev_child_nodes_) < min_gain_) {
      LOGFILE << "Stopping search: KLDGain per node too small.";
      return true;
    }
  }
  prev_visits_ = new_visits;
  prev_child_nodes_ = new_child_nodes;
  return false;
}

///////////////////////////
// SmartPruningStopper
///////////////////////////

namespace {
const int kSmartPruningToleranceMs = 200;
const int kSmartPruningToleranceNodes = 300;
}  // namespace

SmartPruningStopper::SmartPruningStopper(float smart_pruning_factor,
                                         int64_t minimum_batches)
    : smart_pruning_factor_(smart_pruning_factor),
      minimum_batches_(minimum_batches) {}

bool SmartPruningStopper::ShouldStop(const IterationStats& stats,
                                     StoppersHints* hints) {
  if (smart_pruning_factor_ <= 0.0) return false;
  Mutex::Lock lock(mutex_);
  if (stats.edge_n.size() == 1) {
    LOGFILE << "Only one possible move. Moving immediately.";
    return true;
  }
  if (stats.edge_n.size() <= static_cast<size_t>(stats.num_losing_edges + 1)) {
    LOGFILE << "At most one non losing move, stopping search.";
    return true;
  }
  if (stats.win_found) {
    LOGFILE << "Terminal win found, stopping search.";
    return true;
  }
  if (stats.nodes_since_movestart > 0 && !first_eval_time_) {
    first_eval_time_ = stats.time_since_movestart;
    return false;
  }
  if (!first_eval_time_) return false;
  if (stats.edge_n.size() == 0) return false;
  if (stats.time_since_movestart <
      *first_eval_time_ + kSmartPruningToleranceMs) {
    return false;
  }

  const auto nodes = stats.nodes_since_movestart + kSmartPruningToleranceNodes;
  const auto time = stats.time_since_movestart - *first_eval_time_;
  // If nps is populated by someone who knows better, use it. Otherwise use the
  // value calculated here.
  const auto nps = hints->GetEstimatedNps().value_or(1000LL * nodes / time + 1);

  const double remaining_time_s = hints->GetEstimatedRemainingTimeMs() / 1000.0;
  const auto remaining_playouts =
      std::min(remaining_time_s * nps / smart_pruning_factor_,
               hints->GetEstimatedRemainingPlayouts() / smart_pruning_factor_);

  // May overflow if (nps/smart_pruning_factor) > 180 000 000, but that's not
  // very realistic.
  hints->UpdateEstimatedRemainingPlayouts(remaining_playouts);
  if (stats.batches_since_movestart < minimum_batches_) return false;
  if (remaining_playouts <= 0) {
    LOGFILE << remaining_playouts
            << " playouts remaining. SPF calculation no longer meaningful, so "
               "aborting.";
    return true;
  }

  int64_t largest_n = -1;
  int64_t largest_idx = -1;
  int64_t second_largest_n = -1;
  int64_t second_largest_idx = -1;
  int64_t third_largest_n = -1;
  int64_t third_largest_idx = -1;
  int64_t sum = 0;
  int64_t counter = 0;
  for (auto n : stats.edge_n) {
    sum += n;
    if (n > largest_n) {
      third_largest_idx = second_largest_idx;
      third_largest_n = second_largest_n;
      second_largest_idx = largest_idx;
      second_largest_n = largest_n;
      largest_idx = counter;
      largest_n = n;
    } else if (n > second_largest_n) {
      third_largest_idx = second_largest_idx;
      third_largest_n = second_largest_n;
      second_largest_idx = counter;
      second_largest_n = n;
    } else if (n > third_largest_n) {
      third_largest_idx = counter;
      third_largest_n = n;
    }
    counter++;
  }
  bool has_three = stats.edge_n.size() > 2;
  std::vector<double> inputs = {
      static_cast<double>(largest_n) / sum,
      static_cast<double>(second_largest_n) / sum,
      has_three ? static_cast<double>(third_largest_n) / sum : 0.0,
      stats.edge_s[largest_idx],
      stats.edge_s[second_largest_idx],
      has_three ? stats.edge_s[third_largest_idx] : 0.0,
      stats.edge_p[largest_idx],
      stats.edge_p[second_largest_idx],
      has_three ? stats.edge_p[third_largest_idx] : 0.0,
      static_cast<double>(sum) / (sum + remaining_playouts),
      static_cast<double>(largest_n - second_largest_n) / remaining_playouts
  };
  /*
  const double weights[11] = {
      2.43771887,  -7.1116147,  -0.664592385,
      0.929772913, -1.79833412, 0.690726399,
      4.65619, -0.0727774277, 0.94772023,
      1.50536919,
      1.41578805,
  };
  */
  const double weights[11][25] = {
      {
          -6.06307793,  -0.182800382, -4.10089064, -0.495022863, -3.30246592,
          -2.43335342,  -2.50909114,  1.0871,      0.290257603,  -0.186913356,
          -16.9425201,  -8.64100742,  2.58840871,  -1.89929736,  0.842219472,
          0.0308563933, -0.801170588, -4.56860161, 0.0818400756, -0.845116854,
          -0.776992202, -1.09660339,  -2.07050323, -1.45182121,  -2.34963489,
      },
      {
          1.35141909,  -0.384525418, 0.342251599,  1.4203018,    -0.33050704,
          3.04718256,  -2.89109945,  -3.7132957,   0.0586110391, 0.674045,
          1.58624184,  5.37736273,   2.09121943,   -9.15231133,  -2.92887211,
          0.30205518,  -2.87841606,  -1.60127139,  0.204678342,  1.23179889,
          -4.06269026, 1.28002799,   -0.500048339, 1.42918396,   2.16610169,
      },
      {
          4.25767469,  -0.990423858, 4.39364,     -1.22605193,  -0.00999091659,
          -1.54777169, 3.37644029,   7.23421,     0.3538903,    1.0684768,
          5.33190584,  -0.701068223, 0.231039479, -0.605326235, 1.79932821,
          0.141968817, 1.73205292,   0.536088228, 0.268859774,  -0.811413705,
          0.145614073, -1.88357091,  1.53367496,  -3.39719296,  0.0377489701,
      },
      {
          -1.3789916,    6.94240141,  -1.22067153,  -5.8878231,   -0.416531771,
          -5.16204882,   -1.36749744, -2.0166254,   -1.73073471,  7.70863342,
          -0.107319869,  -1.05301225, 6.53093815,   -6.07687187,  -4.21904659,
          -0.0388206616, 1.20693028,  -0.948398829, -0.551526248, -5.80119514,
          1.84903979,    4.67388391,  1.94113994,   3.29483175,   -4.5625658,
      },
      {
          0.515065372,  -5.2777,     -1.17304289,  3.72162771,  1.55392361,
          4.68625593,   1.04897296,  -0.260772467, 1.66570866,  -8.04365,
          -1.88631785,  0.963220298, 7.5145092,    5.44356203,  1.25961936,
          -0.141312465, -4.71373415, -1.69967973,  0.540095568, 5.79644156,
          -2.11453891,  -5.28303862, -2.56903458,  -4.36980057, 5.05003786,
      },
      {
          -1.53502464,  0.395879924, 1.51295793,  0.831418872,  -0.272190362,
          0.994953156,  0.683694422, 6.44323587,  0.0479699597, 0.211647227,
          2.38851714,   1.0250802,   7.77640963,  0.297567904,  0.856606066,
          0.0702350214, -1.18413627, 0.133906707, 0.0191279985, -0.756095231,
          2.3486402,    0.149580821, -2.08849812, -0.118195884, -0.188013166,
      },
      {
          -1.00074244,  1.72984982,   -5.44885159,  1.75785673,    5.78977489,
          0.200034901,  2.6406157,    -5.16312218,  -0.0213638842, 0.768088341,
          -4.68328857,  3.5714972,    -0.234971493, 1.63526595,    -4.2089,
          -0.332841307, 2.85234547,   -9.50640583,  -0.103609562,  -0.197868526,
          1.10565662,   -0.878099561, -2.06647086,  1.38575983,    -0.991603434,
      },
      {
          2.7415092,    -1.5147059,   2.8456738,   -2.17430353,  0.216492414,
          -0.248317346, 5.37043571,   -3.32112527, -0.195515573, 0.379042417,
          5.05633926,   -0.901594639, -7.62183428, 0.904861212,  -4.10003138,
          -0.115343541, 2.46335053,   1.12453187,  -0.135470986, 2.12259746,
          1.49189913,   -0.228291437, 2.54953194,  2.62160516,   1.08741319,
      },
      {
          -5.25058842,  0.611862481,  3.48081899,  0.410674304,  0.0626683459,
          1.04732883,   -8.11200142,  -1.48930109, -0.51209259,  1.5785799,
          5.06432152,   0.348118454,  0.841767311, 0.327296883,  -4.72197819,
          -0.252590418, 1.1250751,    -3.22206235, -0.234125778, 0.0286626238,
          2.04461646,   -0.674155414, 0.30435136,  -0.211393014, -0.265301615,
      },
      {
          1.93841779,    1.16293669,   -0.701742232, 1.65209222,    -4.0030694,
          1.53242624,    -1.16545653,  0.533577502,  -0.0547064021, -3.25374269,
          0.138708889,   0.826340556,  1.54670835,   0.513176501,   -1.47788525,
          -0.0392806269, -0.378505141, 1.26350558,   -0.0594539419, 1.08251357,
          1.6421994,     2.93875289,   1.68832803,   1.47821498,    1.29185963,
      },
      {
          -0.930339277,  -0.817088783, 0.013364139, -0.312276274, -1.59321225,
          0.569491506,   1.4381777,    0.432241887, 0.0149802649, 0.664834142,
          -0.234204412,  -2.74672437,  -1.61567461, 0.218759298,  1.14957905,
          -0.0820594802, -1.04941893,  0.825907826, -0.032416746, 0.0213468466,
          1.61395144,    0.343541294,  2.69469047,  -0.789898157, -0.122947469,
      },
  };
  const double biases[25] = {
      0.610038757,   0.748050928,  1.00923026,   -0.468266279,   -0.554005623,
      -0.361160308,  -0.774051189, 0.91595757,   -0.0280612037,  -0.707624674,
      0.663588226,   -0.109549232, -0.243538946, 0.647557914,    -0.779445,
      -0.0073985057, 0.386412174,  0.692225397,  -0.00958596822, -0.103063129,
      -0.194349915,  -0.743662953, -2.00849819,  -1.47079682,    -0.0908304676,
  };
  const double weights2[25] = {
      -0.9619537,   4.10666084,  -1.32566595,  -1.15534163,   3.39625144,
      -3.48435616,  -3.02020431, -0.93546313,  -0.017583264,  -5.6194,
      -5.69930649,  -2.4404645,  -0.437718451, 7.11238766,    3.42543292,
      0.0018308846, 1.44105971,  -1.59067047,  -0.0109598218, -2.96400547,
      2.23122811,   3.3804028,   1.51862979,   2.20880079,    -1.6139071,
  };

  double signals[25] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  for (int j = 0; j < 11; j++) {
    for (int i = 0; i < 25; i++) {
      signals[i] += inputs[j] * weights[j][i];
    }
  }
  for (int i = 0; i < 25; i++) {
    signals[i] += biases[i];
    // swish activation
    double es = exp(-signals[i]);
    signals[i] = signals[i] / (1+es);
  }

    double score = 0.0;
  for (int i=0; i < 25; i++) {
    score += signals[i] * weights2[i];
  }
  // Final bias.
  score += 0.0978407413;
  // Threshold at sigmoid(x)~=0.95
  if (score > 2.9) {
    LOGFILE << remaining_playouts << " playouts remaining. Best move has "
            << largest_n << " visits, second best -- " << second_largest_n
            << ". Current 'score' " << score
            << ", so stopping the search after "
            << stats.batches_since_movestart << " batches.";
    return true;
  }
  /*
  if (remaining_playouts < (largest_n - second_largest_n)) {
    LOGFILE << remaining_playouts << " playouts remaining. Best move has "
            << largest_n << " visits, second best -- " << second_largest_n
            << ". Difference is " << (largest_n - second_largest_n)
            << ", so stopping the search after "
            << stats.batches_since_movestart << " batches.";

    return true;
  }
  */

  return false;
}

}  // namespace lczero
