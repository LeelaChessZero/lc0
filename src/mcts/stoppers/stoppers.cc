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
                                         int64_t minimum_batches,
                                         float hyperpruning_threshold)
    : smart_pruning_factor_(smart_pruning_factor),
      minimum_batches_(minimum_batches),
      hyperpruning_threshold_(hyperpruning_threshold) {}

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
      static_cast<double>(largest_n - second_largest_n) / remaining_playouts /
          50.0  // scale down by 50.0 to match training.
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
  const double weights1[11][32] = {
      {0.959496319,  -0.200361222, 0.0361186936, 0.101258203,  0.939749062,
       0.0813682228, -7.44220829,  -0.786709487, -0.301591754, 0.13703534,
       -0.131513312, 0.980388939,  -0.461311132, -7.24477291,  1.55345237,
       2.14331317,   1.85829389,   1.40066087,   -1.76916146,  1.30440307,
       0.727509916,  0.799603879,  -7.22882938,  0.0828282535, 0.0984897614,
       1.47036564,   1.51395333,   -2.60902691,  0.251851767,  -7.65943289,
       -0.217784479, 0.0460387021},
      {-0.325465292, -1.48680139, 1.92838323,  0.360813,     1.45642543,
       -0.681456864, 7.21311855,  4.31657791,  -2.60067487,  0.302994579,
       -1.33671093,  2.64087415,  -2.61183286, 6.37365723,   1.55757678,
       2.72606158,   1.35563707,  1.24719548,  2.75983334,   -3.50453973,
       -2.05332923,  -2.14322114, 6.99406242,  -0.622010052, -0.712418735,
       2.3732326,    -2.87012863, 2.71977448,  -1.13126743,  7.50994539,
       -2.17320013,  -1.48751152},
      {-10.8382549,  -0.159101665, -0.275865585, 0.317198962,   1.95949137,
       -0.217712402, -0.103142202, 4.55007172,   -0.0541055277, 0.13385576,
       -0.277354062, 2.05295753,   -2.85527873,  -0.606725276,  3.15373588,
       9.64178085,   7.71455717,   -4.77500725,  -3.33135319,   0.686133444,
       -1.24314058,  0.177445725,  -0.184529752, -0.261571735,  -0.421592474,
       4.00340748,   0.46183756,   -0.436045527, -0.244921118,  -0.911237419,
       -0.396288902, -0.394193172},
      {2.33772922,  0.539127469,   -5.00460434, -4.85850525, -3.66198468,
       3.35606694,  -2.30915308,   -4.17203045, 0.87431854,  -5.18401814,
       0.502635539, -4.27711344,   -3.35282493, -2.52187777, -1.19332492,
       0.683673501, -0.0934339836, 1.51589763,  -5.73675728, 1.58580589,
       -0.44111,    1.0949353,     -2.66292071, 4.01667833,  3.40630651,
       -1.49631,    1.39726388,    -3.81838751, 1.99080586,  -2.76345,
       0.391678691, 0.494881451},
      {-3.52370882,  -0.458030432, 5.83134413,  5.03396368,  4.27918768,
       -3.45053554,  1.71464384,   1.84869182,  -0.418943,   5.33600903,
       -0.535461783, 5.00863,      0.358111262, 2.21942496,  1.49740827,
       -1.30639446,  -3.48442936,  -2.96589971, 5.43436956,  -0.837876618,
       -1.33312964,  -0.708607852, 2.33498812,  -4.16263199, -3.42712069,
       2.11560106,   -1.45981264,  4.12166,     -2.15600204, 2.28069,
       -0.494606018, -0.493449837},
      {-5.51623154,  0.011276341,  -0.637442172, -0.101890251,  -0.54728961,
       0.197273552,  0.553416729,  2.3234632,    -0.612969577,  -0.0641233325,
       0.014068827,  -0.599977851, 3.71839452,   -0.0358366109, -0.776318491,
       0.476919621,  3.10043049,   1.07971025,   -0.488095,     -1.41476619,
       1.92082441,   -0.70102042,  0.315968156,  0.158993542,   0.117068775,
       -0.478682,    -0.783444762, -0.327849269, 0.267693549,   0.257347971,
       0.0239149891, -0.046123337},
      {-2.44937634, 0.516538501,   -0.513147414, -0.719693422, -0.904538035,
       1.00578189,  1.1726445,     -8.41474152,  0.0680269748, -0.600502908,
       0.620176077, -1.24128139,   0.681793571,  1.26806307,   -1.82649267,
       -3.62641931, -5.13347387,   0.385843545,  1.05277658,   -0.42225641,
       0.595016539, -0.0801136717, 1.19571257,   0.973557651,  0.984661818,
       -1.46921027, -0.211841896,  0.248366043,  0.871194839,  1.32469344,
       0.727015376, 0.54992044},
      {-2.12807322,   0.970477819, -2.0750339,   0.117914088,  -0.876665354,
       -0.0865560547, 0.065248847, 1.0858922,    3.41856146,   0.142525464,
       0.676738143,   -1.89071739, 2.53378057,   -0.344626784, -0.149806246,
       0.751895964,   1.49152982,  -7.74222422,  -9.04197311,  -4.25159168,
       -2.32188773,   -2.25591588, 0.109815978,  -0.107275598, -0.145813495,
       -0.764023,     -6.55981302, -0.761685371, 0.213758618,  -0.685446918,
       1.87808883,    0.722902954},
      {-7.54958296, 0.191655427, 1.00158608,   -0.122847512, -0.361567289,
       0.316485643, -1.77561438, -5.33078051,  2.18737078,   -0.254693836,
       0.102986261, 0.135513291, 2.72050309,   0.278844982,  -0.244478866,
       -7.06895685, -8.05303383, -5.8277235,   6.13454,      -2.61714482,
       -0.85935986, -1.6271795,  -0.837259948, 0.191730529,  0.253290385,
       -1.35159075, -3.70659661, 0.716581285,  0.0537874177, -0.196909487,
       1.18477285,  0.03124246},
      {0.649886966,  0.70451349,    -0.293664247,  -0.0632225052,
       -0.283867925, 0.180404082,   -0.0402852558, -0.961946487,
       1.40745294,   -0.0267186034, 0.495011508,   -0.79628849,
       0.505537629,  0.0305919889,  -15.0285368,   -2.69490266,
       -6.6354146,   0.0227334667,  0.249729723,   1.24715352,
       0.745559156,  1.19489789,    -0.0424022563, 0.134242564,
       0.409254789,  -0.784424424,  0.893733561,   -0.000482922653,
       0.546428859,  -0.0232191607, 0.967106402,   0.691070259},
      {5.68358755,  14.0945139,  -15.0952635, -20.7434196, -19.5149784,
       5.84074879,  -23.9754677, -7.35103798, 4.04963,     -18.2428799,
       18.5906219,  -3.11196756, 19.9981956,  -17.1849422, -0.089176774,
       -13.358325,  0.359434217, 3.02988863,  -4.60201693, 15.4555197,
       11.3682623,  10.8814468,  -5.0217576,  4.389328,    2.53764677,
       -20.9209805, 7.44664,     -13.9941025, 2.6867094,   -3.07894254,
       11.318058,   12.7382412},
  };
  const double bias1[32] = {
      -3.93663597,   -0.28197673,  0.0590449236, 0.0437861644,  -0.411463439,
      -0.0787838474, 0.393157721,  0.864125967,  -0.919328392,  0.0222348254,
      -0.141026691,  -0.455886096, -0.151770592, 0.227843359,   -0.122896984,
      -0.600187898,  0.135570407,  0.866005421,  0.377800226,   -0.176506832,
      0.133756354,   -0.205977261, 0.222575858,  -0.0647711679, -0.0434189215,
      -0.749673784,  0.248785242,  0.0801287293, -0.138229609,  0.337892979,
      -0.392687082,  -0.217386276,
  };
  const double weights2[32][8] = {
      {0.553179741, -0.314984947, 0.648046494, 0.558752894, 0.592382073,
       0.617994666, -0.495128721, 0.0301883556},
      {0.385381699, 0.00791060645, 0.221497893, 0.236908138, 0.385463417,
       0.41537407, -0.0378613956, 0.034550108},
      {-0.336613268, -0.0314969458, -0.478708923, -0.28423062, -0.336965263,
       -0.474046677, 0.106092758, 0.577778},
      {-0.38170743, 1.19350886, -0.212474123, -0.335364252, -0.408732802,
       -0.462698251, 1.02529418, 0.871622145},
      {-0.472656876, 0.0754528791, -0.403562665, -0.389348745, -0.518908,
       -0.401327282, 0.27927503, -0.887821555},
      {-0.0147374421, -0.956473172, 0.0543364026, 0.0786347091, -0.119792826,
       -0.0708518699, -0.634182036, -0.254788339},
      {-0.201665699, -0.0274145529, -0.390019238, -0.28499338, -0.100797139,
       -0.341581076, -0.0846389756, 0.847939074},
      {-0.331786036, -0.0115159834, -0.342037439, -0.176234171, -0.358725905,
       -0.284281045, 0.020463191, -0.573110938},
      {-0.0110085653, -0.0453302301, 0.360013813, 0.229313135, 0.197296083,
       0.0937069356, -0.114921428, -0.0031996097},
      {-0.362195849, 1.31778967, -0.300993234, -0.204013422, -0.499248773,
       -0.559638381, 1.4736321, 1.22531867},
      {0.305451602, -0.0346279927, 0.364094883, 0.460803866, 0.354112893,
       0.290866584, 0.00738357566, -0.00661338307},
      {-0.421791166, 0.140418455, -0.355434388, -0.343348116, -0.181024238,
       -0.3062087, -0.0162883811, -0.83280164},
      {0.204587802, -0.0940343738, 0.310932457, 0.284076303, 0.262684852,
       0.303144425, 0.0175617207, -0.0443323106},
      {-0.414259374, 0.213412434, -0.0256229769, -0.255622506, -0.33652848,
       -0.564966202, 0.0736781359, 3.50074},
      {-0.362788677, 0.188043773, -0.363649279, -0.415182412, -0.239662513,
       -0.374332368, 0.0653872788, -0.858794272},
      {-0.341098815, -0.0788506046, -0.32542488, -0.324348211, -0.270455539,
       -0.19576405, 0.0812519416, -0.820747},
      {-0.231351122, 0.17632331, -0.167071283, -0.0505920611, -0.0860037729,
       -0.174276546, 0.209748149, -0.816267133},
      {0.0363944024, -0.216927171, 0.0224344842, 0.116619296, 0.0216974784,
       0.175372586, -0.0432362929, 0.0122813666},
      {-0.166717783, 0.0840484351, -0.19153598, -0.192063764, -0.250433296,
       -0.0384865, 0.105929643, 0.301072359},
      {0.0739231259, 0.0848278925, 0.212136745, 0.193377763, 0.290781349,
       0.168933123, 0.0949708819, 0.0113722021},
      {-0.0470617115, -0.078144744, 0.208388448, 0.170723483, 0.201095104,
       0.249597535, -0.0844759122, 0.059237022},
      {0.229836792, 0.0394926406, 0.225859642, 0.339126557, 0.204854816,
       0.21706818, 0.00863849092, -0.0292248167},
      {-0.292070419, 0.13285251, -0.148773193, -0.339339554, -0.249582812,
       -0.392116368, 0.073857747, 1.8469578},
      {-0.00405977573, -1.08276641, 0.0828017071, 0.0673422739, -0.0341768675,
       0.0926507562, -0.822007895, -0.257526636},
      {0.138274208, -0.874049783, 0.11540895, 0.0698194, 0.0361123942,
       0.14470157, -0.691975236, -0.19851394},
      {-0.332559228, 0.037446823, -0.332067251, -0.375630468, -0.315933853,
       -0.435402334, 0.0428339839, -1.023862},
      {0.039565891, 0.0710440204, 0.129738107, 0.31766057, 0.190239087,
       0.141066134, -0.023714019, 0.0820676535},
      {-0.242190421, 0.23007746, -0.0474243574, -0.129947081, -0.0227203146,
       -0.0593872145, 0.192015648, 2.75663137},
      {0.000553598162, -0.672199667, -0.000892296433, 0.0759906322,
       0.0801599249, 0.145153344, -0.361538589, -0.254101485},
      {-0.19826445, 0.137999296, -0.426721781, -0.409588933, -0.413542837,
       -0.196031764, 0.123828039, 1.16183972},
      {0.301880807, -0.0798251778, 0.0510343313, 0.385989159, 0.256416976,
       0.333559215, -0.163727283, 0.00335605512},
      {0.435788602, 0.0454938039, 0.541451752, 0.31199494, 0.458534747,
       0.29681161, 0.0281655826, -0.0259828195},
  };
  const double bias2[8] = {
      0.066152215,  0.0984139517, 0.0798741356, -0.0270796772,
      0.0801076218, 0.0144859692, 0.0374633595, -0.544778109,
  };
  const double weights3[8] = {
      0.633739829, -0.72698915, 0.581135333, 0.543061554,
      0.482124507, 0.605796278, -0.744041,   -0.391707569,
  };
  const double bias3 = 1.11058724;
 

  double signals1[32] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  for (int j = 0; j < 11; j++) {
    for (int i = 0; i < 32; i++) {
      signals1[i] += inputs[j] * weights1[j][i];
    }
  }
  for (int i = 0; i < 32; i++) {
    signals1[i] += bias1[i];
    // swish activation
    double es = exp(-signals1[i]);
    signals1[i] = signals1[i] / (1 + es);
  }
  double signals2[8] = {
      0, 0, 0, 0, 0, 0, 0, 0,
  };
  for (int j = 0; j < 32; j++) {
    for (int i = 0; i < 8; i++) {
      signals2[i] += signals1[j] * weights2[j][i];
    }
  }
  for (int i = 0; i < 8; i++) {
    signals2[i] += bias2[i];
    // swish activation
    double es = exp(-signals2[i]);
    signals2[i] = signals2[i] / (1 + es);
  }

    double score = 0.0;
  for (int i=0; i < 8; i++) {
    score += signals2[i] * weights3[i];
  }
  // Final bias.
  score += bias3;
  // Threshold at sigmoid(x)~=0.95
  if (score > hyperpruning_threshold_) {
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
