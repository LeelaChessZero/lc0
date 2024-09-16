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
// MateStopper
///////////////////////////
bool MateStopper::ShouldStop(const IterationStats& stats, StoppersHints*) {
  if (stats.mate_depth <= mate_) {
    LOGFILE << "Stopped search: Found mate.";
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
  if (stats.edge_n.size() <= static_cast<size_t>(stats.num_losing_edges) +
                                 (stats.may_resign ? 0 : 1)) {
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
      {1.14794672,   -18.7545223,  -54.701107,  -0.857393682, 2.22063804,
       1.41744506,   -0.168271929, -37.9434853, -6.5917573,   -0.196891233,
       -0.183447286, 1.18721974,   0.50470078,  -0.186302841, -10.1688433,
       -0.217738494, 0.545400083,  -0.12189769, -0.197455242, 6.96107721,
       -0.253735304, 0.913883686,  4.47287369,  -0.180350602, 0.390401423,
       0.619411647,  7.14688158,   14.506093,   -0.124961033, -0.187755674,
       -32.8493156,  0.0333353095},
      {-4.17231941, 20.9989719,   40.2162323,   -3.23016739, -11.5007095,
       8.00982571,  -1.13254023,  32.696228,    6.93844795,  -1.08238375,
       -1.15892255, 6.58062029,   9.12384319,   -1.14040351, 15.0316067,
       -1.14612961, 2.14412808,   0.0199943408, 0.214195356, 9.91371632,
       -3.43588543, -0.140590429, -38.0423508,  -1.17274535, -0.944854259,
       1.75361693,  16.6230927,   15.4630861,   -2.13981652, -1.10414183,
       36.0965576,  2.1376195},
      {-3.80231762, -3.98014569,  13.4425564,   -16.1300087,   12.1433172,
       12.308012,   -0.960034251, 4.89492369,   23.1318665,    -1.1412214,
       -1.19066584, -3.60247946,  8.38334465,   -0.92477,      0.857000232,
       -1.06077731, 6.27726746,   0.0643184409, -0.0696802512, 10.0054598,
       -1.33781028, -0.315327346, -13.6982937,  -1.19515014,   -15.731575,
       3.53038883,  19.7527485,   20.6819859,   15.0534878,    -0.911642,
       2.02976871,  -0.877297103},
      {1.96197093,   -10.8853235, -0.00108498358, -5.32229805, 0.187376872,
       -0.868063629, 0.580085039, -3.82284212,    -8.26275158, 0.526297,
       0.557347655,  8.12638092,  -8.70513,       0.591397,    -22.9811611,
       0.559322119,  0.755741894, 8.89232635,     13.0921164,  9.32366,
       -19.3808098,  -65.388031,  -7.3784318,     0.576256216, 4.08513403,
       1.05391192,   5.70341921,  -16.42873,      -17.4570065, 0.608961,
       -7.11861134,  -71.4393616},
      {-1.9665401,  10.4441843,   6.13849,      1.55319989,   -1.40417349,
       -7.23497438, -0.600571811, -0.016229365, 15.6543484,   -0.549010873,
       -0.60284692, -2.52269626,  -8.66439819,  -0.612738729, 22.4905357,
       -0.58283329, -0.310023904, -9.12976074,  -13.2333841,  -2.49764776,
       1.39013696,  65.2328491,   4.39535189,   -0.585322,    -14.429678,
       0.476696104, -6.10210037,  19.9501534,   -5.19867611,  -0.637529492,
       6.70166063,  71.3247299},
      {-0.830949068, -0.755551219,    -4.67142344,   4.27626324,
       0.742158592,  8.42797661,      -0.0179483183, 3.11823487,
       -7.39438438,  -0.000344585802, -0.0216585826, -6.27365875,
       15.9986172,   -0.033793509,    0.36613363,    -0.018080188,
       -0.588848114, 0.166820481,     0.0952805579,  -6.28102541,
       0.486391962,  0.0935866535,    2.29276228,    -0.0300404206,
       -1.88487279,  -1.50344539,     -0.741579175,  -1.94602382,
       22.0320301,   -0.0310326423,   0.437449187,   -0.0228702631},
      {-1.0424825,  9.86747837,  14.4238577,   -0.191184789, 4.60945654,
       -9.61235428, 0.553594351, -6.29970884,  1.71399391,   0.536565959,
       0.563613594, -16.4644146, -17.4810963,  0.583341,     -1.48188126,
       0.579100788, -1.89514387, 0.872561336,  0.926604748,  -25.0565262,
       0.737563848, -2.23899698, -0.668889284, 0.555412769,  -4.33003855,
       -3.5625515,  3.26483464,  -5.86625099,  0.511280835,  0.557881236,
       1.49456346,  -3.09631467},
      {-3.71249843, -14.1469555,  -5.9517417,   2.91010404,   3.82482862,
       0.703685403, 0.315005064,  1.28607571,   -26.922884,   0.303180844,
       0.344241798, -6.17189,     2.83062959,   0.341350466,  -2.18133235,
       0.345133781, -0.788883567, -0.460065067, -0.740370035, -10.513566,
       0.595693707, 1.488608,     1.40180516,   0.308567256,  12.1891251,
       -2.09197974, -9.90362549,  -5.2117691,   0.644503534,  0.371775568,
       -1.3879602,  0.566029429},
      {-2.18435574,  -1.80221045,  -5.45820284,  1.35242891,    3.38447595,
       -2.33929515,  0.0752798617, 0.85525,      -7.05506802,   0.0699524283,
       0.0463806726, -7.43445683,  1.0945214,    0.0795358196,  -2.72859263,
       0.113413759,  -0.821054041, -0.191370651, -0.0123159094, -4.54033899,
       -0.648362637, -0.272547662, 1.81351972,   0.0393826626,  0.755485475,
       -0.826440334, -10.497508,   -7.64971,     -27.8333,      0.0946150199,
       -6.7868619,   0.150200531},
      {2.09459019,   -1.12392914,   -0.321626693,  0.208355829,  2.45741725,
       -5.21187353,  0.422003418,   -0.377503842,  -1.50020993,  0.395757347,
       0.456261486,  0.878304124,   -18.5883617,   0.417135954,  -0.905581534,
       0.406738847,  -9.85480499,   0.0583452284,  0.0651155412, -0.706687748,
       0.840656817,  -0.0322164968, 1.49885547,    0.47356981,   -0.922010899,
       -33.3531036,  -13.1400461,   -0.0318911783, -1.99560833,  0.391148567,
       -0.956759214, -0.0388168693},
      {14.0123243,      -8.24026203,     -11.4510727,     13.7516479,
       17.2947865,      -1.83809459e-06, 20.6855927,      -7.85870171,
       -10.0761204,     20.187767,       20.1874523,      -1.23597272e-06,
       1.09026551,      21.7189102,      -2.85411763,     22.083374,
       -0.000574219215, 19.1775494,      6.80864334,      3.1883235,
       -19.9523792,     -1.31615758,     -0.00439507328,  19.7592716,
       5.08848763,      -9.23366414e-08, -8.97179063e-07, -8.03814411,
       1.25822318,      22.7298355,      -15.5380497,     0.0939495936},
  };
  const double bias1[32] = {
      0.188088357,  1.95959413,    0.951971829,   1.60259151,    -2.76201582,
      -0.695393384, 0.00830274634, 1.69649732,    1.24886608,    0.0493829772,
      0.0264020655, 0.770188749,   0.30489254,    0.0120162,     0.939658,
      0.0434048623, 0.320552796,   -0.0698675886, -0.0603897534, -3.86618352,
      -0.091933623, 0.124293476,   -1.4125464,    0.028554013,   -10.4530888,
      1.27244556,   -8.52405739,   -10.5191803,   1.40138197,    0.0135735404,
      1.46249473,   0.219407782,
  };
  const double weights2[32][8] = {
      {0.323643744, 0.0264703576, 0.113134362, 0.0176272132, 0.034482643,
       0.152398914, 0.272842556, 0.114890948},
      {-0.624264538, -0.304207, 0.111469232, 0.015259847, -0.145929307,
       -0.0693953, -0.317150325, -0.024003718},
      {-0.484453022, -0.202094823, 0.320823103, 0.0976474807, -0.274225354,
       -0.228105053, -0.726867437, -0.146989569},
      {0.332394689, 0.204788759, 0.0793576837, -0.0400548317, 0.0775731802,
       0.0643017292, -0.316502661, 0.0110901138},
      {-0.0756028295, 0.0382072814, -0.780292273, -0.0958085731, 0.0668280572,
       -0.0249619801, 0.0522523858, 0.0251151342},
      {-0.534490347, -0.0145690357, 0.0391817279, 0.0167695899, -0.0342005454,
       -0.372303903, -0.170018479, -0.0443147309},
      {0.581007719, 0.363609105, 0.00296750735, -0.117618166, 0.299568623,
       0.47298035, -0.265098, 0.187104225},
      {-0.555026472, -0.0473037064, -0.425234765, 0.0686397478, -0.059758883,
       -0.165882051, 3.24551797, -0.0732744634},
      {-0.48677966, -0.0523429178, 0.171731323, 0.0372744277, -0.55329442,
       -0.0547965392, -0.513099372, -0.0199267566},
      {0.761119783, 0.351243317, 0.0500827655, -0.109190747, 0.305673093,
       0.552956283, -0.350052625, 0.182537854},
      {0.651831269, 0.349204123, 0.29998523, -0.120387711, 0.254943192,
       0.499345541, -0.24690862, 0.177332938},
      {-0.34357205, 0.0896424577, 0.101172969, -0.0331054218, 0.195373178,
       0.0537907518, 0.0245637428, 0.0526613221},
      {-0.367809325, -0.0467476435, 0.00478006573, 0.0504103638, -0.0831363201,
       -0.224594191, -1.03473091, -0.0744619295},
      {0.568769276, 0.375880182, 0.102753505, -0.128814504, 0.291973293,
       0.478698522, -0.270076394, 0.173869967},
      {-0.63916254, -0.209040418, 0.148190469, 0.0456272364, -0.0472220927,
       -0.559422374, -0.617413104, -0.0231404863},
      {0.649638593, 0.369685173, -0.00356045761, -0.097173579, 0.310828626,
       0.461438537, -0.367232591, 0.173710376},
      {-0.488330752, -0.136938021, 0.0657872111, -0.0191804413, -0.0748739317,
       -0.38426, -0.0870714635, -0.0880486667},
      {0.0267369039, -0.0217911135, -5.78261089, -0.494335532, 0.0190053843,
       0.0326826572, 0.0282533523, 0.0233958401},
      {0.0295429882, -0.0272119921, -9.60105419, -0.25314644, 0.00515329279,
       0.0279750749, 0.0480551161, 0.0140405865},
      {0.34812966, 0.249345317, 0.0535503328, -0.047476083, 0.0613064133,
       0.132468298, -0.868636191, 0.0925955921},
      {0.0161099229, -1.93343139, 0.0345531777, 0.00644724537, -0.128486097,
       0.00994394161, -0.029758295, -0.000477639318},
      {-0.333838433, -0.320477515, 0.0699868649, 0.0212814379, -0.321987063,
       -0.410934418, -4.42212629, -0.272632599},
      {0.38253051, 0.0701850578, -0.715123177, -0.00926116, 0.195061818,
       0.328199416, 0.679586649, 0.186826661},
      {0.688976467, 0.320925206, 0.317501575, -0.0878804326, 0.2684744,
       0.527997792, -0.245331049, 0.186982021},
      {0.13693054, -0.00667890208, -3.28809166, -1.87065172, 0.0155851869,
       0.0653001517, -0.175871015, 0.0624766387},
      {-0.142124549, -0.197539926, 0.162091032, 0.0494221039, -0.154659361,
       -0.156245381, -0.0278115291, -0.139396936},
      {0.124906868, -0.384284288, -0.450943649, -0.0840286, 0.100903749,
       0.482227921, 0.0748822168, 0.211496368},
      {-1.08778095, 0.00155697856, 0.0353347063, 0.00616296, -0.116954789,
       -0.0508296899, -0.104214363, 0.00254039629},
      {-0.731760561, -0.355601847, 0.257891685, 0.0296318103, -0.16163519,
       -0.0176449213, -0.88179332, -0.0107684359},
      {0.496182621, 0.432816505, -0.0996761397, -0.152270183, 0.299861401,
       0.405812383, -0.344903052, 0.16751419},
      {-0.59856379, -0.187403783, -0.0314681791, 0.0375879817, -0.151514277,
       -0.443638444, -0.509432137, -0.110989593},
      {-0.330157638, -0.248311013, 0.105183326, 0.0208205618, -0.264580429,
       -0.399213433, -1.84077156, -0.229355246},
  };
  const double bias2[8] = {
      -0.0116158137, -0.0367815606, 0.390878737, -0.0672987,
      0.203110144,   0.207609698,   -2.04896736, 0.1362928,
  };
  const double weights3[8] = {
      0.592073202, 0.932121933, -0.212991744, -1.4610076,
      1.07185733,  0.734642327, -0.60594213,  1.57529938,
  };
  const double bias3 = 1.17222774;
 

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
    LOGFILE << std::fixed << remaining_playouts
            << " playouts remaining. Best move has " << largest_n
            << " visits, second best -- " << second_largest_n
            << ". Difference is " << (largest_n - second_largest_n)
            << ", so stopping the search after "
            << stats.batches_since_movestart << " batches.";

    return true;
  }
  */

  return false;
}

}  // namespace lczero
