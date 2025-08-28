#include "search/common/temperature.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "utils/random.h"

namespace lczero {

float EffectiveTau(const TempParams& p, int ply) {
  const int moves = ply / 2;

  float tau = p.temperature;
  if (p.temp_cutoff_move > 0 && (moves + 1) >= p.temp_cutoff_move) {
    tau = p.temp_endgame;
  } else if (tau > 0.0 && p.temp_decay_moves > 0) {
    if (moves >= p.temp_decay_moves) {
      tau = 0.0;
    } else {
      tau *= static_cast<double>(p.temp_decay_moves - moves) /
             static_cast<double>(p.temp_decay_moves);
    }
    // don't allow temp to decay below endgame temp
    if (tau < p.temp_endgame) tau = p.temp_endgame;
  }
  if (tau < 0.0) tau = 0.0;
  return tau;
}

int SampleWithTemperature(std::span<const double> base_weights,
                          std::span<const double> winprob,
                          const TempParams& p,
                          float tau,
                          Random& rng,
                          int fallback_index) {
  const size_t n = base_weights.size();
  std::vector<double> weights(n);
  double max_winprob = -std::numeric_limits<double>::infinity();
  if (!winprob.empty()) {
    for (double w : winprob) {
      if (w > max_winprob) max_winprob = w;
    }
  }
  double sum = 0.0;
  const double inv_tau = tau > 0 ? 1.0 / static_cast<double>(tau) : 0.0;
  for (size_t i = 0; i < n; ++i) {
    double w = base_weights[i];
    if (!winprob.empty() && max_winprob - winprob[i] > p.value_cutoff) {
      w = 0.0;
    }
    if (p.visit_offset != 0.0) {
      w = std::max(w - p.visit_offset, 0.0);
    }
    if (w > 0.0 && tau > 0.0) {
      w = std::pow(w, inv_tau);
    } else {
      w = 0.0;
    }
    weights[i] = w;
    sum += w;
  }
  if (sum <= 0.0) return fallback_index;
  double toss = rng.GetDouble(sum);
  double cumulative = 0.0;
  for (size_t i = 0; i < n; ++i) {
    cumulative += weights[i];
    if (toss < cumulative) return static_cast<int>(i);
  }
  return fallback_index;
}

}  // namespace lczero

