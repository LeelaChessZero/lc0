#pragma once

#include <span>
#include <vector>

namespace lczero {

struct TempParams {
  double temperature;
  int temp_decay_moves;
  int temp_cutoff_move;
  double temp_endgame;
  double value_cutoff;
  double visit_offset;
};

// Returns effective temperature tau for the given fullmove number (1-based).
// Applies cutoff and linear decay. Result is clamped to [0, +inf).
double EffectiveTau(const TempParams& p, int fullmove_number);

class Random;  // Forward declaration from utils/random.h.

// Samples an index from base_weights using temperature tau. Applies value
// cutoff and visit offset. winprob may be empty to skip value cutoff.
// Returns fallback_index if all weights are filtered to zero.
int SampleWithTemperature(std::span<const double> base_weights,
                          std::span<const double> winprob,
                          const TempParams& p,
                          double tau,
                          Random& rng,
                          int fallback_index);

}  // namespace lczero

