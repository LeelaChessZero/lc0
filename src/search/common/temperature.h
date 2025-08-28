#pragma once

#include <span>
#include <vector>

namespace lczero {

struct TemperatureParams {
  float temperature;
  int temp_decay_moves;
  int temp_cutoff_move;
  float temp_endgame;
  int temp_decay_delay_moves;
  float value_cutoff;
  float visit_offset;
};

// Returns effective temperature tau for the given game ply
// The fullmove number is computed internally as (ply / 2) + 1.
// Applies cutoff and linear decay. Result clamped to [0, +inf).
float EffectiveTau(const TemperatureParams& p, int ply);

class Random;  // Forward declaration from utils/random.h.

// Samples an index from base_weights using temperature tau.
// Applies value cutoff and visit offset
// winprob may be empty to skip value cutoff.
// Returns fallback_index if all weights are filtered to zero.
int SampleWithTemperature(std::span<const double> base_weights,
                          std::span<const double> winprob,
                          const TemperatureParams& p,
                          float tau,
                          Random& rng,
                          int fallback_index);

}  // namespace lczero

