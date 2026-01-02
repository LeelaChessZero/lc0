/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2025 The LCZero Authors

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

#include "search/contempt/params.h"

#include <sstream>

namespace lczero {
namespace contempt {
namespace {

// START: ADDED FOR DYNAMIC HYBRID RATIO
// Helper function to parse the manual schedule string.
std::vector<std::pair<int, float>> ParseHybridRatioSchedule(
    const std::string& schedule_str) {
  std::vector<std::pair<int, float>> schedule;
  std::stringstream ss(schedule_str);
  std::string item;

  int nodes;
  float ratio;
  char colon;

  while (ss >> item) {
    std::stringstream item_ss(item);
    if (item_ss >> nodes >> colon >> ratio && colon == ':') {
      schedule.push_back({nodes, ratio});
    }
  }

  // Ensure the schedule is sorted by node count.
  std::sort(schedule.begin(), schedule.end());
  // Remove duplicate node counts, keeping the first one.
  schedule.erase(std::unique(schedule.begin(), schedule.end(),
                             [](const auto& a, const auto& b) {
                               return a.first == b.first;
                             }),
                 schedule.end());

  return schedule;
}

HybridRatioMode EncodeHybridRatioMode(const std::string& mode_str) {
    if (mode_str == "static") return HybridRatioMode::STATIC;
    if (mode_str == "manual_schedule") return HybridRatioMode::MANUAL_SCHEDULE;
    if (mode_str == "linear") return HybridRatioMode::LINEAR;
    if (mode_str == "logarithmic") return HybridRatioMode::LOGARITHMIC;
    if (mode_str == "power") return HybridRatioMode::POWER;
    if (mode_str == "root") return HybridRatioMode::ROOT;
    if (mode_str == "sigmoid") return HybridRatioMode::SIGMOID;
    if (mode_str == "exponential") return HybridRatioMode::EXPONENTIAL;
    if (mode_str == "step_decay") return HybridRatioMode::STEP_DECAY;
    if (mode_str == "inverse_sigmoid") return HybridRatioMode::INVERSE_SIGMOID;
    if (mode_str == "steps") return HybridRatioMode::STEPS;
    if (mode_str == "plateau") return HybridRatioMode::PLATEAU;
    if (mode_str == "gaussian_peak") return HybridRatioMode::GAUSSIAN_PEAK;
    if (mode_str == "double_peak") return HybridRatioMode::DOUBLE_PEAK;
    if (mode_str == "sawtooth_wave") return HybridRatioMode::SAWTOOTH_WAVE;
    if (mode_str == "oscillating") return HybridRatioMode::OSCILLATING;
    if (mode_str == "heartbeat") return HybridRatioMode::HEARTBEAT;
    if (mode_str == "multi_timescale") return HybridRatioMode::MULTI_TIMESCALE;
    if (mode_str == "thermal_annealing") return HybridRatioMode::THERMAL_ANNEALING;
    if (mode_str == "asymptotic_approach") return HybridRatioMode::ASYMPTOTIC_APPROACH;
    if (mode_str == "chaotic") return HybridRatioMode::CHAOTIC;
    return HybridRatioMode::STATIC; // Default
}
// END: ADDED FOR DYNAMIC HYBRID RATIO

}  // namespace

const OptionId SearchParams::kScLimitId{
    "search-contempt-node-limit", "ScLimit",
    "UCT until this number of nodes"
    "thompson sampling beyond this limit."};
const OptionId SearchParams::kContemptModeTBId{
    "contempt-mode-tb", "ContemptModeTB",
    "Choose asymmetric tablebase probing method. If a position has greator or "
    "eqaul number of piecese, probing uses only tablebase result on winning "
    "positions. Setting it to 6 uses only winning information if position has "
    "6 or more pieces. If a position has less pieces, tablebase is used to "
    "avoid losing moves too. Setting it to 0 disables assymetric probe."
    };
const OptionId SearchParams::kHybridSamplingRatioId{
    "hybrid-sampling-ratio", "HybridSamplingRatio",
    "The ratio of Thompson Sampling to use in hybrid search-contempt mode. "
    "1.0 is pure TS, 0.0 is pure PUCT."};
// START: ADDED FOR DYNAMIC HYBRID RATIO
const OptionId SearchParams::kHybridRatioModeId{
    "hybrid-ratio-mode", "HybridRatioMode",
    "Selects the function or mode for dynamically adjusting the hybrid "
    "sampling ratio."};
const OptionId SearchParams::kHybridRatioScheduleId{
    "hybrid-ratio-schedule", "HybridRatioSchedule",
    "A space-separated string of 'nodes:ratio' pairs for manual schedule "
    "mode."};
const OptionId SearchParams::kHybridMinRatioId{
    "hybrid-min-ratio", "HybridMinRatio",
    "The minimum ratio for dynamic functions after ScLimit is reached."};
const OptionId SearchParams::kHybridMaxRatioId{
    "hybrid-max-ratio", "HybridMaxRatio",
    "The maximum ratio for dynamic functions."};
const OptionId SearchParams::kHybridScalingFactorId{
    "hybrid-scaling-factor", "HybridScalingFactor",
    "The number of nodes over which a function transitions from min to max "
    "ratio."};
const OptionId SearchParams::kHybridShapeParam1Id{
    "hybrid-shape-param1", "HybridShapeParam1",
    "First generic shape parameter for complex functions (e.g., exponent, "
    "peak center, period)."};
const OptionId SearchParams::kHybridShapeParam2Id{
    "hybrid-shape-param2", "HybridShapeParam2",
    "Second generic shape parameter for complex functions (e.g., peak width)."};
// END: ADDED FOR DYNAMIC HYBRID RATIO

void SearchParams::Populate(OptionsParser* options) {
  classic::SearchParams::Populate(options);
  options->Add<IntOption>(kScLimitId, 1, 1000000000) = 1000000000;
  options->Add<IntOption>(kContemptModeTBId, 0, 9) = 6;
  options->Add<FloatOption>(kHybridSamplingRatioId, 0.0f, 1.0f) = 0.8f;

  // START: ADDED FOR DYNAMIC HYBRID RATIO
  std::vector<std::string> hybrid_modes = {
      "static", "manual_schedule", "linear", "logarithmic", "power", "root",
      "sigmoid", "exponential", "step_decay", "inverse_sigmoid", "steps", "plateau",
      "gaussian_peak", "double_peak", "sawtooth_wave", "oscillating", "heartbeat",
      "multi_timescale", "thermal_annealing", "asymptotic_approach", "chaotic"};
  options->Add<ChoiceOption>(kHybridRatioModeId, hybrid_modes) = "static";
  options->Add<StringOption>(kHybridRatioScheduleId) = "1000:0.5 8000:0.9";
  options->Add<FloatOption>(kHybridMinRatioId, 0.0f, 1.0f) = 0.2f;
  options->Add<FloatOption>(kHybridMaxRatioId, 0.0f, 1.0f) = 0.9f;
  options->Add<IntOption>(kHybridScalingFactorId, 1, 100000000) = 10000;
  options->Add<FloatOption>(kHybridShapeParam1Id, -10000.0f, 10000.0f) = 0.5f;
  options->Add<FloatOption>(kHybridShapeParam2Id, -10000.0f, 10000.0f) = 0.1f;
  // END: ADDED FOR DYNAMIC HYBRID RATIO
}

SearchParams::SearchParams(const OptionsDict& options)
    : classic::SearchParams(options),
      kContemptModeTB(options.Get<int>(kContemptModeTBId)),
      // START: ADDED FOR DYNAMIC HYBRID RATIO
      kHybridRatioMode(EncodeHybridRatioMode(options.Get<std::string>(kHybridRatioModeId))),
      kHybridRatioSchedule(ParseHybridRatioSchedule(options.Get<std::string>(kHybridRatioScheduleId)))
      // END: ADDED FOR DYNAMIC HYBRID RATIO
      {}

// START: ADDED FOR DYNAMIC HYBRID RATIO
float SearchParams::GetDynamicHybridRatio(int node_count) const {
    const int sc_limit = GetScLimit();

    // Core Rule: Ratio is 0.0 before the search contempt limit is reached.
    if (node_count < sc_limit) {
        return 0.0f;
    }

    const auto mode = kHybridRatioMode;

    if (mode == HybridRatioMode::MANUAL_SCHEDULE) {
        if (kHybridRatioSchedule.empty() || node_count <= kHybridRatioSchedule.front().first) {
            return kHybridRatioSchedule.empty() ? 0.0f : kHybridRatioSchedule.front().second;
        }
        if (node_count >= kHybridRatioSchedule.back().first) {
            return kHybridRatioSchedule.back().second;
        }
        for (size_t i = 0; i < kHybridRatioSchedule.size() - 1; ++i) {
            const auto& p1 = kHybridRatioSchedule[i];
            const auto& p2 = kHybridRatioSchedule[i + 1];
            if (node_count >= p1.first && node_count < p2.first) {
                int node_range = p2.first - p1.first;
                float ratio_range = p2.second - p1.second;
                if (node_range == 0) return p1.second;
                float progress = static_cast<float>(node_count - p1.first) / node_range;
                return std::clamp(p1.second + ratio_range * progress, 0.0f, 1.0f);
            }
        }
        return kHybridRatioSchedule.back().second;
    }

    const float min_r = GetHybridMinRatio();
    const float max_r = GetHybridMaxRatio();
    const float scale = static_cast<float>(GetHybridScalingFactor());
    const float p1 = GetHybridShapeParam1();
    const float p2 = GetHybridShapeParam2();

    // Progress is a value from 0.0 to 1.0+ representing how far we are through the scaling phase.
    const float progress = (scale > 0) ? (static_cast<float>(node_count - sc_limit) / scale) : 1.0f;
    const float clamped_progress = std::min(1.0f, progress);

    float ratio = min_r; // Default value

    switch (mode) {
        case HybridRatioMode::STATIC:
            return GetHybridSamplingRatio();
        case HybridRatioMode::LINEAR:
            ratio = min_r + (max_r - min_r) * clamped_progress;
            break;
        case HybridRatioMode::LOGARITHMIC:
            ratio = min_r + (max_r - min_r) * (std::log(clamped_progress * (2.71828f - 1.0f) + 1.0f));
            break;
        case HybridRatioMode::POWER:
            ratio = min_r + (max_r - min_r) * std::pow(clamped_progress, p1); // p1 is exponent
            break;
        case HybridRatioMode::ROOT:
            ratio = min_r + (max_r - min_r) * std::sqrt(clamped_progress);
            break;
        case HybridRatioMode::SIGMOID:
            ratio = min_r + (max_r - min_r) / (1.0f + std::exp(-12.0f * (clamped_progress - 0.5f)));
            break;
        case HybridRatioMode::EXPONENTIAL:
            ratio = min_r + (max_r - min_r) * (std::exp(clamped_progress) - 1.0f) / (2.71828f - 1.0f);
            break;
        case HybridRatioMode::STEP_DECAY:
            if (p1 > 0) { // p1 is number of steps
                ratio = max_r - (max_r - min_r) * std::floor(clamped_progress * p1) / p1;
            }
            break;
        case HybridRatioMode::INVERSE_SIGMOID:
            ratio = min_r + (max_r - min_r) * (1.0f - (1.0f / (1.0f + std::exp(-12.0f * (clamped_progress - 0.5f)))));
            break;
        case HybridRatioMode::STEPS:
            if (p1 > 0) { // p1 is number of steps
                ratio = min_r + (max_r - min_r) * std::floor(clamped_progress * p1) / p1;
            }
            break;
        case HybridRatioMode::GAUSSIAN_PEAK: { // p1=center [0,1], p2=width_sq
            float exponent = -std::pow(clamped_progress - p1, 2) / (2.0f * std::max(0.001f, p2));
            ratio = min_r + (max_r - min_r) * std::exp(exponent);
            break;
        }
        case HybridRatioMode::DOUBLE_PEAK: { // p1=center1, p2=center2
            float width_sq = 0.02f; // Hardcoded small width
            float exp1 = -std::pow(clamped_progress - p1, 2) / (2.0f * width_sq);
            float exp2 = -std::pow(clamped_progress - p2, 2) / (2.0f * width_sq);
            ratio = min_r + (max_r - min_r) * std::max(std::exp(exp1), std::exp(exp2));
            break;
        }
        case HybridRatioMode::SAWTOOTH_WAVE:
            if (p1 > 0) { // p1 is period in nodes
                ratio = min_r + (max_r - min_r) * (std::fmod(static_cast<float>(node_count - sc_limit), p1) / p1);
            }
            break;
        case HybridRatioMode::OSCILLATING:
            if (p1 > 0) { // p1 is period in nodes
                const float pi = 3.1415926535f;
                ratio = min_r + (max_r - min_r) * (0.5f + 0.5f * std::sin(2.0f * pi * (node_count - sc_limit) / p1));
            }
            break;
        case HybridRatioMode::THERMAL_ANNEALING:
            ratio = (max_r * (1.0f - clamped_progress)) * Random::Get().GetFloat(1.0f) + (min_r + (max_r - min_r) * clamped_progress);
            break;
        case HybridRatioMode::ASYMPTOTIC_APPROACH:
            ratio = max_r - (max_r - min_r) * std::exp(-5.0f * clamped_progress);
            break;
        case HybridRatioMode::CHAOTIC: { // p1 is 'r' param for logistic map
            chaotic_state_ = p1 * chaotic_state_ * (1.0f - chaotic_state_);
            ratio = min_r + (max_r - min_r) * chaotic_state_;
            break;
        }
        case HybridRatioMode::PLATEAU: // p1=mid_ratio, p2=width [0,1]
             if (clamped_progress < (0.5f - p2/2.0f)) { // Ramp up
                 float sub_progress = clamped_progress / std::max(0.001f, 0.5f - p2/2.0f);
                 ratio = min_r + (p1 - min_r) * sub_progress;
             } else if (clamped_progress > (0.5f + p2/2.0f)) { // Ramp up again
                 float sub_progress = (clamped_progress - (0.5f + p2/2.0f)) / std::max(0.001f, 0.5f - p2/2.0f);
                 ratio = p1 + (max_r - p1) * sub_progress;
             } else { // Plateau
                 ratio = p1;
             }
             break;
        // Simplified versions for complex multi-wave functions
        case HybridRatioMode::HEARTBEAT:
        case HybridRatioMode::MULTI_TIMESCALE:
            if (p1 > 0 && p2 > 0) { // p1=slow_period, p2=fast_period
                const float pi = 3.1415926535f;
                float slow_wave = 0.6f * (0.5f + 0.5f * std::sin(2.0f * pi * (node_count - sc_limit) / p1));
                float fast_wave = 0.4f * (0.5f + 0.5f * std::sin(2.0f * pi * (node_count - sc_limit) / p2));
                ratio = min_r + (max_r - min_r) * (slow_wave + fast_wave);
            }
            break;
        case HybridRatioMode::MANUAL_SCHEDULE: // Already handled above.
             break;
    }

    return std::clamp(ratio, 0.0f, 1.0f);
}
// END: ADDED FOR DYNAMIC HYBRID RATIO

}  // namespace contempt
}  // namespace lczero
