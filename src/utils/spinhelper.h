/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2023 The LCZero Authors

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

#pragma once

#include <cstddef>
#include <random>

#include "utils/mutex.h"

namespace lczero {

class SpinHelper {
 public:
  virtual ~SpinHelper() = default;
  virtual void Backoff() {}
  virtual void Wait() {}
};

class ExponentialBackoffSpinHelper : public SpinHelper {
 public:
  ExponentialBackoffSpinHelper()
      : backoff_iters_(kMinBackoffIters), spin_to_sleep_iters_(0) {}

  virtual void Backoff() {
    thread_local std::uniform_int_distribution<size_t> distribution;
    thread_local std::minstd_rand generator(std::random_device{}());
    const size_t spin_count = distribution(
        generator, decltype(distribution)::param_type{0, backoff_iters_});

    for (size_t i = 0; i < spin_count; i++) SpinloopPause();

    backoff_iters_ = std::min(2 * backoff_iters_, kMaxBackoffIters);
    spin_to_sleep_iters_ = 0;
  }

  // Spin to sleep
  virtual void Wait() {
    if (spin_to_sleep_iters_ < kMaxSpinToSleepIters) {
      spin_to_sleep_iters_++;
      SpinloopPause();
    } else {
      spin_to_sleep_iters_ = 0;
      std::this_thread::sleep_for(kSleepDuration);
    }
  }

 private:
  static constexpr size_t kMaxSpinToSleepIters = 0x10000;
  static constexpr size_t kMinBackoffIters = 0x20;
  static constexpr size_t kMaxBackoffIters = 0x400;
  static constexpr std::chrono::microseconds kSleepDuration{1000};

  size_t backoff_iters_;
  size_t spin_to_sleep_iters_;
};

}  // namespace lczero
