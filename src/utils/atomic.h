/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

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
#include <atomic>
#include <version>
#if __cpp_lib_atomic_wait < 201907L
#include <condition_variable>
#include "utils/mutex.h"
#endif

namespace lczero {

#if __cpp_lib_atomic_wait >= 201907L
template <typename T>
using WaitableAtomic = std::atomic<T>;
#else
template <typename T>
class WaitableAtomic : public std::atomic<T> {
  using Base = std::atomic<T>;

 public:
  using Base::atomic;
  using value_type = typename Base::value_type;

  void wait(value_type old, std::memory_order order =
                                std::memory_order_seq_cst) const noexcept {
    Mutex::Lock lock(mutex_);
    cv_.wait(lock.get_raw(), [this, old, order]() { return this->load(order) != old; });
  }

  void notify_one() noexcept {
    Mutex::Lock lock(mutex_);
    cv_.notify_one();
  }

  void notify_all() noexcept {
    Mutex::Lock lock(mutex_);
    cv_.notify_all();
  }

 private:
  mutable Mutex mutex_;
  mutable std::condition_variable cv_;
};
#endif

}  // namespace lczero
