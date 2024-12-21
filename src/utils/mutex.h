/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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
#include <mutex>
#include <shared_mutex>
#include <thread>

#if defined(__x86_64__)
#include <emmintrin.h>
#endif

#include "utils/cppattributes.h"

namespace lczero {

// Implementation of reader-preferenced shared mutex. Based on fair shared
// mutex.
class CAPABILITY("mutex") RpSharedMutex {
 public:
  RpSharedMutex() : waiting_readers_(0) {}

  void lock() ACQUIRE() {
    while (true) {
      mutex_.lock();
      if (waiting_readers_ == 0) return;
      mutex_.unlock();
    }
  }
  void unlock() RELEASE() { mutex_.unlock(); }
  void lock_shared() ACQUIRE_SHARED() {
    ++waiting_readers_;
    mutex_.lock_shared();
  }
  void unlock_shared() RELEASE_SHARED() {
    --waiting_readers_;
    mutex_.unlock_shared();
  }

 private:
  std::shared_timed_mutex mutex_;
  std::atomic<int> waiting_readers_;
};

// std::mutex wrapper for clang thread safety annotation.
class CAPABILITY("mutex") Mutex {
 public:
  // std::unique_lock<std::mutex> wrapper.
  class SCOPED_CAPABILITY Lock {
   public:
    Lock(Mutex& m) ACQUIRE(m) : lock_(m.get_raw()) {}
    ~Lock() RELEASE() {}
    std::unique_lock<std::mutex>& get_raw() { return lock_; }

   private:
    std::unique_lock<std::mutex> lock_;
  };

  void lock() ACQUIRE() { mutex_.lock(); }
  void unlock() RELEASE() { mutex_.unlock(); }
  std::mutex& get_raw() { return mutex_; }

 private:
  std::mutex mutex_;
};

// std::shared_mutex wrapper for clang thread safety annotation.
class CAPABILITY("mutex") SharedMutex {
 public:
  // std::unique_lock<std::shared_mutex> wrapper.
  class SCOPED_CAPABILITY Lock {
   public:
    Lock(SharedMutex& m) ACQUIRE(m) : lock_(m.get_raw()) {}
    ~Lock() RELEASE() {}

   private:
    std::unique_lock<std::shared_timed_mutex> lock_;
  };

  // std::shared_lock<std::shared_mutex> wrapper.
  class SCOPED_CAPABILITY SharedLock {
   public:
    SharedLock(SharedMutex& m) ACQUIRE_SHARED(m) : lock_(m.get_raw()) {}
    ~SharedLock() RELEASE() {}

   private:
    std::shared_lock<std::shared_timed_mutex> lock_;
  };

  void lock() ACQUIRE() { mutex_.lock(); }
  void unlock() RELEASE() { mutex_.unlock(); }
  void lock_shared() ACQUIRE_SHARED() { mutex_.lock_shared(); }
  void unlock_shared() RELEASE_SHARED() { mutex_.unlock_shared(); }

  std::shared_timed_mutex& get_raw() { return mutex_; }

 private:
  std::shared_timed_mutex mutex_;
};

static inline void SpinloopPause() {
#if defined(__x86_64__) || defined(_M_X64)
  _mm_pause();
#elif defined(_MSC_VER)
  __asm {}
#else
  asm volatile("");
#endif
}

// A very simple spin lock.
class CAPABILITY("mutex") SpinMutex {
 public:
  // std::unique_lock<SpinMutex> wrapper.
  class SCOPED_CAPABILITY Lock {
   public:
    Lock(SpinMutex& m) ACQUIRE(m) : lock_(m) {}
    ~Lock() RELEASE() {}

   private:
    std::unique_lock<SpinMutex> lock_;
  };

  void lock() ACQUIRE() {
    int spins = 0;
    while (true) {
      int val = 0;
      if (mutex_.compare_exchange_weak(val, 1, std::memory_order_acq_rel)) {
        break;
      }
      ++spins;
      // Help avoid complete resource starvation by yielding occasionally if
      // needed.
      if (spins % 512 == 0) {
        std::this_thread::yield();
      } else {
        SpinloopPause();
      }
    }
  }
  void unlock() RELEASE() { mutex_.store(0, std::memory_order_release); }

 private:
  std::atomic<int> mutex_{0};
};

}  // namespace lczero
