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
*/

#pragma once

#include <atomic>
#include <shared_mutex>

namespace lczero {

// Implementation of reader-preferenced shared mutex. Based on fair shared
// mutex.
class rp_shared_mutex {
 public:
  void lock() {
    while (true) {
      mutex_.lock();
      if (waiting_readers_ == 0) return;
      mutex_.unlock();
    }
  }
  void unlock() { mutex_.unlock(); }
  void lock_shared() {
    ++waiting_readers_;
    mutex_.lock_shared();
  }
  void unlock_shared() {
    --waiting_readers_;
    mutex_.unlock_shared();
  }

 private:
  std::shared_mutex mutex_;
  std::atomic<int> waiting_readers_ = 0;
};

}  // namespace lczero