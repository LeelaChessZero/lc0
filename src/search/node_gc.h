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
#include <thread>
#include "utils/mutex.h"

#if __cpp_lib_atomic_wait < 201907L
#define NO_STD_ATOMIC_WAIT 1
#include <condition_variable>
#endif

namespace lczero {

// Implement thread local queues. It tracks GC thread to allow faster removal in
// the thread.
template<typename Node, size_t kCapacity>
class ReleaseNodesWork {
public:
  ReleaseNodesWork(bool gc_thread = false);
  ~ReleaseNodesWork();
  bool IsWorker() const;

  // A limited vector like interface to operate on the container.
  void emplace_back(std::unique_ptr<Node>&& node);
  bool empty() const;

  // Swap is used to transfer queue into a new stack variable. The stack
  // variable will flush the queue in the desctructor.
  void swap(ReleaseNodesWork &other);
private:
  // Flush the local queue to the shared queue.
  void Submit();

  // No locks required because only one thread can access this object.
  std::vector<std::unique_ptr<Node>> released_nodes_;
  bool is_gc_thread_;
};

template<typename Node, size_t kCapacity>
class NodeGarbageCollector {
  NodeGarbageCollector();
  ~NodeGarbageCollector();
public:
  enum State {
    Running,
    GoToSleep,
    Sleeping,
    Exit,
  };

  // Access to the singleton which is only created on the demand.
  static NodeGarbageCollector& Instance() {
    static NodeGarbageCollector singleton;
    return singleton;
  }
  // Delays node destruction until GC thread activates.
  template<typename UniquePtr>
  void AddToGcQueue(UniquePtr& node);

  // Allow search to control when garbage collection runs.
  void Start();
  void Stop();
  State Wait() const;
  void Abort();

  // Moves thread local GC queue to the shared queue. This avoid case where a
  // thread frees only a few branches which will be stuck in the thread local
  // queue. A few big branches can have a major memory impact. If thread exits,
  // there is no need to call this.
  void NotifyThreadGoingSleep();

private:
  // Helper to transition between states safely
  bool SetState(State& old, State desired);
  bool IsActive() const;
  bool ShouldQueue(std::unique_ptr<Node>& node) const;
  // The collection thread implementation.
  void GCThread();
  // Thread local collection queue. Local queues flush to the shared queue
  // in batches to avoid lock contention.
  static ReleaseNodesWork<Node, kCapacity>& LocalWork(bool gc_thread = false) {
    static thread_local ReleaseNodesWork<Node, kCapacity> shared{gc_thread};
    return shared;
  }

  std::atomic<State> state_ = {Sleeping};
#ifdef NO_STD_ATOMIC_WAIT
  // Fallback conditional variable when c++ library doesn't implement
  // std::atomic::wait().
  mutable Mutex state_mutex_;
  mutable std::condition_variable state_signal_;
#endif
  std::thread gc_thread_;
  SpinMutex mutex_;
  std::deque<std::vector<std::unique_ptr<Node>>> released_nodes_ GUARDED_BY(mutex_);

  friend class ReleaseNodesWork<Node, kCapacity>;
};

template<typename Node, size_t kCapacity>
NodeGarbageCollector<Node, kCapacity>::NodeGarbageCollector() :
  gc_thread_{[this]() {GCThread();}} {
}

template<typename Node, size_t kCapacity>
template<typename UniquePtr>
void NodeGarbageCollector<Node, kCapacity>::AddToGcQueue(UniquePtr& shared_node) {
  std::unique_ptr<Node> node(shared_node.release());
  if (ShouldQueue(node)) {
    LocalWork().emplace_back(std::move(node));
  }
}

template<typename Node, size_t kCapacity>
NodeGarbageCollector<Node, kCapacity>::~NodeGarbageCollector() {
  state_.store(Exit, std::memory_order_release);
#ifndef NO_STD_ATOMIC_WAIT
  state_.notify_all();
#else
  {
    Mutex::Lock lock(state_mutex_);
    state_signal_.notify_all();
  }
#endif
  gc_thread_.join();
}

template<typename Node, size_t kCapacity>
bool NodeGarbageCollector<Node, kCapacity>::SetState(State& old, State desired) {
  bool rv =  state_.compare_exchange_strong(old, desired,
                                            std::memory_order_acq_rel);
  if (rv) {
#ifndef NO_STD_ATOMIC_WAIT
    state_.notify_all();
#else
    Mutex::Lock lock(state_mutex_);
    state_signal_.notify_all();
#endif
  }
  return rv;
}

template<typename Node, size_t kCapacity>
void NodeGarbageCollector<Node, kCapacity>::Start() {
  State s = state_.load(std::memory_order_acquire);
  do {
    if (s == Running)
      break;
    assert(s != Exit);
  } while (!SetState(s, Running));
}

template<typename Node, size_t kCapacity>
void NodeGarbageCollector<Node, kCapacity>::Stop() {
  State old = Running;
  SetState(old, GoToSleep);
}

template<typename Node, size_t kCapacity>
void NodeGarbageCollector<Node, kCapacity>::Abort() {
  Stop();
}

template<typename Node, size_t kCapacity>
NodeGarbageCollector<Node, kCapacity>::State NodeGarbageCollector<Node, kCapacity>::Wait() const {
  State s;
  while ((s = state_.load(std::memory_order_acquire)) != Sleeping) {
    assert(s != Exit);
#ifndef NO_STD_ATOMIC_WAIT
    state_.wait(s, std::memory_order_acquire);
#else
    Mutex::Lock lock(state_mutex_);
    state_signal_.wait(lock.get_raw(), [this, s]() {return s != state_;});
#endif
  }
  return s;
}

template<typename Node, size_t kCapacity>
void NodeGarbageCollector<Node, kCapacity>::NotifyThreadGoingSleep() {
  if (LocalWork().empty()) {
    return;
  }
  ReleaseNodesWork<Node, kCapacity> new_work;
  LocalWork().swap(new_work);
}

template<typename Node, size_t kCapacity>
bool NodeGarbageCollector<Node, kCapacity>::IsActive() const {
  return state_.load(std::memory_order_acquire) == Running;
}

template<typename Node, size_t kCapacity>
bool NodeGarbageCollector<Node, kCapacity>::ShouldQueue(std::unique_ptr<Node>& node) const {
  // We don't want to queue null pointers.
  if (!node) {
    return false;
  }

  // If state is exit, it means thread local queues have been destroyed.
  State s = state_.load(std::memory_order_acquire);
  if (s == Exit) {
    return false;
  }

  // We directly free the node, if queue is running and we are in the GC thread.
  // All other queue request should be pushed to the thread local batch.
  return s != Running || !LocalWork().IsWorker();
}

template<typename Node, size_t kCapacity>
void NodeGarbageCollector<Node, kCapacity>::GCThread() {
  auto& shared_work = LocalWork(true);
  assert(shared_work.IsWorker());
  State s;
  while ((s = state_.load(std::memory_order_acquire)) != Exit) {
    if (s == GoToSleep) {
      // Signal other threads that we have stopped destruction work.
      if (SetState(s, Sleeping)) {
        s = Sleeping;
      } else {
        continue;
      }
    }
    if (s == Sleeping) {
#ifndef NO_STD_ATOMIC_WAIT
      state_.wait(Sleeping, std::memory_order_acquire);
#else
      {
        Mutex::Lock lock(state_mutex_);
        state_signal_.wait(lock.get_raw(), [this]() {return Sleeping != state_;});
      }
#endif
      if (!shared_work.empty()) {
        // Check for early exit from previous free. The work can be freed
        // before the batch is full.
        ReleaseNodesWork<Node, kCapacity> new_work(true);
        new_work.swap(shared_work);
      }
      continue;
    }

    assert(s == Running);

    bool empty = true;
    std::vector<std::unique_ptr<Node>> nodes;
    {
      SpinMutex::Lock lock(mutex_);
      if (!released_nodes_.empty()) {
        empty = false;
        nodes = std::move(released_nodes_.front());
        released_nodes_.pop_front();
      }
    }

    // Free nodes one by one. LowNode destructor calls AddToGcQueue which allows
    // recursive destruction terminate before freeing a whole branch.
    while (!nodes.empty()) {
      if (!IsActive()) {
        break;
      }
      nodes.pop_back();
    }

    // Go to sleep if empty or search stopped.
    if (empty || !IsActive()) {
      // Lock is requrired to avoid race between other thread queueing work and
      // calling Start().
      SpinMutex::Lock lock(mutex_);
      // There wasn't enough time to free all nodes. They must go back to the
      // list.
      if (!nodes.empty()) {
        released_nodes_.emplace_front(std::move(nodes));
      }

      // Going to sleep if the queue is empty.
      if (released_nodes_.empty()) {
        State old = Running;
        SetState(old, Sleeping);
      }
    }
  }
}

template<typename Node, size_t kCapacity>
ReleaseNodesWork<Node, kCapacity>::ReleaseNodesWork(bool gc_thread) :
    is_gc_thread_(gc_thread) {
  released_nodes_.reserve(kCapacity);
}

template<typename Node, size_t kCapacity>
bool ReleaseNodesWork<Node, kCapacity>::IsWorker() const {
  return is_gc_thread_;
}

template<typename Node, size_t kCapacity>
void ReleaseNodesWork<Node, kCapacity>::emplace_back(std::unique_ptr<Node>&& node) {
  if (!node) return;
  released_nodes_.emplace_back(std::forward<std::unique_ptr<Node>>(node));
  if (released_nodes_.size() == kCapacity) {
    ReleaseNodesWork new_work(is_gc_thread_);
    swap(new_work);
  }
}

template<typename Node, size_t kCapacity>
bool ReleaseNodesWork<Node, kCapacity>::empty() const {
  return released_nodes_.empty();
}

template<typename Node, size_t kCapacity>
void ReleaseNodesWork<Node, kCapacity>::swap(ReleaseNodesWork &other) {
  assert(IsWorker() == other.IsWorker());
  std::swap(released_nodes_, other.released_nodes_);
}

template<typename Node, size_t kCapacity>
ReleaseNodesWork<Node, kCapacity>::~ReleaseNodesWork() {
  Submit();
}

template<typename Node, size_t kCapacity>
void ReleaseNodesWork<Node, kCapacity>::Submit() {
  if (released_nodes_.empty()) {
    return;
  }
  auto& worker = NodeGarbageCollector<Node, kCapacity>::Instance();
  SpinMutex::Lock lock(worker.mutex_);
  // If this is worker, we have oldest nodes. Keep them at front of the queue.
  if (IsWorker()) {
    worker.released_nodes_.emplace_front(std::move(released_nodes_));
  } else {
    worker.released_nodes_.emplace_back(std::move(released_nodes_));
  }
}

} // namespace lczero
