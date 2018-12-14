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

#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "utils/hashcat.h"

namespace lczero {

// Generic LRU cache. Thread-safe.
template <class K, class V>
class LruCache {
 public:
  LruCache(int capacity = 128) : capacity_(capacity) {}

  // Inserts the element under key @key with value @val.
  // If the element is pinned, old value is still kept (until fully unpinned),
  // but new lookups will return updated value.
  // If @pinned, pins inserted element, Unpin has to be called to unpin.
  // In any case, puts element to front of the queue (makes it last to evict).
  V* Insert(K key, std::unique_ptr<V> val, bool pinned = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    V* new_val = val.get();
    auto iter = lookup_.find(key);
    if (iter != lookup_.end()) {
      auto list_iter = iter->second;
      MakePending(list_iter);
      list_iter->pins = pinned ? 1 : 0;
      list_iter->value = std::move(val);
      BringToFront(list_iter);
    } else {
      MaybeCleanup(capacity_ - 1);
      lru_.emplace_front(key, std::move(val), pinned ? 1 : 0);
      lookup_[key] = lru_.begin();
    }
    return new_val;
  }

  // Looks up and pins the element by key. Returns nullptr if not found.
  // If found, brings the element to the head of the queue (makes it last to
  // evict).
  V* Lookup(K key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = lookup_.find(key);
    if (iter == lookup_.end()) return nullptr;
    ++iter->second->pins;
    return iter->second->value.get();
  }

  // Checks whether a key exists. Doesn't lock. Of course the next moment the
  // key may be evicted.
  bool ContainsKey(K key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = lookup_.find(key);
    return iter != lookup_.end();
  }

  // Unpins the element given key and value.
  void Unpin(K key, V* value) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto pending_iter = pending_pins_.find({key, value});
    if (pending_iter != pending_pins_.end()) {
      if (--pending_iter->second.pins == 0) pending_pins_.erase(pending_iter);
      return;
    }
    auto iter = lookup_.find(key);
    --iter->second->pins;
  }

  // Sets the capacity of the cache. If new capacity is less than current size
  // of the cache, oldest entries are evicted.
  void SetCapacity(int capacity) {
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_ = capacity;
    MaybeCleanup(capacity);
  }

  int GetSize() const { return lru_.size(); }
  int GetCapacity() const { return capacity_; }

 private:
  // Mutex should be locked when calling this function.
  void MaybeCleanup(int size) {
    if (size >= lookup_.size()) return;
    int to_delete = lookup_.size() - size;
    auto iter = std::prev(lru_.end());
    for (int i = 0; i < to_delete; ++i) {
      lookup_.erase(iter->key);
      MakePending(iter);
      iter = lru_.erase(iter);
      --iter;
    }
  }

  struct Item {
    Item(K key, std::unique_ptr<V> value, int pins)
        : key(key), value(std::move(value)), pins(pins) {}
    K key;
    std::unique_ptr<V> value;
    int pins;
  };

  void MakePending(typename std::list<Item>::iterator iter) {
    if (iter->pins > 0) {
      K key = iter->key;
      V* val = iter->value.get();
      int pins = iter->pins;
      pending_pins_.emplace(std::make_pair(key, val),
                            Item{key, std::move(iter->value), pins});
    }
  }

  void BringToFront(typename std::list<Item>::iterator iter) {
    if (iter != lru_.begin())
      lru_.splice(lru_.begin(), lru_, iter, std::next(iter));
  }
  // Fresh in front, stale on back.
  int capacity_;
  std::list<Item> lru_;
  using ListIter = typename std::list<Item>::iterator;
  std::unordered_map<K, ListIter> lookup_;

  struct PairHash {
    std::size_t operator()(const std::pair<K, V*>& p) const {
      return HashCat({std::hash<V*>{}(p.second), std::hash<K>{}(p.first)});
    }
  };
  std::unordered_map<std::pair<K, V*>, Item, PairHash> pending_pins_;
  std::mutex mutex_;
};

template <class K, class V>
class LruCacheLock {
 public:
  // Looks up the value in @cache by @key and pins it if found.
  LruCacheLock(LruCache<K, V>* cache, K key)
      : cache_(cache), key_(key), value_(cache->Lookup(key_)) {}

  // Unpins the cache entry (if holds).
  ~LruCacheLock() {
    if (value_) cache_->Unpin(key_, value_);
  }

  // Returns whether lock holds any value.
  operator bool() const { return value_; }

  // Gets the value.
  V* operator->() const { return value_; }
  V* operator*() const { return value_; }

  LruCacheLock() {}
  LruCacheLock(LruCacheLock&& other)
      : cache_(other.cache_), key_(other.key_), value_(other.value_) {
    other.value_ = nullptr;
  }
  void operator=(LruCacheLock&& other) {
    cache_ = other.cache_;
    key_ = other.key_;
    value_ = other.value_;
    other.value_ = nullptr;
  }

 private:
  LruCache<K, V>* cache_ = nullptr;
  K key_;
  V* value_ = nullptr;
};

}  // namespace lczero
