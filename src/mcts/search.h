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

#include <functional>
#include <shared_mutex>
#include <thread>
#include "chess/callbacks.h"
#include "chess/uciloop.h"
#include "mcts/node.h"
#include "neural/cache.h"
#include "neural/network.h"
#include "utils/mutex.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"

namespace lczero {

struct SearchLimits {
  std::int64_t visits = -1;
  std::int64_t playouts = -1;
  std::int64_t time_ms = -1;
  bool infinite = false;
  MoveList searchmoves;
};

class Search {
 public:
  Search(const NodeTree& tree, Network* network,
         BestMoveInfo::Callback best_move_callback,
         ThinkingInfo::Callback info_callback, const SearchLimits& limits,
         const OptionsDict& options, NNCache* cache);

  ~Search();

  // Populates UciOptions with search parameters.
  static void PopulateUciParams(OptionsParser* options);

  // Starts worker threads and returns immediately.
  void StartThreads(size_t how_many);

  // Starts search with k threads and wait until it finishes.
  void RunBlocking(size_t threads);

  // Runs search single-threaded, blocking.
  void RunSingleThreaded();

  // Stops search. At the end bestmove will be returned. The function is not
  // blocking, so it returns before search is actually done.
  void Stop();
  // Stops search, but does not return bestmove. The function is not blocking.
  void Abort();
  // Blocks until all worker thread finish.
  void Wait();

  // Returns best move, from the point of view of white player. And also ponder.
  std::pair<Move, Move> GetBestMove() const;

  // Strings for UCI params. So that others can override defaults.
  static const char* kMiniBatchSizeStr;
  static const char* kMaxPrefetchBatchStr;
  static const char* kCpuctStr;
  static const char* kTemperatureStr;
  static const char* kTempDecayMovesStr;
  static const char* kNoiseStr;
  static const char* kVerboseStatsStr;
  static const char* kSmartPruningStr;
  static const char* kVirtualLossBugStr;
  static const char* kFpuReductionStr;
  static const char* kCacheHistoryLengthStr;
  static const char* kExtraVirtualLossStr;
  static const char* kPolicySoftmaxTempStr;
  static const char* kAllowedNodeCollisionsStr;

 private:
  // Can run several copies of it in separate threads.
  void Worker();

  std::pair<Move, Move> GetBestMoveInternal() const;

  // Returns a child with most visits.
  Node* GetBestChild(Node* parent) const;
  Node* GetBestChildWithTemperature(Node* parent, float temperature) const;

  int64_t GetTimeSinceStart() const;
  void UpdateRemainingMoves();
  void MaybeTriggerStop();
  void MaybeOutputInfo();
  void SendMovesStats() const;
  bool AddNodeToCompute(Node* node, CachingComputation* computation,
                        const PositionHistory& history,
                        bool add_if_cached = true);
  int PrefetchIntoCache(Node* node, int budget, CachingComputation* computation,
                        PositionHistory* history);

  void SendUciInfo();  // Requires nodes_mutex_ to be held.

  std::pair<Node*, bool> PickNodeToExtend(Node* node, PositionHistory* history);
  void ExtendNode(Node* node, const PositionHistory& history);

  mutable Mutex counters_mutex_ ACQUIRED_AFTER(nodes_mutex_);
  // Tells all threads to stop.
  bool stop_ GUARDED_BY(counters_mutex_) = false;
  // There is already one thread that responded bestmove, other threads
  // should not do that.
  bool responded_bestmove_ GUARDED_BY(counters_mutex_) = false;
  // Becomes true when smart pruning decides
  bool found_best_move_ GUARDED_BY(counters_mutex_) = false;
  // Stored so that in the case of non-zero temperature GetBestMove() returns
  // consistent results.
  std::pair<Move, Move> best_move_ GUARDED_BY(counters_mutex_);

  Mutex threads_mutex_;
  std::vector<std::thread> threads_ GUARDED_BY(threads_mutex_);

  Node* root_node_;
  NNCache* cache_;
  // Fixed positions which happened before the search.
  const PositionHistory& played_history_;

  Network* const network_;
  const SearchLimits limits_;
  const std::chrono::steady_clock::time_point start_time_;
  const int64_t initial_visits_;

  mutable SharedMutex nodes_mutex_;
  Node* best_move_node_ GUARDED_BY(nodes_mutex_) = nullptr;
  Node* last_outputted_best_move_node_ GUARDED_BY(nodes_mutex_) = nullptr;
  ThinkingInfo uci_info_ GUARDED_BY(nodes_mutex_);
  int64_t total_playouts_ GUARDED_BY(nodes_mutex_) = 0;
  int remaining_playouts_ GUARDED_BY(nodes_mutex_) =
      std::numeric_limits<int>::max();

  BestMoveInfo::Callback best_move_callback_;
  ThinkingInfo::Callback info_callback_;

  // External parameters.
  const int kMiniBatchSize;
  const int kMaxPrefetchBatch;
  const float kCpuct;
  const float kTemperature;
  const int kTempDecayMoves;
  const bool kNoise;
  const bool kVerboseStats;
  const bool kSmartPruning;
  const float kVirtualLossBug;
  const float kFpuReduction;
  const bool kCacheHistoryLength;
  const float kExtraVirtualLoss;
  const float kPolicySoftmaxTemp;
  const int kAllowedNodeCollisions;
};

}  // namespace lczero
