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

#include <boost/process.hpp>
#include <functional>
#include <shared_mutex>
#include <thread>
#include <queue>
#include "chess/callbacks.h"
#include "chess/uciloop.h"
#include "mcts/node.h"
#include "mcts/params.h"
#include "neural/cache.h"
#include "neural/network.h"
#include "syzygy/syzygy.h"
#include "utils/logging.h"
#include "utils/mutex.h"
#include "utils/optional.h"

namespace lczero {

struct SearchLimits {
  // Type for N in nodes is currently uint32_t, so set limit in order not to
  // overflow it.
  std::int64_t visits = 4000000000;
  std::int64_t playouts = -1;
  int depth = -1;
  optional<std::chrono::steady_clock::time_point> search_deadline;
  bool infinite = false;
  MoveList searchmoves;

  std::string DebugString() const;
};

class Search {
 public:
  Search(const NodeTree& tree, Network* network,
         BestMoveInfo::Callback best_move_callback,
         ThinkingInfo::Callback info_callback, const SearchLimits& limits,
         const OptionsDict& options, NNCache* cache,
         SyzygyTablebase* syzygy_tb);

  ~Search();

  // Starts worker threads and returns immediately.
  void StartThreads(size_t how_many);

  // Starts search with k threads and wait until it finishes.
  void RunBlocking(size_t threads);

  // Stops search. At the end bestmove will be returned. The function is not
  // blocking, so it returns before search is actually done.
  void Stop();
  // Stops search, but does not return bestmove. The function is not blocking.
  void Abort();
  // Blocks until all worker thread finish.
  void Wait();
  // Returns whether search is active. Workers check that to see whether another
  // search iteration is needed.
  bool IsSearchActive() const;

  // Returns best move, from the point of view of white player. And also ponder.
  // May or may not use temperature, according to the settings.
  std::pair<Move, Move> GetBestMove();
  // Returns the evaluation of the best move, WITHOUT temperature. This differs
  // from the above function; with temperature enabled, these two functions may
  // return results from different possible moves.
  // Returns pair {Q, D}.
  std::pair<float, float> GetBestEval() const;
  // Returns the total number of playouts in the search.
  std::int64_t GetTotalPlayouts() const;
  // Returns the search parameters.
  const SearchParams& GetParams() const { return params_; }

  //CurrentPosition current_position_;
  std::string current_position_fen_;
  std::vector<std::string> current_position_moves_;
  std::string current_uci_;

 private:
  // Computes the best move, maybe with temperature (according to the settings).
  void EnsureBestMoveKnown();

  // Returns a child with most visits, with or without temperature.
  // NoTemperature is safe to use on non-extended nodes, while WithTemperature
  // accepts only nodes with at least 1 visited child.
  EdgeAndNode GetBestChildNoTemperature(Node* parent) const;
  std::vector<EdgeAndNode> GetBestChildrenNoTemperature(Node* parent,
                                                        int count) const;
  EdgeAndNode GetBestChildWithTemperature(Node* parent,
                                          float temperature) const;

  int64_t GetTimeSinceStart() const;
  int64_t GetTimeToDeadline() const;
  void UpdateRemainingMoves();
  void UpdateKLDGain();
  void MaybeTriggerStop();
  void MaybeOutputInfo();
  void SendUciInfo();  // Requires nodes_mutex_ to be held.
  // Sets stop to true and notifies watchdog thread.
  void FireStopInternal();

  void SendMovesStats() const;
  // Function which runs in a separate thread and watches for time and
  // uci `stop` command;
  void WatchdogThread();

  // Populates the given list with allowed root moves.
  // Returns true if the population came from tablebase.
  bool PopulateRootMoveLimit(MoveList* root_moves) const;

  // Returns verbose information about given node, as vector of strings.
  std::vector<std::string> GetVerboseStats(Node* node,
                                           bool is_black_to_move) const;

  // Returns NN eval for a given node from cache, if that node is cached.
  NNCacheLock GetCachedNNEval(Node* node) const;

  mutable Mutex counters_mutex_ ACQUIRED_AFTER(nodes_mutex_);
  // Tells all threads to stop.
  std::atomic<bool> stop_{false};
  // Condition variable used to watch stop_ variable.
  std::condition_variable watchdog_cv_;
  // Tells whether it's ok to respond bestmove when limits are reached.
  // If false (e.g. during ponder or `go infinite`) the search stops but nothing
  // is responded until `stop` uci command.
  bool ok_to_respond_bestmove_ GUARDED_BY(counters_mutex_) = true;
  // There is already one thread that responded bestmove, other threads
  // should not do that.
  bool bestmove_is_sent_ GUARDED_BY(counters_mutex_) = false;
  // Becomes true when smart pruning decides that no better move can be found.
  bool only_one_possible_move_left_ GUARDED_BY(counters_mutex_) = false;
  // Stored so that in the case of non-zero temperature GetBestMove() returns
  // consistent results.
  EdgeAndNode final_bestmove_ GUARDED_BY(counters_mutex_);
  EdgeAndNode final_pondermove_ GUARDED_BY(counters_mutex_);

  Mutex threads_mutex_;
  std::vector<std::thread> threads_ GUARDED_BY(threads_mutex_);

  Node* root_node_;
  NNCache* cache_;
  SyzygyTablebase* syzygy_tb_;
  // Fixed positions which happened before the search.
  const PositionHistory& played_history_;

  Network* const network_;
  const SearchLimits limits_;
  const std::chrono::steady_clock::time_point start_time_;
  const int64_t initial_visits_;
  optional<std::chrono::steady_clock::time_point> nps_start_time_;

  mutable SharedMutex nodes_mutex_;
  EdgeAndNode current_best_edge_ GUARDED_BY(nodes_mutex_);
  Edge* last_outputted_info_edge_ GUARDED_BY(nodes_mutex_) = nullptr;
  ThinkingInfo last_outputted_uci_info_ GUARDED_BY(nodes_mutex_);
  int64_t total_playouts_ GUARDED_BY(nodes_mutex_) = 0;
  int64_t remaining_playouts_ GUARDED_BY(nodes_mutex_) =
      std::numeric_limits<int64_t>::max();
  // If kldgain minimum checks enabled, this was the visit distribution at the
  // last kldgain interval triggering.
  std::vector<uint32_t> prev_dist_ GUARDED_BY(counters_mutex_);
  // Total visits at the last time prev_dist_ was cached.
  uint32_t prev_dist_visits_total_ GUARDED_BY(counters_mutex_) = 0;
  // If true, search should exit as kldgain evaluation showed too little change.
  bool kldgain_too_small_ GUARDED_BY(counters_mutex_) = false;
  // Maximum search depth = length of longest path taken in PickNodetoExtend.
  uint16_t max_depth_ GUARDED_BY(nodes_mutex_) = 0;
  // Cummulative depth of all paths taken in PickNodetoExtend.
  uint64_t cum_depth_ GUARDED_BY(nodes_mutex_) = 0;
  std::atomic<int> tb_hits_{0};

  BestMoveInfo::Callback best_move_callback_;
  ThinkingInfo::Callback info_callback_;
  const SearchParams params_;

  void OpenAuxEngine();
  void AuxEngineWorker();
  void AuxWait();
  void DoAuxEngine(Node* n);
  static boost::process::ipstream auxengine_is_;
  static boost::process::opstream auxengine_os_;
  static boost::process::child auxengine_c_;
  static bool auxengine_ready_;
  std::queue<Node*> auxengine_queue_;
  std::mutex auxengine_mutex_;
  std::condition_variable auxengine_cv_;
  std::vector<std::thread> auxengine_threads_;

  friend class SearchWorker;
};

// Single thread worker of the search engine.
// That used to be just a function Search::Worker(), but to parallelize it
// within one thread, have to split into stages.
class SearchWorker {
 public:
  SearchWorker(Search* search, const SearchParams& params)
      : search_(search), history_(search_->played_history_), params_(params) {}

  // Runs iterations while needed.
  void RunBlocking() {
    LOGFILE << "Started search thread.";
    // A very early stop may arrive before this point, so the test is at the end
    // to ensure at least one iteration runs before exiting.
    do {
      ExecuteOneIteration();
    } while (search_->IsSearchActive());
  }

  // Does one full iteration of MCTS search:
  // 1. Initialize internal structures.
  // 2. Gather minibatch.
  // 3. Prefetch into cache.
  // 4. Run NN computation.
  // 5. Retrieve NN computations (and terminal values) into nodes.
  // 6. Propagate the new nodes' information to all their parents in the tree.
  // 7. Update the Search's status and progress information.
  void ExecuteOneIteration();

  // The same operations one by one:
  // 1. Initialize internal structures.
  // @computation is the computation to use on this iteration.
  void InitializeIteration(std::unique_ptr<NetworkComputation> computation);

  // 2. Gather minibatch.
  void GatherMinibatch();

  // 3. Prefetch into cache.
  void MaybePrefetchIntoCache();

  // 4. Run NN computation.
  void RunNNComputation();

  // 5. Retrieve NN computations (and terminal values) into nodes.
  void FetchMinibatchResults();

  // 6. Propagate the new nodes' information to all their parents in the tree.
  void DoBackupUpdate();

  // 7. Update the Search's status and progress information.
  void UpdateCounters();

 private:
  struct NodeToProcess {
    bool IsExtendable() const { return !is_collision && !node->IsTerminal(); }
    bool IsCollision() const { return is_collision; }
    bool CanEvalOutOfOrder() const {
      return is_cache_hit || node->IsTerminal();
    }

    // The node to extend.
    Node* node;
    // Value from NN's value head, or -1/0/1 for terminal nodes.
    float v;
    // Draw probability for NN's with WDL value head
    float d;
    int multivisit = 0;
    uint16_t depth;
    bool nn_queried = false;
    bool is_cache_hit = false;
    bool is_collision = false;

    static NodeToProcess Collision(Node* node, uint16_t depth,
                                   int collision_count) {
      return NodeToProcess(node, depth, true, collision_count);
    }
    static NodeToProcess Extension(Node* node, uint16_t depth) {
      return NodeToProcess(node, depth, false, 1);
    }
    static NodeToProcess TerminalHit(Node* node, uint16_t depth,
                                     int visit_count) {
      return NodeToProcess(node, depth, false, visit_count);
    }

   private:
    NodeToProcess(Node* node, uint16_t depth, bool is_collision, int multivisit)
        : node(node),
          multivisit(multivisit),
          depth(depth),
          is_collision(is_collision) {}
  };

  NodeToProcess PickNodeToExtend(int collision_limit);
  void ExtendNode(Node* node);
  bool AddNodeToComputation(Node* node, bool add_if_cached);
  int PrefetchIntoCache(Node* node, int budget);
  void FetchSingleNodeResult(NodeToProcess* node_to_process,
                             int idx_in_computation);
  void DoBackupUpdateSingleNode(const NodeToProcess& node_to_process);

  Search* const search_;
  // List of nodes to process.
  std::vector<NodeToProcess> minibatch_;
  std::unique_ptr<CachingComputation> computation_;
  // History is reset and extended by PickNodeToExtend().
  PositionHistory history_;
  MoveList root_move_filter_;
  bool root_move_filter_populated_ = false;
  int number_out_of_order_ = 0;
  const SearchParams& params_;
  std::unique_ptr<Node> precached_node_;

  void AuxMaybeEnqueueNode(Node* n);
};

}  // namespace lczero
