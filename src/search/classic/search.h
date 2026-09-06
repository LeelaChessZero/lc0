/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2023 The LCZero Authors

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

#include <array>
#include <condition_variable>
#include <functional>
#include <optional>
#include <shared_mutex>
#include <thread>

#include "chess/callbacks.h"
#include "chess/uciloop.h"
#include "neural/backend.h"
#include "search/classic/node.h"
#include "search/classic/params.h"
#include "search/classic/stoppers/timemgr.h"
#include "syzygy/syzygy.h"
#include "utils/logging.h"
#include "utils/mutex.h"

namespace lczero {
namespace classic {

class Search {
 public:
  Search(const NodeTree& tree, Backend* network,
         std::unique_ptr<UciResponder> uci_responder,
         const MoveList& searchmoves,
         std::chrono::steady_clock::time_point start_time,
         std::unique_ptr<SearchStopper> stopper, bool infinite, bool ponder,
         const OptionsDict& options, SyzygyTablebase* syzygy_tb);

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
  // return results from different possible moves. If @move and @is_terminal are
  // not nullptr they are set to the best move and whether it leads to a
  // terminal node respectively.
  Eval GetBestEval(Move* move = nullptr, bool* is_terminal = nullptr) const;
  // Returns the total number of playouts in the search.
  std::int64_t GetTotalPlayouts() const;
  // Returns the search parameters.
  const SearchParams& GetParams() const { return params_; }

  // If called after GetBestMove, another call to GetBestMove will have results
  // from temperature having been applied again.
  void ResetBestMove();

 private:
  // Computes the best move, maybe with temperature (according to the settings).
  void EnsureBestMoveKnown();

  // Returns a child with most visits, with or without temperature.
  // NoTemperature is safe to use on non-extended nodes, while WithTemperature
  // accepts only nodes with at least 1 visited child.
  EdgeAndNode GetBestChildNoTemperature(Node* parent, int depth) const;
  std::vector<EdgeAndNode> GetBestChildrenNoTemperature(Node* parent, int count,
                                                        int depth) const;
  EdgeAndNode GetBestRootChildWithTemperature(float temperature) const;

  int64_t GetTimeSinceStart() const;
  int64_t GetTimeSinceFirstBatch() const;
  void MaybeTriggerStop(const IterationStats& stats, StoppersHints* hints);
  void MaybeOutputInfo();
  void SendUciInfo();  // Requires nodes_mutex_ to be held.
  // Sets stop to true and notifies watchdog thread.
  void FireStopInternal();

  void SendMovesStats() const;
  // Function which runs in a separate thread and watches for time and
  // uci `stop` command;
  void WatchdogThread();

  // Fills IterationStats with global (rather than per-thread) portion of search
  // statistics. Currently all stats there (in IterationStats) are global
  // though.
  void PopulateCommonIterationStats(IterationStats* stats);

  // Returns verbose information about given node, as vector of strings.
  // Node can only be root or ponder (depth 1).
  std::vector<std::string> GetVerboseStats(const Node* node) const;

  // Returns the draw score at the root of the search. At odd depth pass true to
  // the value of @is_odd_depth to change the sign of the draw score.
  // Depth of a root node is 0 (even number).
  float GetDrawScore(bool is_odd_depth) const;

  // Ensure that all shared collisions are cancelled and clear them out.
  void CancelSharedCollisions();

  PositionHistory GetPositionHistoryAtNode(const Node* node) const;

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
  // Stored so that in the case of non-zero temperature GetBestMove() returns
  // consistent results.
  Move final_bestmove_ GUARDED_BY(counters_mutex_);
  Move final_pondermove_ GUARDED_BY(counters_mutex_);
  std::unique_ptr<SearchStopper> stopper_ GUARDED_BY(counters_mutex_);

  Mutex threads_mutex_;
  std::vector<std::thread> threads_ GUARDED_BY(threads_mutex_);

  Node* root_node_;
  SyzygyTablebase* syzygy_tb_;
  // Fixed positions which happened before the search.
  const PositionHistory& played_history_;

  Backend* const backend_;
  BackendAttributes backend_attributes_;
  const SearchParams params_;
  const MoveList searchmoves_;
  const std::chrono::steady_clock::time_point start_time_;
  int64_t initial_visits_;
  // root_is_in_dtz_ must be initialized before root_move_filter_.
  bool root_is_in_dtz_ = false;
  // tb_hits_ must be initialized before root_move_filter_.
  std::atomic<int> tb_hits_{0};
  const MoveList root_move_filter_;

  mutable SharedMutex nodes_mutex_;
  EdgeAndNode current_best_edge_ GUARDED_BY(nodes_mutex_);
  Edge* last_outputted_info_edge_ GUARDED_BY(nodes_mutex_) = nullptr;
  ThinkingInfo last_outputted_uci_info_ GUARDED_BY(nodes_mutex_);
  int64_t total_playouts_ GUARDED_BY(nodes_mutex_) = 0;
  int64_t network_evaluations_ GUARDED_BY(nodes_mutex_) = 0;
  int64_t total_batches_ GUARDED_BY(nodes_mutex_) = 0;
  // Maximum search depth = length of longest path taken in PickNodetoExtend.
  uint16_t max_depth_ GUARDED_BY(nodes_mutex_) = 0;
  // Cumulative depth of all paths taken in PickNodetoExtend.
  uint64_t cum_depth_ GUARDED_BY(nodes_mutex_) = 0;

  std::optional<std::chrono::steady_clock::time_point> nps_start_time_
      GUARDED_BY(counters_mutex_);

  std::atomic<int> pending_searchers_{0};
  std::atomic<int> backend_waiting_counter_{0};
  std::atomic<int> thread_count_{0};

  std::vector<std::pair<Node*, int>> shared_collisions_
      GUARDED_BY(nodes_mutex_);

  std::unique_ptr<UciResponder> uci_responder_;
  ContemptMode contempt_mode_;
  friend class SearchWorker;
};

// Single thread worker of the search engine.
// That used to be just a function Search::Worker(), but to parallelize it
// within one thread, have to split into stages.
class SearchWorker {
 public:
  SearchWorker(Search* search, const SearchParams& params)
      : search_(search),
        history_(search_->played_history_),
        params_(params),
        moves_left_support_(search_->backend_attributes_.has_mlh) {
    task_workers_ = params.GetTaskWorkersPerSearchWorker();
    if (task_workers_ < 0) {
      if (search_->backend_attributes_.runs_on_cpu) {
        task_workers_ = 0;
      } else {
        int working_threads = std::max(
            search_->thread_count_.load(std::memory_order_acquire) - 1, 1);
        task_workers_ = std::min(
            std::thread::hardware_concurrency() / working_threads - 1, 4U);
      }
    }
    for (int i = 0; i < task_workers_; i++) {
      task_workspaces_.emplace_back();
      task_threads_.emplace_back([this, i]() { this->RunTasks(i); });
    }
    target_minibatch_size_ = params_.GetMiniBatchSize();
    if (target_minibatch_size_ == 0) {
      target_minibatch_size_ =
          search_->backend_attributes_.recommended_batch_size;
    }
    max_out_of_order_ =
        std::max(1, static_cast<int>(params_.GetMaxOutOfOrderEvalsFactor() *
                                     target_minibatch_size_));
  }

  ~SearchWorker() {
    {
      task_count_.store(-1, std::memory_order_release);
      Mutex::Lock lock(picking_tasks_mutex_);
      exiting_ = true;
      task_added_.notify_all();
    }
    for (size_t i = 0; i < task_threads_.size(); i++) {
      task_threads_[i].join();
    }
  }

  // Runs iterations while needed.
  void RunBlocking() {
    LOGFILE << "Started search thread.";
    try {
      // A very early stop may arrive before this point, so the test is at the
      // end to ensure at least one iteration runs before exiting.
      do {
        ExecuteOneIteration();
      } while (search_->IsSearchActive());
    } catch (std::exception& e) {
      std::cerr << "Unhandled exception in worker thread: " << e.what()
                << std::endl;
      abort();
    }
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
  void InitializeIteration();

  // 2. Gather minibatch.
  void GatherMinibatch();

  // 2b. Copy collisions into shared_collisions_.
  void CollectCollisions();

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
    std::unique_ptr<EvalResult> eval;
    int multivisit = 0;
    // If greater than multivisit, and other parameters don't imply a lower
    // limit, multivist could be increased to this value without additional
    // change in outcome of next selection.
    int maxvisit = 0;
    uint16_t depth;
    bool nn_queried = false;
    bool is_cache_hit = false;
    bool is_collision = false;
    // Only populated for visits,
    std::vector<Move> moves_to_visit;

    // Details that are filled in as we go.
    bool ooo_completed = false;

    static NodeToProcess Collision(Node* node, uint16_t depth,
                                   int collision_count) {
      return NodeToProcess(node, depth, true, collision_count, 0);
    }
    static NodeToProcess Collision(Node* node, uint16_t depth,
                                   int collision_count, int max_count) {
      return NodeToProcess(node, depth, true, collision_count, max_count);
    }
    static NodeToProcess Visit(Node* node, uint16_t depth) {
      return NodeToProcess(node, depth, false, 1, 0);
    }

   private:
    NodeToProcess(Node* node, uint16_t depth, bool is_collision, int multivisit,
                  int max_count)
        : node(node),
          eval(std::make_unique<EvalResult>()),
          multivisit(multivisit),
          maxvisit(max_count),
          depth(depth),
          is_collision(is_collision) {}
  };

  // Holds per task worker scratch data
  struct TaskWorkspace {
    std::array<Node::Iterator, 256> cur_iters;
    std::vector<std::unique_ptr<std::array<int, 256>>> vtp_buffer;
    std::vector<std::unique_ptr<std::array<int, 256>>> visits_to_perform;
    std::vector<int> vtp_last_filled;
    std::vector<int> current_path;
    std::vector<Move> moves_to_path;
    PositionHistory history;
    TaskWorkspace() {
      vtp_buffer.reserve(30);
      visits_to_perform.reserve(30);
      vtp_last_filled.reserve(30);
      current_path.reserve(30);
      moves_to_path.reserve(30);
      history.Reserve(30);
    }
  };

  struct PickTask {
    enum PickTaskType { kGathering, kProcessing };
    PickTaskType task_type;

    // For task type gathering.
    Node* start;
    int base_depth;
    int collision_limit;
    std::vector<Move> moves_to_base;
    std::vector<NodeToProcess> results;

    // Task type post gather processing.
    int start_idx;
    int end_idx;

    bool complete = false;

    PickTask(Node* node, uint16_t depth, const std::vector<Move>& base_moves,
             int collision_limit)
        : task_type(kGathering),
          start(node),
          base_depth(depth),
          collision_limit(collision_limit),
          moves_to_base(base_moves) {}
    PickTask(int start_idx, int end_idx)
        : task_type(kProcessing), start_idx(start_idx), end_idx(end_idx) {}
  };

  NodeToProcess PickNodeToExtend(int collision_limit);
  int PrefetchIntoCache(Node* node, int budget, bool is_odd_depth);
  void DoBackupUpdateSingleNode(const NodeToProcess& node_to_process);
  // Returns whether a node's bounds were set based on its children.
  bool MaybeSetBounds(Node* p, float m, int* n_to_fix, float* v_delta,
                      float* d_delta, float* m_delta) const;
  void PickNodesToExtend(int collision_limit);
  void PickNodesToExtendTask(Node* starting_point, int base_depth,
                             int collision_limit,
                             const std::vector<Move>& moves_to_base,
                             std::vector<NodeToProcess>* receiver,
                             TaskWorkspace* workspace);
  void EnsureNodeTwoFoldCorrectForDepth(Node* node, int depth);
  void ProcessPickedTask(int batch_start, int batch_end,
                         TaskWorkspace* workspace);
  void ExtendNode(Node* node, int depth, const std::vector<Move>& moves_to_add,
                  PositionHistory* history);
  void FetchSingleNodeResult(NodeToProcess* node_to_process);
  void RunTasks(int tid);
  void ResetTasks();
  // Returns how many tasks there were.
  int WaitForTasks();

  Search* const search_;
  // List of nodes to process.
  std::vector<NodeToProcess> minibatch_;
  std::unique_ptr<BackendComputation> computation_;
  int task_workers_;
  int target_minibatch_size_;
  int max_out_of_order_;
  // History is reset and extended by PickNodeToExtend().
  PositionHistory history_;
  int number_out_of_order_ = 0;
  const SearchParams& params_;
  const bool moves_left_support_;
  IterationStats iteration_stats_;
  StoppersHints latest_time_manager_hints_;

  // Multigather task related fields.

  Mutex picking_tasks_mutex_;
  std::vector<PickTask> picking_tasks_;
  std::atomic<int> task_count_ = -1;
  std::atomic<int> task_taking_started_ = 0;
  std::atomic<int> tasks_taken_ = 0;
  std::atomic<int> completed_tasks_ = 0;
  std::condition_variable task_added_;
  std::vector<std::thread> task_threads_;
  std::vector<TaskWorkspace> task_workspaces_;
  TaskWorkspace main_workspace_;
  bool exiting_ = false;
};

}  // namespace classic
}  // namespace lczero
