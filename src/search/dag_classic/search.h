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

#include <absl/cleanup/cleanup.h>

#include <array>
#include <bit>
#include <condition_variable>
#include <optional>
#include <span>
#include <thread>
#include <tuple>
#include <vector>

#include "chess/callbacks.h"
#include "neural/backend.h"
#include "search/classic/stoppers/timemgr.h"
#include "search/dag_classic/node.h"
#include "search/dag_classic/params.h"
#include "syzygy/syzygy.h"
#include "utils/atomic_vector.h"
#include "utils/logging.h"
#include "utils/mutex.h"

#if __cpp_lib_atomic_wait < 201907L
#include <condition_variable>
#define NO_STD_ATOMIC_WAIT 1
#endif

namespace lczero {
namespace dag_classic {

class SearchWorker;
class StackLikeArenaTag {};

// Simple memory allocation arena which allows cheap batched deallocation.
template <size_t bytes, size_t alignment>
class StackLikeArena {
 public:
  StackLikeArena() = default;
  StackLikeArena(StackLikeArenaTag) : StackLikeArena() {}
  StackLikeArena(const StackLikeArena&) = delete;
  ~StackLikeArena();

  char* Begin() { return std::begin(buffer_); }
  char* End() { return std::end(buffer_); }

 private:
  alignas(alignment) char buffer_[bytes];
};

// Implement memory allocation which has automatic life SearchWorker iteration
// life time. This memory can be allocated rapidly from a StackLikeAreana.
// Deallocation will be delayed until the next iteration starts.
class IterationMemoryManager {
 public:
  using ArenaType = StackLikeArena<16 * 1024 - sizeof(void*) - sizeof(size_t),
                                   alignof(std::max_align_t)>;
  using ArenaTuple = std::tuple<ArenaType, size_t>;
  IterationMemoryManager();
  IterationMemoryManager(const IterationMemoryManager&) = delete;
  IterationMemoryManager(IterationMemoryManager&&) = default;
  IterationMemoryManager& operator=(IterationMemoryManager&&) = default;

  template <typename T>
  T* Allocate(size_t n);

  // Thread local manager for Allocator.
  static IterationMemoryManager& LocalManager();
  // Tells manager that a new iteration has started. It must be called before
  // the first call to LocalManager in the thread.
  static void ResetLocalManager(SearchWorker& worker, int tid);

 private:
  // Threads add randmoness to where thinks are allocated. Deallocations will be
  // delayed for a few seconds to avoid constantly allocating and deallocating
  // memory.
  static constexpr size_t kMaxArenaAge = 200;
  // Helper to acess the active arena.
  ArenaType& GetActiveArena();
  // Activate a new empty arena.
  ArenaType& GetNewArena();
  // Honor type required alignment.
  char* AlignedPointer(size_t align);

  // If age has changed, move all allocations to the free list.
  void Reset(size_t age);
  // Helper to check the age of active arena.
  size_t Age() const;

  std::forward_list<ArenaTuple> alloc_;
  std::forward_list<ArenaTuple> free_;
  char* pointer_;

  static thread_local IterationMemoryManager* local_manager_;
};

// Allocator interface to IterationMemoryManager. It can be used for stl
// containers or replacement for new/unique_ptr.
template <typename T>
class IterationMemoryAllocator {
 public:
  using value_type = T;
  using propagate_on_container_move_assignment = std::false_type;

  IterationMemoryAllocator(const IterationMemoryAllocator&) = default;
  IterationMemoryAllocator& operator=(const IterationMemoryAllocator&) = delete;
  template <typename U>
  IterationMemoryAllocator(const IterationMemoryAllocator<U>&) {}

  IterationMemoryAllocator() = default;

  T* allocate(size_t n);
  void deallocate(T*, size_t) noexcept;

 private:
  template <typename U>
  friend class IterationMemoryAllocator;
};

// The tuple elements are (node, repetitons, moves left).
typedef std::vector<std::tuple<Node*, int, int>> BackupPath;

struct SearchWorkerCachedState;
struct SearchCachedState;

#if __cpp_lib_hardware_interference_size >= 201603
static constexpr auto kCacheLineSize =
    std::hardware_destructive_interference_size;
#else
static constexpr size_t kCacheLineSize = 64;
#endif

class TaskQueue {
 public:
  static constexpr int kTaskCountDigits = std::numeric_limits<int>::digits + 1;
  static constexpr int kTasksTakenShift = kTaskCountDigits / 2;
  static constexpr int kTasksTakenOne = 1 << kTasksTakenShift;

  TaskQueue();
  ~TaskQueue();

  // Task base type. Derived classes can be scheduled to task workers. Task
  // worker threads start counting from thread id 1 because zero is reserved for
  // the thraed which schedules tasks. Derived classes should use final keyword
  // to let compiler optimise virtual function calls.
  struct PickTask {
    PickTask(const PickTask& other) = default;
    PickTask() = default;
    virtual ~PickTask();

    void operator()(int tid);

    // Dervied class should implement it if there is need to wait for a specific
    // task completing. Default implementation is to do nothing.
    virtual void Wait(int tid) const { std::ignore = tid; };

   private:
    // Dervied class implements task processing. It is called from
    // operator()(int tid).
    virtual void DoTask(int tid) = 0;
  };

  using PickTaskPtr = std::atomic<const PickTask*>;

  size_t Size() const;

  bool IsTasksIdle() const;
  PickTask* PickTaskToProcess();
  // Process a queued task.
  void ProcessTask(int tid);
  // Submit list of tasks to the queue.
  template <typename TaskVector>
  void SubmitTasks(const TaskVector& tasks, int tid);
  template <typename TaskType>
  void SubmitTask(const TaskType& task, int tid);
  // Activate worker threads when we are about to submit tasks.
  void ActivateTasks();
  // Deactivate worker threads.
  void DeactivateTasks();

  // Make sure the state matches the latest user configuration.
  void StartANewSearch(size_t task_workers);

 private:
  void ShutdownThreads();
  void RunTasks(int tid);

  std::vector<std::thread> task_threads_;
  bool exiting_ = false;
  alignas(kCacheLineSize) std::condition_variable task_added_;
  Mutex picking_tasks_mutex_;
  // Size is the smalles power of two which has enough space to hold all child
  // nodes in any positions. Typically there is much less visited children. The
  // bigger size helps avoid cache line contention when scaling to more threads.
  std::array<PickTaskPtr, 256> picking_tasks_;
  // A packed atomic. LSB half is task_count_. MSB half is tasks_taken_.
  alignas(kCacheLineSize) std::atomic<int> task_count_ = 0;
  alignas(kCacheLineSize) std::atomic<int> active_users_ = 0;
  alignas(kCacheLineSize) std::atomic<int> sleeping_threads_ = 0;
};

class Search {
 public:
  Search(SearchCachedState& state, const NodeTree& tree, Backend* backend,
         std::unique_ptr<UciResponder> uci_responder,
         const MoveList& searchmoves,
         std::chrono::steady_clock::time_point start_time,
         std::unique_ptr<classic::SearchStopper> stopper, bool infinite,
         bool ponder, const OptionsDict& options, TranspositionTable* tt,
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
  // return results from different possible moves. If @move and @is_terminal are
  // not nullptr they are set to the best move and whether it leads to a
  // terminal node respectively.
  Eval GetBestEval(Move* move = nullptr, bool* is_terminal = nullptr) const;
  // Returns the total number of playouts in the search.
  std::int64_t GetTotalPlayouts() const;

  // If called after GetBestMove, another call to GetBestMove will have results
  // from temperature having been applied again.
  void ResetBestMove();

  void RecordNPSStartTime();

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
  void MaybeTriggerStop(const classic::IterationStats& stats,
                        classic::StoppersHints* hints);
  void MaybeOutputInfo(const classic::IterationStats& stats);
  // Requires nodes_mutex_ to be held.
  void SendUciInfo(const classic::IterationStats& stats);
  // Sets stop to true and notifies watchdog thread.
  void FireStopInternal();

  void SendMovesStats() const;
  // Function which runs in a separate thread and watches for time and
  // uci `stop` command;
  void WatchdogThread();

  // Fills IterationStats with global (rather than per-thread) portion of search
  // statistics. Currently all stats there (in IterationStats) are global
  // though.
  void PopulateCommonIterationStats(classic::IterationStats* stats);

  // Returns verbose information about given node, as vector of strings.
  // Node can only be root or ponder (depth 1) and move_to_node is only given
  // for the ponder node.
  std::vector<std::string> GetVerboseStats(
      const Node* node, std::optional<Move> move_to_node) const;

  // Returns the draw score at the root of the search. At odd depth pass true to
  // the value of @is_odd_depth to change the sign of the draw score.
  // Depth of a root node is 0 (even number).
  float GetDrawScore(bool is_odd_depth) const;

  SearchCachedState& state_;
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
  // Node garbage collection has been started for this search.
  bool gc_started_ GUARDED_BY(counters_mutex_) = false;
  // Stored so that in the case of non-zero temperature GetBestMove() returns
  // consistent results.
  Move final_bestmove_ GUARDED_BY(counters_mutex_);
  Move final_pondermove_ GUARDED_BY(counters_mutex_);
  std::unique_ptr<classic::SearchStopper> stopper_ GUARDED_BY(counters_mutex_);

  // List of threads. Only accessed by main thread.
  std::vector<std::thread> threads_;

  Node* root_node_;
  TranspositionTable* tt_;
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
  int64_t total_batches_ GUARDED_BY(nodes_mutex_) = 0;
  // Maximum search depth = length of longest path taken in PickNodetoExtend.
  uint16_t max_depth_ GUARDED_BY(nodes_mutex_) = 0;
  // Cumulative depth of all paths taken in PickNodetoExtend.
  uint64_t cum_depth_ GUARDED_BY(nodes_mutex_) = 0;

  // The start time of search. It is set when the first thread exits
  // GatherMinibatch. It is guarded by nodes mutex until set once.
  std::optional<std::chrono::steady_clock::time_point> nps_start_time_;

  std::atomic<int> pending_searchers_{0};
  std::atomic<int> backend_waiting_counter_{0};
  std::atomic<int> thread_count_{0};
#if NO_STD_ATOMIC_WAIT
  Mutex fallback_threads_mutex_;
  std::condition_variable fallback_threads_cond_;
#endif
  int total_workers_ = 0;

  std::unique_ptr<UciResponder> uci_responder_;
  ContemptMode contempt_mode_;
  friend class SearchWorker;
};

template <typename TaskType>
using TaskVector = std::vector<TaskType, IterationMemoryAllocator<TaskType>>;

// Combine visits to perform, index, and node state flags into a packed
// variable. Packed value stores required visit infromation which can be
// pushed into the current_path stack.
struct CurrentPath {
  struct Bits {
    uint32_t visits_ : 21;       // <= collision limit
    uint32_t last_child_ : 1;    // bool
    uint32_t visit_child_ : 1;   // bool
    uint32_t stop_picking_ : 1;  // bool
    uint32_t index_ : 8;         // < 218
  };
  union {
    Bits bits_;
    uint32_t value_;
  };
  CurrentPath(unsigned visits, bool last, bool visit, bool stop, unsigned index)
      : bits_(visits, last, visit, stop, index) {}
  // Implicit conversion from int to allow comparing to a visit integer.
  CurrentPath(int visits) : bits_(visits, 0, 0, 0, 0) {}
  CurrentPath() {}

  auto operator<=>(CurrentPath b) const {
    return (uint32_t)bits_.visits_ <=> (uint32_t)b.bits_.visits_;
  }
  bool operator==(CurrentPath b) const {
    return (uint32_t)bits_.visits_ == (uint32_t)b.bits_.visits_;
  }
  explicit operator bool() const { return !!bits_.visits_; }

  CurrentPath& operator+=(unsigned visits) {
    CurrentPath temp(*this);
    std::ignore = temp;
    assert(temp.bits_.visits_ += visits == visits + bits_.visits_);
    value_ += visits;
    return *this;
  }
  CurrentPath& operator-=(unsigned visits) {
    assert(bits_.visits_ >= visits);
    value_ -= visits;
    return *this;
  }
};

struct TaskWorkspace;

// Single thread worker of the search engine.
// That used to be just a function Search::Worker(), but to parallelize it
// within one thread, have to split into stages.
class SearchWorker {
 public:
  static constexpr int kMaxMovesInPosition = 218;

  SearchWorker(int tid, SearchWorkerCachedState& state, Search* search,
               const SearchParams& params);

  ~SearchWorker();

  // Runs iterations while needed.
  void RunBlocking();

  // Does one full iteration of MCTS search:
  // 1. Initialize internal structures.
  // 2. Gather minibatch.
  // 3.
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

  // 4. Run NN computation.
  void RunNNComputation();

  // 5. Retrieve NN computations (and terminal values) into nodes.
  void FetchMinibatchResults();

  // 6. Propagate the new nodes' information to all their parents in the tree.
  void DoBackupUpdate();

  // 7. Update the Search's status and progress information.
  void UpdateCounters();

  // Interface for IterationMemoryAllocator support.
  IterationMemoryManager& GetIterationMemoryManager(int tid);
  size_t GetIterationAge() const;

  // Interface for tasks.
  template <bool starting_from_root = false>
  std::conditional_t<starting_from_root, std::tuple<int, int>, int>
  PickNodesToExtendTask(int collision_limit, int tid,
                        const EdgeAndNode& current_best_edge = {});
  void ProcessPickedTask(int start_idx, int end_idx);
  void CancelCollisionsTask(int start, int end, bool stop);
  int AddCollisions(int collisions);

  void StartTasks(int count);
  void CompleteTask();

  // Returns the search parameters.
  const SearchParams& GetParams() const { return params_; }

  // Return task workspace for the current thread.
  TaskWorkspace& GetWorkspace(int tid);
  // Return history at the root node.
  const PositionHistory& GetPlayedHistory() const {
    return search_->played_history_;
  }

  // Tasks cancel collisions.
  struct PickTaskCancelCollisions final : public TaskQueue::PickTask {
    using Base = TaskQueue::PickTask;
    PickTaskCancelCollisions(SearchWorker& worker);
    ~PickTaskCancelCollisions();
    void Reset(int start_idx, int end_idx, bool stop);
    void Wait(int tid) const override;

   private:
    SearchWorker& worker_;
    int start_idx_;
    int end_idx_;
    bool stop_;
    std::atomic<bool> completed_ = false;

    void DoTask(int) override;
  };

 private:
  // TODO: Is there false sharing issues when inserting to AtomicVector?
  struct CollisionNode {
    // The path to the node to extend.
    int multivisit = 0;
    // If greater than multivisit, and other parameters don't imply a lower
    // limit, multivist could be increased to this value without additional
    // change in outcome of next selection.
    int maxvisit = 0;

    std::string DebugString(const BackupPath& path) const {
      std::ostringstream oss;
      oss << "<CollisionNode> This:" << this << " Depth:" << path.size()
          << " Multivisit:" << multivisit << " Path:";
      for (auto it = path.cbegin(); it != path.cend(); ++it) {
        if (it != path.cbegin()) oss << "->";
        auto n = std::get<0>(*it);
        const auto& nl = n->GetLowNode();
        oss << n << ":" << n->GetNInFlight();
        if (nl) {
          oss << "(" << nl << ")";
        }
      }
      auto node = std::get<0>(path.back());
      oss << " --- " << node->DebugString();
      if (node->GetLowNode())
        oss << " --- " << node->GetLowNode()->DebugString();

      return oss.str();
    }

    CollisionNode() = default;

    CollisionNode(uint32_t multivisit, int maxvisit)
        : multivisit(multivisit), maxvisit(maxvisit) {}
  };
  struct NodeToProcess {
    bool IsExtendable(const BackupPath& path) const {
      auto node = std::get<0>(path.back());
      return !node->IsTerminal() && !node->GetLowNode();
    }
    bool CanEvalOutOfOrder(const BackupPath& path) const {
      auto node = std::get<0>(path.back());
      return is_tt_hit || is_cache_hit || node->IsTerminal() ||
             node->GetLowNode();
    }

    // The node to extend.
    int eval_index = -1;
    bool nn_queried = false;
    bool is_tt_hit = false;
    bool is_cache_hit = false;
    bool is_black_to_move = false;

    // Details that are filled in as we go.
    uint64_t hash;
    std::shared_ptr<LowNode> tt_low_node;

    std::string DebugString(const BackupPath& path) const {
      auto node = std::get<0>(path.back());
      std::ostringstream oss;
      oss << "<NodeToProcess> This:" << this << " Depth:" << path.size()
          << " Node:" << node << " NNQueried:" << nn_queried
          << " TTHit:" << is_tt_hit << " CacheHit:" << is_cache_hit << " Path:";
      for (auto it = path.cbegin(); it != path.cend(); ++it) {
        if (it != path.cbegin()) oss << "->";
        auto n = std::get<0>(*it);
        const auto& nl = n->GetLowNode();
        oss << n << ":" << n->GetNInFlight();
        if (nl) {
          oss << "(" << nl << ")";
        }
      }
      oss << " --- " << std::get<0>(path.back())->DebugString();
      if (node->GetLowNode())
        oss << " --- " << node->GetLowNode()->DebugString();

      return oss.str();
    }

    NodeToProcess& operator=(NodeToProcess&&) = default;
    NodeToProcess(NodeToProcess&&) = default;

    NodeToProcess(const PositionHistory& history)
        : is_black_to_move(history.IsBlackToMove()) {}
  };

  NodeToProcess PickNodeToExtend(int collision_limit);
  // Adjust parameters for updating node @n and its parent low node if node is
  // terminal or its child low node is a transposition. Also update bounds and
  // terminal status of node @n using information from its child low node.
  // Return true if adjustment happened.
  bool MaybeAdjustForTerminalOrTransposition(Node* n,
                                             const std::shared_ptr<LowNode>& nl,
                                             float& v, float& d, float& m,
                                             uint32_t& n_to_fix, float& v_delta,
                                             float& d_delta, float& m_delta,
                                             bool& update_parent_bounds) const;
  void DoBackupUpdateSingleNode(const NodeToProcess& node_to_process,
                                const BackupPath& path);
  // Returns whether a node's bounds were set based on its children.
  bool MaybeSetBounds(Node* p, float m, uint32_t* n_to_fix, float* v_delta,
                      float* d_delta, float* m_delta) const;
  std::tuple<int, int> PickNodesToExtend(int collision_limit);
  void ScheduleCancelTask(int start, int end, bool stop);
  int ExpandCollision(int idenx, int collisions_left);

  // Add visits or collisions to nodes
  int Collision(const BackupPath& path, int collision_count, int maxvisits);
  void Visit(const BackupPath& path, const PositionHistory& history);

  // Check if there is a reason to stop picking and pick @node.
  bool ShouldStopPickingHere(Node* node, bool is_root_node, int repetitions);
  void ExtendNode(NodeToProcess& picked_node, const BackupPath& path,
                  const PositionHistory& history);
  void FetchSingleNodeResult(NodeToProcess* node_to_process,
                             const BackupPath& path);
  // Process a queued task.
  void ProcessTask(int tid);
  void WaitForTasks();

  // Helpers to lookup picked node paths.
  const BackupPath& GetMinibatchPath(int index) const;
  const BackupPath& GetOutOfOrderPath(int index) const;
  const BackupPath& GetCollisionPath(int index) const;
  // Helpers to assign picked node paths.
  void AssignMinibatchPath(int index, const BackupPath& path);
  void AssignOutOfOrderPath(int index, const BackupPath& path);
  void AssignCollisionPath(int index, const BackupPath& path);

  Search* const search_;
  SearchWorkerCachedState& state_;
  int tid_ = -1;
  int target_minibatch_size_;
  int max_out_of_order_;
  int number_out_of_order_ = 0;
  size_t iteration_memory_age_ = 0;
  std::vector<IterationMemoryManager> iteration_memory_managers_;
  // List of nodes to process.
  alignas(kCacheLineSize) std::atomic<int> collisions_left_;
  alignas(kCacheLineSize) std::atomic<int> eval_used_;
  std::unique_ptr<BackendComputation> computation_;
  const SearchParams& params_;
  const bool moves_left_support_;
  classic::IterationStats iteration_stats_;
  classic::StoppersHints latest_time_manager_hints_;

  // Multigather task related fields.

  alignas(kCacheLineSize) std::atomic<int> outstanding_tasks_ = 0;
  PickTaskCancelCollisions cancel_task_;
  friend struct SearchWorkerCachedState;
};

// Holds per task worker scratch data
struct TaskWorkspace {
  std::array<Node::Iterator, SearchWorker::kMaxMovesInPosition> cur_iters;
  std::vector<CurrentPath> current_path;
  std::vector<std::unique_ptr<BackupPath>> fp_buffer;
  std::vector<std::unique_ptr<BackupPath>> full_path;
  std::vector<std::unique_ptr<PositionHistory>> h_buffer;
  std::vector<std::unique_ptr<PositionHistory>> history;

  int go_count_ = 0;
  int history_age_ = 0;

  TaskWorkspace() {
    // Reserve everything for a small number of recursions in a large tree.
    current_path.reserve(1024);
    full_path.reserve(8);
    fp_buffer.reserve(8);
    history.reserve(8);
    h_buffer.reserve(8);
  }

  [[nodiscard]]
  auto Push(std::span<const BackupPath::value_type> path,
            std::span<const Position> in_history,
            const PositionHistory& played_history) {
    assert(path.size() == in_history.size() + 1);
    if (go_count_ != history_age_) {
      assert(history.empty());
      auto played = played_history.GetPositions();
      for (auto& history_ptr : h_buffer) {
        history_ptr->Assign(played.begin(), played.end());
      }
      history_age_ = go_count_;
    }
    if (h_buffer.empty()) {
      int expected_size =
          std::bit_ceil(played_history.GetLength() + in_history.size() + 64);
      history.push_back(std::make_unique<PositionHistory>());
      history.back()->Reserve(expected_size);
      auto played = played_history.GetPositions();
      history.back()->Assign(played.begin(), played.end());
    } else {
      history.push_back(std::move(h_buffer.back()));
      h_buffer.pop_back();
    }
    assert(history.back()->GetLength() >= played_history.GetLength());
    history.back()->Trim(played_history.GetLength());
    history.back()->Insert(in_history.begin(), in_history.end());

    if (fp_buffer.empty()) {
      int expected_size = std::bit_ceil(path.size() + 32);
      full_path.push_back(std::make_unique<BackupPath>());
      full_path.back()->reserve(expected_size);
    } else {
      full_path.push_back(std::move(fp_buffer.back()));
      fp_buffer.pop_back();
    }
    full_path.back()->assign(path.begin(), path.end());
    return absl::Cleanup{[&] { Pop(); }};
  }

  void StartANewSearch() {
    go_count_++;
  }

 private:
  void Pop() {
    fp_buffer.push_back(std::move(full_path.back()));
    full_path.pop_back();
    h_buffer.push_back(std::move(history.back()));
    history.pop_back();
  }
};

// Cached worker state between subsequent searches.
struct SearchWorkerCachedState {
  // Make sure the cached state is ready for a new search.
  void StartANewSearch(const SearchParams& params, size_t target_minibatch_size,
                       size_t max_out_of_order);

  alignas(kCacheLineSize) AtomicVector<SearchWorker::NodeToProcess> minibatch_;
  alignas(kCacheLineSize) AtomicVector<SearchWorker::NodeToProcess> ooobatch_;
  alignas(kCacheLineSize) AtomicVector<SearchWorker::CollisionNode> collisions_;
  std::vector<EvalResult> eval_results_;
  std::vector<BackupPath> node_paths_;
};

// Cached state between subsequent searches.
struct SearchCachedState {
  void StartANewSearch(int task_workers, int search_workers);

  std::vector<TaskWorkspace> task_workspaces_;
  TaskQueue task_queue_;
  std::vector<SearchWorkerCachedState> worker_states_;
};

}  // namespace dag_classic
}  // namespace lczero
