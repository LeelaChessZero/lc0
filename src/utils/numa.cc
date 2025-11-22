/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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

#include "utils/numa.h"

#include "numa_config.h"
#if HAVE_LIBHWLOC
#include <hwloc.h>
#endif

#include <cassert>
#include <cstring>
#include <mutex>
#include <random>
#if __has_include(<source_location>)
#include <source_location>
#endif
#include <sstream>
#include <system_error>

#include "utils/logging.h"
#include "utils/optionsparser.h"
#ifdef _WIN32
// clang-format off
#include <windows.h>
#include <tlhelp32.h>
// clang-format on
#endif

namespace lczero {

namespace {

const OptionId kUseThreadAfinityOptionId{
    {.long_flag = "use-thread-affinity",
     .uci_option = "UseThreadAffinity",
     .help_text = "Pin search and task worker threads to specific CPU cores "
                  "to improve execution latency.",
     .visibility = OptionId::kAlwaysVisible}};
const OptionId kUseAllCoresOptionId{
    {.long_flag = "use-all-cores",
     .uci_option = "UseAllCores",
     .help_text = "Default is to use only core which were included in the "
                  "initial CPU affinity mask. This options allows search and "
                  "task workers to use cores also outside the initial set.",
     .visibility = OptionId::kProOnly}};
const OptionId kSearchNodeOptionId{
    {.long_flag = "search-numa-node",
     .uci_option = "SearchNUMANode",
     .help_text = "The NUMA node to use for the search threads.",
     .visibility = OptionId::kProOnly}};
const OptionId kShuffleCoreReservationOptionId{
    {.long_flag = "shuffle-core-reservation",
     .uci_option = "ShuffleCoreReservation",
     .help_text = "Randomize the core reservation order to reduce contention "
                  "when multiple lc0 instances are running.",
     .visibility = OptionId::kProOnly}};

#if HAVE_LIBHWLOC
template <typename T>
void ReportHWLocError(T result
#if __cpp_lib_source_location >= 201907L
                      ,
                      std::source_location loc = std::source_location::current()
#endif
) {
  if (result != 0) {
    std::ostringstream ss;
    ss << "HWLoc error "
#if __cpp_lib_source_location >= 201907L
          "at "
       << loc.file_name() << ":" << loc.line() << " - "
#endif
       << result << ": " << std::system_category().message(errno);
    CERR << ss.str();
    throw Exception(ss.str());
  }
}
template <typename T>
void ReportHWLocError(T* result
#if __cpp_lib_source_location >= 201907L
                      ,
                      std::source_location loc = std::source_location::current()
#endif
) {
  if (result == nullptr) {
    std::ostringstream ss;
    ss << "HWLoc error "
#if __cpp_lib_source_location >= 201907L
          "at "
       << loc.file_name() << ":" << loc.line() << " - "
#endif
       << std::system_category().message(errno);
    CERR << ss.str();
    throw Exception(ss.str());
  }
}

class CpuSet {
 public:
  CpuSet() : cpuset_(hwloc_bitmap_alloc()) { ReportHWLocError(cpuset_); }

  ~CpuSet() { hwloc_bitmap_free(cpuset_); }

  CpuSet(CpuSet&& other) : cpuset_(other.cpuset_) { other.cpuset_ = nullptr; }

  CpuSet& operator=(CpuSet&& other) {
    std::swap(cpuset_, other.cpuset_);
    return *this;
  }

  void Clear() { hwloc_bitmap_zero(cpuset_); }

  void Set(size_t cpu) { ReportHWLocError(hwloc_bitmap_set(cpuset_, cpu)); }
  void Unset(size_t cpu) { ReportHWLocError(hwloc_bitmap_clr(cpuset_, cpu)); }
  bool IsSet(size_t cpu) const { return hwloc_bitmap_isset(cpuset_, cpu); }

  bool operator==(const CpuSet& other) const {
    return hwloc_bitmap_isequal(cpuset_, other.cpuset_);
  }

  CpuSet& operator&=(const CpuSet& other) {
    ReportHWLocError(hwloc_bitmap_and(cpuset_, cpuset_, other.cpuset_));
    return *this;
  }
  CpuSet& operator|=(const CpuSet& other) {
    ReportHWLocError(hwloc_bitmap_or(cpuset_, cpuset_, other.cpuset_));
    return *this;
  }
  CpuSet& operator^=(const CpuSet& other) {
    ReportHWLocError(hwloc_bitmap_xor(cpuset_, cpuset_, other.cpuset_));
    return *this;
  }

  CpuSet& operator&=(const hwloc_bitmap_t other) {
    ReportHWLocError(hwloc_bitmap_and(cpuset_, cpuset_, other));
    return *this;
  }
  CpuSet& operator|=(const hwloc_bitmap_t other) {
    ReportHWLocError(hwloc_bitmap_or(cpuset_, cpuset_, other));
    return *this;
  }
  CpuSet& operator^=(const hwloc_bitmap_t other) {
    ReportHWLocError(hwloc_bitmap_xor(cpuset_, cpuset_, other));
    return *this;
  }

  CpuSet operator~() const {
    CpuSet result;
    ReportHWLocError(hwloc_bitmap_not(result.cpuset_, cpuset_));
    return result;
  }
  explicit operator bool() const { return !hwloc_bitmap_iszero(cpuset_); }

  size_t Count() const { return hwloc_bitmap_weight(cpuset_); }

  operator hwloc_cpuset_t() const { return cpuset_; }

 private:
  hwloc_cpuset_t cpuset_;
};

struct Config {
  // https://nuclear.llnl.gov/CNP/rng/rngman/node4.html
  using GeneratorType =
      std::linear_congruential_engine<uint64_t, 2862933555777941757, 3037000493,
                                      0>;
  Config() : rng_(std::random_device{}()) {
    ReportHWLocError(hwloc_topology_init(&initial_topology_));
    // TODO: Add filters to make discovery faster.
    // TODO: Maybe use xml loading instead of detection.
    ReportHWLocError(hwloc_topology_load(initial_topology_));
    ReportHWLocError(hwloc_topology_dup(&topology_, initial_topology_));
    effiency_.resize(hwloc_cpukinds_get_nr(topology_, 0));
    for (size_t i = 0; i < effiency_.size(); i++) {
      CpuSet cpuset;
      int eff;
      ReportHWLocError(hwloc_cpukinds_get_info(topology_, i, cpuset, &eff,
                                               nullptr, nullptr, 0));

      effiency_[eff < 0 ? i : eff] = std::move(cpuset);
    }

    if (IsAffinitySupported()) {
      use_search_thread_affinity_ = true;
      GetAffinity(initial_affinity_);
    } else {
      CERR << "HWLoc reports that thread CPU affinity is not supported on this "
              "system. Disabling thread affinity.";
      use_search_thread_affinity_ = false;
    }
  }

  bool IsAffinitySupported() const {
    const auto* support = hwloc_topology_get_support(topology_);
    return support->cpubind->get_thread_cpubind;
  }

  ~Config() {
    if (initial_topology_) hwloc_topology_destroy(initial_topology_);
    if (topology_) hwloc_topology_destroy(topology_);
  }

  void SetAffinity(const CpuSet& cpuset_) {
    assert(topology_);
    assert(cpuset_);
    ReportHWLocError(
        hwloc_set_cpubind(topology_, cpuset_, HWLOC_CPUBIND_THREAD));
  }

  void GetAffinity(CpuSet& cpuset_) {
    assert(topology_);
    ReportHWLocError(
        hwloc_get_cpubind(topology_, cpuset_, HWLOC_CPUBIND_THREAD));
  }

  void SetAffintyAll(const CpuSet& cpuset_) {
    assert(topology_);
    assert(cpuset_);
#if _WIN32
    HANDLE snapshot =
        CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, GetCurrentProcessId());
    if (snapshot == INVALID_HANDLE_VALUE) {
      CERR << "Failed to create thread snapshot. Error code: "
           << std::system_category().message(GetLastError());
      SetAffinity(cpuset_);
      return;
    }
    THREADENTRY32 te = {};
    te.dwSize = sizeof(THREADENTRY32);
    if (!Thread32First(snapshot, &te)) {
      CERR << "Failed to get first thread from snapshot. Error code: "
           << std::system_category().message(GetLastError());
      CloseHandle(snapshot);
      SetAffinity(cpuset_);
      return;
    }
    do {
      if (te.dwSize >=
          FIELD_OFFSET(THREADENTRY32, th32ThreadID) + sizeof(te.th32ThreadID)) {
        HANDLE thread =
            OpenThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION, FALSE,
                       te.th32ThreadID);
        if (!thread) {
          CERR << "Failed to open thread " << te.th32ThreadID
               << ". Error code: "
               << std::system_category().message(GetLastError());
        } else {
          CERR << "Setting affinity for thread " << te.th32ThreadID << ".";
          ReportHWLocError(hwloc_set_thread_cpubind(topology_, thread, cpuset_,
                                                    HWLOC_CPUBIND_THREAD));
          CloseHandle(thread);
        }
      }
      te.dwSize = sizeof(THREADENTRY32);
    } while (Thread32Next(snapshot, &te));
    CloseHandle(snapshot);
#else
    assert(topology_);
    ReportHWLocError(
        hwloc_set_cpubind(topology_, cpuset_, HWLOC_CPUBIND_PROCESS));
#endif
  }

  void ProcessProcessors() {
    hwloc_topology_destroy(topology_);

    ReportHWLocError(hwloc_topology_dup(&topology_, initial_topology_));
    if (!use_all_cores_) {
      ReportHWLocError(hwloc_topology_restrict(
          topology_, initial_affinity_, HWLOC_RESTRICT_FLAG_REMOVE_CPULESS));
    }
  }

  void GetSocketSet(size_t socket_id, CpuSet& cpuset) const {
    assert(socket_id < GetSocketCount());
    hwloc_obj_t socket_obj;
    ReportHWLocError(socket_obj = hwloc_get_obj_by_type(
                         topology_, HWLOC_OBJ_PACKAGE, socket_id));
    cpuset |= socket_obj->cpuset;
    cpuset &= ~reserved_set_;
  }

  struct ObjectIterator {
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = hwloc_obj_t;
    using difference_type = std::ptrdiff_t;
    using pointer = hwloc_obj_t;
    using reference = hwloc_obj_t&;

    const hwloc_topology_t topology_;
    hwloc_obj_t current_;

    ObjectIterator& operator++() {
      current_ = current_->next_cousin;
      return *this;
    }
    ObjectIterator operator++(int) {
      ObjectIterator temp = *this;
      return ++temp;
    }
    ObjectIterator& operator--() {
      current_ = current_->prev_cousin;
      return *this;
    }
    ObjectIterator operator--(int) {
      ObjectIterator temp = *this;
      return --temp;
    }
    auto operator<=>(const ObjectIterator& other) const {
      return current_->logical_index <=> other.current_->logical_index;
    }
    bool operator==(const ObjectIterator& other) const {
      return current_ == other.current_;
    }

    hwloc_obj_t operator*() const { return current_; }
    hwloc_obj_t operator->() const { return current_; }
  };

  struct ObjectRange {
    const hwloc_topology_t topology_;
    hwloc_obj_t parent_;
    hwloc_obj_type_t type_;

    ObjectIterator begin() const {
      hwloc_obj_t parent = parent_;
      while (parent && parent->type != type_) {
        parent = parent->first_child;
      }
      return ObjectIterator{topology_, parent};
    }
    ObjectIterator end() const {
      hwloc_obj_t parent = parent_;
      while (parent && parent->type != type_) {
        parent = parent->last_child;
      }
      return ObjectIterator{topology_, parent->next_cousin};
    }
  };

  void ReserveCoresOnNode(size_t node_id, size_t count) {
    hwloc_obj_t numa_obj;
    reserved_set_.Clear();
    reserved_cores_.clear();

    ReportHWLocError(numa_obj = hwloc_get_obj_by_type(
                         topology_, HWLOC_OBJ_PACKAGE, node_id));
    ReserveCores(numa_obj, count);
  }

  void ReserveCores(size_t count) {
    hwloc_obj_t root_obj = hwloc_get_root_obj(topology_);
    reserved_set_.Clear();
    reserved_cores_.clear();

    ReserveCores(root_obj, count);
  }

  void ReserveCores(hwloc_obj_t parent, size_t count) {
    ObjectRange cores{topology_, parent, HWLOC_OBJ_CORE};
    std::vector<hwloc_obj_t> core_objs;
    std::copy(cores.begin(), cores.end(), std::back_inserter(core_objs));
    // Use random shuffle to avoid using same cores for all SearchWorkers when
    // multiple lc0 process are runnig at the same time.
    if (shuffle_reservations_) {
      GeneratorType rng_copy = rng_;
      std::shuffle(core_objs.begin(), core_objs.end(), rng_copy);
      rng_ = rng_copy;
    }
    std::stable_sort(
        core_objs.begin(), core_objs.end(),
        [this](hwloc_obj_t a, hwloc_obj_t b) {
          if (effiency_.size() > 1) {
            for (const auto& eff_set : effiency_) {
              bool a_in_eff = hwloc_bitmap_intersects(a->cpuset, eff_set);
              bool b_in_eff = hwloc_bitmap_intersects(b->cpuset, eff_set);
              if (a_in_eff != b_in_eff) {
                return a_in_eff;  // prefer less efficient cores
              }
              if (a_in_eff) break;
            }
          }
          return a->arity > b->arity;
        });
    unsigned arity = 2;
    // duplicate cores until all threads have PU or there is no more PUs
    while (core_objs.size() < count && core_objs.front()->arity >= arity) {
      CERR << core_objs.size() << " cores reserved, need " << count
           << ", adding cores with arity >= " << arity << ".";
      std::copy_if(core_objs.begin(), core_objs.end(),
                   std::back_inserter(core_objs),
                   [arity](hwloc_obj_t obj) { return obj->arity >= arity; });
      arity++;
    }
    if (core_objs.size() > count) {
      CERR << "Only " << count
           << " cores requested, trimming reserved cores from "
           << core_objs.size() << ".";
      core_objs.erase(core_objs.begin() + count, core_objs.end());
    }
    std::for_each(core_objs.begin(), core_objs.end(),
                  [&](hwloc_obj_t obj) { reserved_set_ |= obj->cpuset; });
    reserved_cores_ = std::move(core_objs);
  }

  void GetCoreForThread(size_t id, CpuSet& cpuset) {
    id %= reserved_cores_.size();

    hwloc_obj_t core = reserved_cores_[id];

    cpuset |= core->cpuset;
  }

  bool CheckReservedCores(size_t num_workers) {
    return reserved_cores_.size() == num_workers;
  }

  bool CheckReservedCores(size_t node_id, size_t num_workers) {
    if (reserved_cores_.size() != num_workers) return false;
    hwloc_obj_t socket_obj;
    ReportHWLocError(socket_obj = hwloc_get_obj_by_type(
                         topology_, HWLOC_OBJ_NUMANODE, node_id));
    for (auto core : reserved_cores_) {
      if (!hwloc_bitmap_intersects(socket_obj->cpuset, core->cpuset)) {
        return false;
      }
    }
    return true;
  }

  class ConfigLock {
   public:
    explicit ConfigLock(Config& config)
        : config_(config), lock_(config.mutex_) {}
    ConfigLock(const ConfigLock&) = delete;
    ConfigLock& operator=(const ConfigLock&) = delete;
    ~ConfigLock() {}
    Config* operator->() { return &config_; }

   private:
    Config& config_;
    std::unique_lock<std::mutex> lock_;
  };

  void SetOptions(const OptionsDict& options) { options_ = &options; }

  void UpdateOptions() {
    bool all_cores = options_->Get<bool>(kUseAllCoresOptionId);
    bool shuffle_reservations =
        options_->Get<bool>(kShuffleCoreReservationOptionId);
    use_search_thread_affinity_ =
        options_->Get<bool>(kUseThreadAfinityOptionId) && IsAffinitySupported();
    size_t node_id = options_->Get<int>(kSearchNodeOptionId);
    if (all_cores != use_all_cores_) {
      use_all_cores_ = all_cores;
      ProcessProcessors();
    }
    if (shuffle_reservations != shuffle_reservations_) {
      shuffle_reservations_ = shuffle_reservations;
      reserved_set_.Clear();
      reserved_cores_.clear();
    }
    if (node_id >= GetNodeCount()) {
      CERR << "Requested search NUMA node " << node_id << " but only "
           << GetNodeCount() << " nodes available. "
           << "Using node " << GetNodeCount() - 1 << " instead." << std::endl;
      node_id = GetNodeCount() - 1;
    }
    search_node_id_ = node_id;
  }

  static ConfigLock Lock() { return ConfigLock(Instance()); }

  static Config& Instance() {
    static Config instance{};
    return instance;
  }

  size_t GetThreadCount() const {
    return hwloc_get_nbobjs_by_type(topology_, HWLOC_OBJ_PU);
  }
  size_t GetMaxThread() const {
    return hwloc_get_obj_by_type(topology_, HWLOC_OBJ_PU, GetThreadCount() - 1)
        ->logical_index;
  }
  size_t GetCoreCount() const {
    return hwloc_get_nbobjs_by_type(topology_, HWLOC_OBJ_CORE);
  }
  size_t GetSocketCount() const {
    return hwloc_get_nbobjs_by_type(topology_, HWLOC_OBJ_PACKAGE);
  }
  size_t GetNodeCount() const {
    return hwloc_get_nbobjs_by_type(topology_, HWLOC_OBJ_NUMANODE);
  }

  CpuSet initial_affinity_;
  CpuSet reserved_set_;
  std::vector<hwloc_obj_t> reserved_cores_;
  const OptionsDict* options_ = nullptr;
  bool use_search_thread_affinity_ = true;
  bool use_all_cores_ = false;
  bool shuffle_reservations_ = true;
  size_t search_node_id_ = 0;
  GeneratorType rng_;

  hwloc_topology_t topology_ = nullptr;
  hwloc_topology_t initial_topology_;
  std::vector<CpuSet> effiency_;
  mutable std::mutex mutex_;
};
#endif

}  // namespace

void Numa::Init(OptionsParser* parser) {
  parser->Add<BoolOption>(kUseThreadAfinityOptionId) = true;
  parser->Add<BoolOption>(kUseAllCoresOptionId) = false;
  parser->Add<BoolOption>(kShuffleCoreReservationOptionId) = true;
  parser->Add<IntOption>(kSearchNodeOptionId, 0, 512) = 0;
#if HAVE_LIBHWLOC
  auto config = Config::Lock();
  config->SetOptions(parser->GetOptionsDict());
  size_t threads = config->GetThreadCount();
  size_t cores = config->GetCoreCount();
  size_t sockets = config->GetSocketCount();
  size_t nodes = config->GetNodeCount();
  CERR << "Detected " << cores << (cores == 1 ? " core and " : " cores and ")
       << threads << (threads == 1 ? " thread in " : " threads in ") << sockets
       << (sockets == 1 ? " socket in " : " sockets in ") << nodes
       << (nodes == 1 ? " node." : " nodes.") << " The highest thread id is "
       << config->GetMaxThread() << ".";
#else
  CERR << "libhwloc support not compiled in.";
#endif
}

void Numa::BindThread([[maybe_unused]] size_t id) {
#if HAVE_LIBHWLOC
  auto config = Config::Lock();
  if (!config->use_search_thread_affinity_) return;
  if (!config->reserved_set_) return;
  CpuSet cpuset;
  config->GetCoreForThread(id, cpuset);
  config->SetAffinity(cpuset);
#endif
}

void Numa::ReserveSearchWorkers([[maybe_unused]] size_t num_workers,
                                bool runs_on_cpu) {
#if HAVE_LIBHWLOC
  auto config = Config::Lock();
  config->UpdateOptions();
  if (!config->use_search_thread_affinity_) return;
  unsigned node_id = config->search_node_id_;
  if (runs_on_cpu) {
    if (config->CheckReservedCores(node_id, num_workers)) {
      return;
    }
    config->ReserveCoresOnNode(node_id, num_workers);
  } else {
    if (config->CheckReservedCores(num_workers)) {
      return;
    }
    config->ReserveCores(num_workers);
  }

  CpuSet cpuset;
  cpuset |= config->initial_affinity_;
  cpuset &= ~config->reserved_set_;

  config->SetAffintyAll(!cpuset ? config->initial_affinity_ : cpuset);
#endif
}

void Numa::BindTaskWorkersToSocket() {
#if HAVE_LIBHWLOC
  auto config = Config::Lock();
  if (!config->use_search_thread_affinity_) return;
  CpuSet cpuset;
  config->GetSocketSet(config->search_node_id_, cpuset);
  config->SetAffinity(cpuset);
#endif
}
}  // namespace lczero
