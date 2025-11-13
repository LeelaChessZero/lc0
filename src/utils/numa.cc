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

#include "numa_config.h"
#include "utils/numa.h"

#include <pthread.h>
#include <string.h>
#if HAVE_PTHREAD_SETAFFINITY_NP
#include <unistd.h>
#endif

#include "chess/bitboard.h"
#include "utils/logging.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace lczero {

namespace {

struct Config {
  struct LogicalProcessor {
    unsigned cpu_;
    unsigned core_;
    unsigned socket_;
    unsigned node_;
  };

  Config() = default;

  Config(std::vector<LogicalProcessor>& logical_processors)
      : logical_processors_(logical_processors) {
    sort();
    size_t nodes = 0;
    size_t current_node = -1;
    size_t sockets = 0;
    size_t current_socket = -1;
    size_t cores = 0;
    size_t current_core = -1;
    for (const auto& lp : logical_processors_) {
      if (current_node != lp.node_) {
        current_node = lp.node_;
        nodes++;
      }
      if (current_socket != lp.socket_) {
        current_socket = lp.socket_;
        sockets++;
        cores += current_core + 1;
        current_core = 0;
      }
      current_core = std::max(current_core, static_cast<size_t>(lp.core_));
    }
    nodes_ = nodes;
    sockets_ = sockets;
    cores_ = cores + current_core + 1;
  }

  template <typename Func>
  void ForEachOnSocket(size_t socket_id, Func func) const {
    assert(socket_id < GetSocketCount());
    for (const auto& lp : logical_processors_) {
      if (lp.socket_ == socket_id) {
        func(lp.cpu_);
      }
    }
  }

  template <typename Func>
  void ForEachCore(size_t logical_id, Func func) const {
    assert(logical_id < GetThreadCount());
    const auto& lp = logical_processors_[logical_id];
    for (const auto& other_lp : logical_processors_) {
      if (other_lp.core_ == lp.core_ && other_lp.socket_ == lp.socket_ &&
          other_lp.node_ == lp.node_) {
        func(other_lp.cpu_);
      }
    }
  }

  void ReserveCoreOnSocket(size_t socket_id) {
    assert(socket_id < GetSocketCount());
    bool socket_found = false;
    size_t core_id;
    for (const auto& lp : logical_processors_) {
      if (socket_found && lp.socket_ == socket_id && lp.core_ == core_id) {
        reserved_processors_.push_back(lp);
      }
      if (!socket_found && lp.socket_ == socket_id) {
        core_id = lp.core_;
        reserved_processors_.push_back(lp);
        socket_found = true;
      }
    }
    logical_processors_.erase(
        std::remove_if(logical_processors_.begin(), logical_processors_.end(),
                       [&](const LogicalProcessor& lp) {
                         return lp.socket_ == socket_id && lp.core_ == core_id;
                       }),
        logical_processors_.end());
  }

  bool CheckReservedCores(size_t socket_id, size_t num_workers) {
    size_t reserved_cores = 0;
    size_t core = -1;
    for (const auto& lp : reserved_processors_) {
      if (lp.socket_ != socket_id) {
        break;
      }
      if (lp.core_ != core) {
        core = lp.core_;
        reserved_cores++;
      }
    }
    if (reserved_cores == num_workers) return true;

    std::copy(reserved_processors_.begin(), reserved_processors_.end(),
              std::back_inserter(logical_processors_));
    reserved_processors_.clear();
    sort();
    return false;
  }

  void sort() {
    std::sort(logical_processors_.begin(), logical_processors_.end(),
              [](const auto& a, const auto& b) { return a.cpu_ < b.cpu_; });
  }

  size_t GetThreadCount() const { return logical_processors_.size(); }
  size_t GetCoreCount() const { return cores_; }
  size_t GetSocketCount() const { return sockets_; }
  size_t GetNodeCount() const { return nodes_; }

  size_t cores_ = 0;
  size_t sockets_ = 0;
  size_t nodes_ = 0;
  std::vector<LogicalProcessor> logical_processors_;
  std::vector<LogicalProcessor> reserved_processors_;
};

#if HAVE_PTHREAD_SETAFFINITY_NP

#define USE_THREAD_AFINITTY 1

class LSCPUInit {
 public:
  LSCPUInit() : pipe_(popen("lscpu -p", "r")) {
    if (!pipe_) {
      CERR << "Failed to run lscpu. Assumming single socket system with one "
              "thread per core. Install lscpu to enable NUMA support.";
      std::vector<Config::LogicalProcessor> logical_processors;
      for (unsigned cpu = 0; cpu < std::thread::hardware_concurrency(); cpu++) {
        logical_processors.push_back({cpu, cpu, 0, 0});
      }
      configs_ = Config(logical_processors);
      return;
    }
  }

  ~LSCPUInit() { Close(); }

  void Close() {
    if (!pipe_) return;
    pclose(pipe_);
    pipe_ = nullptr;
  }

  Config& ReadConfig() {
    if (!pipe_) return configs_;
    std::vector<Config::LogicalProcessor> configs;
    char buffer[128];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe_) != NULL) {
      if (buffer[0] == '#') continue;

      Config::LogicalProcessor config;

      sscanf(buffer, "%u,%u,%u,%u", &config.cpu_, &config.core_,
             &config.socket_, &config.node_);

      configs.push_back(config);
    }

    Close();
    configs_ = Config(configs);
    return configs_;
  }

 private:
  FILE* pipe_;
  Config configs_;
} numa_config;

#endif
}  // namespace

int Numa::threads_per_core_ = 1;

void Numa::Init() {
#if defined(_WIN64) && _WIN32_WINNT >= 0x0601
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* buffer;
  DWORD len = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
  buffer = static_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(malloc(len));
  GetLogicalProcessorInformationEx(RelationProcessorCore, buffer, &len);
  if (buffer->Processor.Flags & LTP_PC_SMT) {
    threads_per_core_ = BitBoard(buffer->Processor.GroupMask[0].Mask).count();
  }
  free(buffer);

  int group_count = GetActiveProcessorGroupCount();
  int thread_count = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  int core_count = thread_count / threads_per_core_;
  CERR << "Detected " << core_count << " core(s) and " << thread_count
       << " thread(s) in " << group_count << " group(s).";
  for (int group_id = 0; group_id < group_count; group_id++) {
    int group_threads = GetActiveProcessorCount(group_id);
    int group_cores = group_threads / threads_per_core_;
    CERR << "Group " << group_id << " has " << group_cores << " core(s) and "
         << group_threads << " thread(s).";
  }
#elif HAVE_PTHREAD_SETAFFINITY_NP
  const auto& config = numa_config.ReadConfig();
  size_t threads = config.GetThreadCount();
  size_t cores = config.GetCoreCount();
  size_t nodes = config.GetNodeCount();
  CERR << "Detected " << cores << (cores == 1 ? " core and " : " cores and ")
       << threads << (threads == 1 ? " thread in " : " threads in ") << nodes
       << (nodes == 1 ? " node." : " nodes.");
#endif
}

void Numa::BindThread([[maybe_unused]] int id) {
#if defined(_WIN64) && _WIN32_WINNT >= 0x0601
  int group_count = GetActiveProcessorGroupCount();
  int thread_count = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  int core_count = thread_count / threads_per_core_;
  int core_id = id;
  GROUP_AFFINITY affinity = {};
  for (int group_id = 0; group_id < group_count; group_id++) {
    int group_threads = GetActiveProcessorCount(group_id);
    int group_cores = group_threads / threads_per_core_;
    // Allocate cores of each group in order, and distribute remaining threads
    // to all groups.
    if ((id < core_count && core_id < group_cores) ||
        (id >= core_count && (id - core_count) % group_count == group_id)) {
      affinity.Group = group_id;
      affinity.Mask = ~0ULL >> (64 - group_threads);
      SetThreadGroupAffinity(GetCurrentThread(), &affinity, NULL);
      break;
    }
    core_id -= group_cores;
  }
#elif HAVE_PTHREAD_SETAFFINITY_NP
  const auto& config = numa_config.ReadConfig();
  id %= config.GetThreadCount();
  auto thread = pthread_self();
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  config.ForEachCore(id, [&](unsigned cpu_id) { CPU_SET(cpu_id, &cpuset); });
  int err;
  if ((err = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset))) {
    CERR << "Failed to set thread affinity. Error code: " << err;
  }
#endif
}

void Numa::ReserveSearchWorkers(size_t socket_id, size_t num_workers) {
  if (num_workers == 0) return;
#if HAVE_PTHREAD_SETAFFINITY_NP
  auto& config = numa_config.ReadConfig();
  if (socket_id >= config.GetSocketCount()) {
    CERR << "Requested to reserve workers on invalid socket " << socket_id
         << ". Only " << config.GetSocketCount()
         << " socket(s) available. Using socket "
         << (config.GetSocketCount() - 1) << " instead.";
    socket_id = config.GetSocketCount() - 1;
  }
  if (config.CheckReservedCores(socket_id, num_workers)) {
    return;
  }
  for (size_t i = 0; i < num_workers; i++) {
    config.ReserveCoreOnSocket(socket_id);
  }

  std::sort(
      config.reserved_processors_.begin(), config.reserved_processors_.end(),
      [](const Config::LogicalProcessor& a, const Config::LogicalProcessor& b) {
        return a.core_ < b.core_;
      });
  cpu_set_t cpuset, emptyset;
  auto thread = pthread_self();
  int err;
  if ((err = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset))) {
    CERR << "Failed to get thread affinity. Error code: " << err;
    return;
  }
  for (const auto& rp : config.reserved_processors_) {
    CPU_CLR(rp.cpu_, &cpuset);
  }
  if ((err = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset))) {
    CERR << "Failed to remove reserved affinity. Error code: " << err;
  }
  CPU_ZERO(&emptyset);
  if (CPU_EQUAL(&cpuset, &emptyset)) {
    return;
  }
  std::ostringstream ss;
  std::array<unsigned long, sizeof(cpuset) / sizeof(unsigned long)> bitset;
  memcpy((void*)bitset.data(), &cpuset, sizeof(cpuset));

  ss << "pgrep -wg " << getpid() << " | xargs -n1 taskset -p " << std::hex;

  auto start = std::find_if(bitset.rbegin(), bitset.rend(),
                            [](const auto& a) { return a != 0; });
  if (start != bitset.rend()) {
    ss << *start++ << std::setfill('0');
  }
  std::for_each(start, bitset.rend(),
                [&](const auto& a) { ss << std::setw(sizeof(a) * 2) << a; });

  ss << " > /dev/null";

  if ((err = std::system(ss.str().c_str()))) {
    CERR << "Failed to run command:" << ss.str();
    CERR << "Error code: " << err;
  }
#endif
}

void Numa::BindSearchWorker(size_t id) {
#if HAVE_PTHREAD_SETAFFINITY_NP
  auto& config = numa_config.ReadConfig();
  if (config.reserved_processors_.empty()) return;
  size_t core = -1;
  for (const auto& processor : config.reserved_processors_) {
    if (processor.core_ != core) {
      core = processor.core_;
      if (id-- == 0) break;
    }
  }
  cpu_set_t cpuset;
  auto thread = pthread_self();
  int err;
  CPU_ZERO(&cpuset);
  for (const auto& processor : config.reserved_processors_) {
    if (processor.core_ == core) {
      CPU_SET(processor.cpu_, &cpuset);
    }
  }
  if ((err = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset))) {
    CERR << "Failed to set search worker affinity. Error code: " << err;
  }
#endif
}

void Numa::BindTaskWorkersToSocket(size_t socket_id) {
#if HAVE_PTHREAD_SETAFFINITY_NP
  auto& config = numa_config.ReadConfig();
  socket_id = std::min(socket_id, config.GetSocketCount() - 1);
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  config.ForEachOnSocket(socket_id,
                         [&](unsigned cpu_id) { CPU_SET(cpu_id, &cpuset); });
  auto thread = pthread_self();
  int err;
  if ((err = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset))) {
    CERR << "Failed to set task workers affinity. Error code: " << err;
  }
#endif
}
}  // namespace lczero
