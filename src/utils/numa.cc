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

#include <cassert>
#include <cstring>
#include <mutex>

#include "numa_config.h"
#include "utils/optionsparser.h"
#if HAVE_PTHREAD_SETAFFINITY_NP
#include <pthread.h>
#include <unistd.h>
#endif

#include <system_error>

#include "utils/logging.h"
#include "utils/numa.h"

#ifdef _WIN32
// clang-format off
#include <windows.h>
#include <tlhelp32.h>
// clang-format on

#include <bit>

#include "chess/bitboard.h"
#include "utils/bititer.h"
#endif

namespace lczero {

namespace {

const OptionId kUseAllCoresOptionId{
    {.long_flag = "use-all-cores",
     .uci_option = "UseAllCores",
     .help_text = "Default is to use only core which were included in the "
                  "initial CPU affinity mask. This options allows search and "
                  "task workers to use cores also outside the initial set.",
     .visibility = OptionId::kProOnly}};
const OptionId kSearchSocketOptionId{
    {.long_flag = "search-numa-socket",
     .uci_option = "SearchNUMASocket",
     .help_text = "The NUMA socket to use for the search threads.",
     .visibility = OptionId::kProOnly}};

#ifdef _WIN32
static bool CheckIsWin11() {
  OSVERSIONINFOA osvi = {};
  osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
  GetVersionExA(&osvi);
  return osvi.dwMajorVersion >= 11 ||
         (osvi.dwMajorVersion == 10 && osvi.dwBuildNumber >= 20348);
}

static bool IsWin11() {
  static bool is_win11 = CheckIsWin11();
  return is_win11;
}

static constexpr unsigned kMaxProcessorsPerGroup = 64;

#endif

auto DivUp = [](size_t a, size_t b) { return (a + b - 1) / b; };

struct LogicalProcessor {
  unsigned cpu_;
  unsigned core_;
  unsigned socket_;
  unsigned node_;
};

#if HAVE_PTHREAD_SETAFFINITY_NP

#define USE_THREAD_AFINITTY 1

class LSCPUInit {
 public:
  LSCPUInit() : pipe_(popen("lscpu -p", "r")) {}

  ~LSCPUInit() { Close(); }

  void Close() {
    if (!pipe_) return;
    pclose(pipe_);
    pipe_ = nullptr;
  }

  std::vector<LogicalProcessor> ReadConfig() {
    if (!pipe_) {
      CERR << "Failed to run lscpu. Assumming single socket system with one "
              "thread per core. Install lscpu to enable NUMA support.";
      return {};
    }
    std::vector<LogicalProcessor> configs;
    char buffer[128];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe_) != NULL) {
      if (buffer[0] == '#') continue;

      LogicalProcessor config;

      if (sscanf(buffer, "%u,%u,%u,%u", &config.cpu_, &config.core_,
                 &config.socket_, &config.node_) != 4) {
        CERR << "Failed to parse lscpu output line: " << buffer;
        continue;
      }

      configs.push_back(config);
    }

    Close();
    return configs;
  }

 private:
  FILE* pipe_;
} numa_config;

class CpuSet {
 public:
  CpuSet() : cpuset_{nullptr} {}

  CpuSet(unsigned max_lp) : cpuset_(CPU_ALLOC(max_lp + 1)), size_(max_lp + 1) {
    CPU_ZERO_S(Bytes(), cpuset_);
  }
  ~CpuSet() { CPU_FREE(cpuset_); }

  CpuSet& operator=(CpuSet&& other) {
    std::swap(cpuset_, other.cpuset_);
    std::swap(size_, other.size_);
    return *this;
  }

  void Set(size_t cpu) {
    assert(cpu < size_);
    CPU_SET_S(cpu, Bytes(), cpuset_);
  }
  void Unset(size_t cpu) {
    assert(cpu < size_);
    CPU_CLR_S(cpu, Bytes(), cpuset_);
  }
  bool IsSet(size_t cpu) const {
    assert(cpu < size_);
    return CPU_ISSET(cpu, cpuset_);
  }

  bool operator==(const CpuSet& other) const {
    assert(size_ == other.size_);
    return CPU_EQUAL_S(Bytes(), cpuset_, other.cpuset_);
  }

  CpuSet& operator&=(const CpuSet& other) {
    assert(size_ == other.size_);
    CPU_AND_S(Bytes(), cpuset_, cpuset_, other.cpuset_);
    return *this;
  }
  CpuSet& operator|=(const CpuSet& other) {
    assert(size_ == other.size_);
    CPU_OR_S(Bytes(), cpuset_, cpuset_, other.cpuset_);
    return *this;
  }
  CpuSet& operator^=(const CpuSet& other) {
    assert(size_ == other.size_);
    CPU_XOR_S(Bytes(), cpuset_, cpuset_, other.cpuset_);
    return *this;
  }

  size_t Count() const { return CPU_COUNT_S(Bytes(), cpuset_); }

  size_t Bytes() const { return CPU_ALLOC_SIZE(size_); }

  void SetAffinity() {
    int err;
    auto thread = pthread_self();
    if ((err = pthread_setaffinity_np(thread, Bytes(), cpuset_))) {
      CERR << "Failed to set thread affinity. Error code: "
           << std::system_category().message(err);
    }
  }

  void GetAffinity() {
    int err;
    auto thread = pthread_self();
    if ((err = pthread_getaffinity_np(thread, Bytes(), cpuset_))) {
      CERR << "Failed to get thread affinity(" << thread << ", " << Bytes()
           << ", " << cpuset_
           << "). Error code: " << std::system_category().message(err);
    }
  }

  void SetAffintyAll() {
    std::ostringstream ss;
    const size_t elements = DivUp(size_, sizeof(unsigned long) * 8);
    unsigned long bitset[elements];
    std::memset(bitset, 0, elements * sizeof(unsigned long));

    for (unsigned i = 0; i < elements; i++) {
      auto& element = bitset[i];
      for (unsigned bit = 0; bit < sizeof(element) * 8; bit++) {
        size_t cpu = i * sizeof(element) * 8 + bit;
        if (cpu >= size_) break;
        if (IsSet(cpu)) {
          element |= 1UL << bit;
        }
      }
    }
    int err;
    static bool have_pgrep = !std::system("which pgrep > /dev/null");
    static bool have_taskset = !std::system("which taskset > /dev/null");

    if (!have_pgrep || !have_taskset) {
      std::once_flag warn_flag;
      std::call_once(warn_flag, []() {
        CERR << (have_pgrep ? "" : "pgrep")
             << (!have_pgrep && !have_taskset ? " or " : "")
             << (have_taskset ? "" : "taskset")
             << " command not found. Cannot set affinity for all threads.";
      });
      SetAffinity();
      return;
    }

    ss << "pgrep -wg " << getpid() << " | xargs -n1 taskset -p " << std::hex;

    auto end = std::reverse_iterator(bitset);
    auto start = std::find_if(std::reverse_iterator(bitset + elements), end,
                              [](const auto& a) { return a != 0; });
    if (start != end) {
      ss << *start++ << std::setfill('0');
    }
    std::for_each(start, end,
                  [&](const auto& a) { ss << std::setw(sizeof(a) * 2) << a; });

    ss << " > /dev/null";

    if ((err = std::system(ss.str().c_str()))) {
      CERR << "Failed to run command:" << ss.str();
      CERR << "Error code: " << std::system_category().message(err);
    }
  }

 private:
  cpu_set_t* cpuset_;
  unsigned size_;
};

#elif defined(_WIN64) && _WIN32_WINNT >= 0x0601

#define USE_THREAD_AFINITTY 1

std::vector<LogicalProcessor> WindowsReadConfig() {
  std::vector<LogicalProcessor> logical_processors;

  DWORD len = 0;

  GetLogicalProcessorInformationEx(RelationAll, NULL, &len);

  auto buffer = std::make_unique<char[]>(len);
  auto* info =
      reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(&buffer[0]);
  if (!GetLogicalProcessorInformationEx(RelationAll, info, &len)) {
    CERR << "GetLogicalProcessorInformationEx failed to get data: "
         << std::system_category().message(GetLastError());
    return {};
  }

  [[maybe_unused]]
  bool support_numa_groups = IsWin11();

  unsigned offset = 0;
  unsigned core_id = 0;
  unsigned socket_id = 0;
  auto get_or_insert_lp = [&](unsigned cpu_index) -> LogicalProcessor& {
    auto lp_iter =
        std::find_if(logical_processors.begin(), logical_processors.end(),
                     [&](const auto& lp) { return lp.cpu_ == cpu_index; });
    if (lp_iter != logical_processors.end()) {
      return *lp_iter;
    }
    logical_processors.push_back({cpu_index, 0, 0, 0});
    return logical_processors.back();
  };
  while (offset < len) {
    info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
        &buffer[offset]);
    switch (info->Relationship) {
      case RelationProcessorCore: {
        auto& core_info = info->Processor;
        unsigned this_core = core_id++;
        for (DWORD i = 0; i < core_info.GroupCount; i++) {
          uint64_t mask = core_info.GroupMask[i].Mask;
          for (auto bit : IterateBits(mask)) {
            unsigned cpu_index = bit + i * kMaxProcessorsPerGroup;
            auto& lp = get_or_insert_lp(cpu_index);
            lp.core_ = this_core;
          }
        }
        break;
      }
      case RelationProcessorPackage: {
        auto& package_info = info->Processor;
        unsigned this_socket = socket_id++;
        for (DWORD i = 0; i < package_info.GroupCount; i++) {
          uint64_t mask = package_info.GroupMask[i].Mask;
          for (auto bit : IterateBits(mask)) {
            unsigned cpu_index = bit + i * kMaxProcessorsPerGroup;
            auto& lp = get_or_insert_lp(cpu_index);
            lp.socket_ = this_socket;
          }
        }
        break;
      }
      case RelationNumaNode: {
        auto& node_info = info->NumaNode;
        unsigned this_node = info->NumaNode.NodeNumber;
#if HAVE_WINDOWS_NUMA_NODE_GROUP
        // Windows 11 build 20348 and later returns multiple group masks for
        // NUMA nodes.
        WORD count = support_numa_groups ? node_info.GroupCount : 1;
        for (DWORD i = 0; i < count; i++) {
          uint64_t mask = node_info.GroupMasks[i].Mask;
#else
        {
          DWORD i = 0;
          uint64_t mask = node_info.GroupMask.Mask;
#endif
          for (auto bit : IterateBits(mask)) {
            unsigned cpu_index = bit + i * kMaxProcessorsPerGroup;
            auto& lp = get_or_insert_lp(cpu_index);
            lp.node_ = this_node;
          }
        }
        break;
      }
      default:
        break;
    }
    offset += info->Size;
  }

  std::sort(logical_processors.begin(), logical_processors.end(),
            [](const auto& a, const auto& b) {
              if (a.node_ != b.node_) return a.node_ < b.node_;
              if (a.socket_ != b.socket_) return a.socket_ < b.socket_;
              if (a.core_ != b.core_) return a.core_ < b.core_;
              return a.cpu_ < b.cpu_;
            });

  unsigned current_socket_id = -1;
  unsigned first_core_id = -1;
  for (auto& lp : logical_processors) {
    if (lp.socket_ != current_socket_id) {
      current_socket_id = lp.socket_;
      first_core_id = lp.core_;
    }
    lp.core_ -= first_core_id;
  }

  return {logical_processors};
}

class CpuSet {
 public:
  CpuSet() {}

  CpuSet(unsigned max_lp)
      : group_affinities_(std::make_unique<GROUP_AFFINITY[]>(
            DivUp(max_lp + 1, kMaxProcessorsPerGroup))),
        size_(max_lp + 1) {
    for (unsigned i = 0; i < DivUp(size_, kMaxProcessorsPerGroup); i++) {
      group_affinities_[i].Group = i;
      group_affinities_[i].Mask = 0;
    }
  }

  CpuSet& operator=(CpuSet&& other) {
    std::swap(group_affinities_, other.group_affinities_);
    std::swap(size_, other.size_);
    return *this;
  }

  void Set(size_t cpu) {
    assert(cpu < size_);
    size_t idx = cpu / kMaxProcessorsPerGroup;
    size_t bit = cpu % kMaxProcessorsPerGroup;
    group_affinities_[idx].Mask |= 1ULL << bit;
  }
  void Unset(size_t cpu) {
    assert(cpu < size_);
    size_t idx = cpu / kMaxProcessorsPerGroup;
    size_t bit = cpu % kMaxProcessorsPerGroup;
    group_affinities_[idx].Mask &= ~(1ULL << bit);
  }
  bool IsSet(size_t cpu) const {
    assert(cpu < size_);
    size_t idx = cpu / kMaxProcessorsPerGroup;
    size_t bit = cpu % kMaxProcessorsPerGroup;
    return (group_affinities_[idx].Mask & (1ULL << bit)) != 0;
  }

  bool operator==(const CpuSet& other) const {
    assert(size_ == other.size_);
    for (unsigned i = 0; i < DivUp(size_, kMaxProcessorsPerGroup); i++) {
      if (group_affinities_[i].Mask != other.group_affinities_[i].Mask) {
        return false;
      }
    }
    return true;
  }

  CpuSet& operator&=(const CpuSet& other) {
    assert(size_ == other.size_);
    for (unsigned i = 0; i < DivUp(size_, kMaxProcessorsPerGroup); i++) {
      group_affinities_[i].Mask &= other.group_affinities_[i].Mask;
    }
    return *this;
  }
  CpuSet& operator|=(const CpuSet& other) {
    assert(size_ == other.size_);
    for (unsigned i = 0; i < DivUp(size_, kMaxProcessorsPerGroup); i++) {
      group_affinities_[i].Mask |= other.group_affinities_[i].Mask;
    }
    return *this;
  }
  CpuSet& operator^=(const CpuSet& other) {
    assert(size_ == other.size_);
    for (unsigned i = 0; i < DivUp(size_, kMaxProcessorsPerGroup); i++) {
      group_affinities_[i].Mask ^= other.group_affinities_[i].Mask;
    }
    return *this;
  }

  size_t Count() const {
    size_t count = 0;
    for (unsigned i = 0; i < DivUp(size_, kMaxProcessorsPerGroup); i++) {
      count += std::popcount(group_affinities_[i].Mask);
    }
    return count;
  }

  void SetAffinity(HANDLE thread = GetCurrentThread()) {
    // Windows 11 must use SetThreadSelectedCpuSetMasks to avoid pinning threads
    // to only a processor group.
    // see details: https://stackoverflow.com/questions/76317127
    if (IsWin11()) {
      if (!SetThreadSelectedCpuSetMasks(thread, group_affinities_.get(),
                                        DivUp(size_, kMaxProcessorsPerGroup))) {
        CERR << "Failed to set thread affinity. Error code: "
             << std::system_category().message(GetLastError());
      }
      return;
    } else {
      // Fallback implementation for Windows versions prior to 11.
      // We select the first available group affinity.
      for (unsigned i = 0; i < DivUp(size_, kMaxProcessorsPerGroup); i++) {
        if (group_affinities_[i].Mask != 0) {
          if (!SetThreadGroupAffinity(thread, &group_affinities_[i], nullptr)) {
            CERR << "Failed to set thread group affinity. Error code: "
                 << std::system_category().message(GetLastError());
            break;
          }
        }
      }
      return;
    }
  }

  void GetAffinity() {
    if (IsWin11()) {
      HANDLE thread = GetCurrentThread();
      USHORT return_length = 0;
      if (!GetThreadSelectedCpuSetMasks(thread, group_affinities_.get(),
                                        DivUp(size_, kMaxProcessorsPerGroup),
                                        &return_length)) {
        CERR << "Failed to get thread affinity. Error code: "
             << std::system_category().message(GetLastError());
      }
      return;
    } else {
      HANDLE thread = GetCurrentThread();
      GROUP_AFFINITY current_affinity;
      if (!GetThreadGroupAffinity(thread, &current_affinity)) {
        CERR << "Failed to get thread group affinity. Error code: "
             << std::system_category().message(GetLastError());
        return;
      }
      group_affinities_[current_affinity.Group].Mask = current_affinity.Mask;
      return;
    }
  }

  void SetAffintyAll() {
    HANDLE snapshot =
        CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, GetCurrentProcessId());
    if (snapshot == INVALID_HANDLE_VALUE) {
      CERR << "Failed to create thread snapshot. Error code: "
           << std::system_category().message(GetLastError());
      SetAffinity();
      return;
    }
    THREADENTRY32 te = {};
    te.dwSize = sizeof(THREADENTRY32);
    if (!Thread32First(snapshot, &te)) {
      CERR << "Failed to get first thread from snapshot. Error code: "
           << std::system_category().message(GetLastError());
      CloseHandle(snapshot);
      SetAffinity();
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
          SetAffinity(thread);
          CloseHandle(thread);
        }
      }
      te.dwSize = sizeof(THREADENTRY32);
    } while (Thread32Next(snapshot, &te));
    CloseHandle(snapshot);
  }

 private:
  std::unique_ptr<GROUP_AFFINITY[]> group_affinities_;
  unsigned size_;
};

#endif

struct Config {
  static std::vector<LogicalProcessor> FallbackProcessorConfig() {
    std::vector<LogicalProcessor> logical_processors;
    for (unsigned cpu = 0; cpu < std::thread::hardware_concurrency(); cpu++) {
      logical_processors.push_back({cpu, cpu, 0, 0});
    }
    return logical_processors;
  }

  Config(const std::vector<LogicalProcessor>& logical_processors)
      : logical_processors_(logical_processors) {
    if (logical_processors_.empty()) {
      logical_processors_ = FallbackProcessorConfig();
    }
    ProcessProcessors();
  }

  void ProcessProcessors() {
    Sort();
    if (!use_all_cores_) {
      logical_processors_.erase(
          std::remove_if(logical_processors_.begin(), logical_processors_.end(),
                         [&](const LogicalProcessor& lp) {
                           if (!initial_affinity_.IsSet(lp.cpu_)) {
                             restricted_processors_.push_back(lp);
                             return true;
                           }
                           return false;
                         }),
          logical_processors_.end());
    }
    size_t nodes = 0;
    size_t current_node = -1;
    size_t sockets = 0;
    size_t current_socket = -1;
    size_t cores = 0;
    size_t current_core = -1;
    size_t max_thread = 0;
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
      max_thread = std::max(max_thread, static_cast<size_t>(lp.cpu_));
    }
    nodes_ = nodes;
    sockets_ = sockets;
    cores_ = cores + current_core + 1;
    max_thread_ = max_thread;
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

  unsigned GetCoreForThread(int id) {
#ifdef _WIN32
    if (!IsWin11() && GetMaxThread() >= kMaxProcessorsPerGroup) {
      // Distribute threads to different processor groups on Windows before 11.
      id *= DivUp(logical_processors_.size(),
                  DivUp(GetMaxThread(), kMaxProcessorsPerGroup));
    }
#endif
    id %= logical_processors_.size();
    return logical_processors_[id].cpu_;
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
    Sort();
    return false;
  }

  void Sort() {
    std::sort(logical_processors_.begin(), logical_processors_.end(),
              [](const auto& a, const auto& b) { return a.cpu_ < b.cpu_; });
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

  void SetOptions(const OptionsDict& options) {
    options_ = &options;
    initial_affinity_ = CpuSet(GetMaxThread());
    initial_affinity_.GetAffinity();
  }

  void UpdateOptions() {
    bool all_cores = options_->Get<bool>(kUseAllCoresOptionId);
    search_socket_id_ = options_->Get<int>(kSearchSocketOptionId);
    if (all_cores != use_all_cores_) {
      use_all_cores_ = all_cores;
      std::copy(restricted_processors_.begin(), restricted_processors_.end(),
                std::back_inserter(logical_processors_));
      restricted_processors_.clear();
      ProcessProcessors();
    }
    if (std::none_of(logical_processors_.begin(),
                    logical_processors_.end(),
                    [&](const LogicalProcessor& lp) {
                      return lp.socket_ == search_socket_id_;
                    })) {
      CERR << "Requested search to use socket " << search_socket_id_
           << " is out of range. Only " << GetSocketCount()
           << " socket(s) available. Using socket " << logical_processors_.front().socket_
           << " instead.";
      search_socket_id_ = logical_processors_.front().socket_;
    }
  }

  static ConfigLock Lock() { return ConfigLock(Instance()); }

  static Config& Instance() {
#if HAVE_PTHREAD_SETAFFINITY_NP
    static Config instance{numa_config.ReadConfig()};
#elif defined(_WIN32) && _WIN32_WINNT >= 0x0601
    static Config instance{WindowsReadConfig()};
#else
    static Config instance{FallbackProcessorConfig()};
#endif
    return instance;
  }

  size_t GetThreadCount() const { return logical_processors_.size(); }
  size_t GetMaxThread() const { return max_thread_; }
  size_t GetCoreCount() const { return cores_; }
  size_t GetSocketCount() const { return sockets_; }
  size_t GetNodeCount() const { return nodes_; }

  size_t max_thread_ = 0;
  size_t cores_ = 0;
  size_t sockets_ = 0;
  size_t nodes_ = 0;

  CpuSet initial_affinity_;
  const OptionsDict* options_ = nullptr;
  bool use_all_cores_ = true;
  size_t search_socket_id_ = 0;

  std::vector<LogicalProcessor> logical_processors_;
  std::vector<LogicalProcessor> reserved_processors_;
  std::vector<LogicalProcessor> restricted_processors_;
  mutable std::mutex mutex_;
};

}  // namespace

void Numa::Init(OptionsParser* parser) {
  parser->Add<BoolOption>(kUseAllCoresOptionId) = false;
  parser->Add<IntOption>(kSearchSocketOptionId, 0, 512) = 0;
  auto config = Config::Lock();
  config->SetOptions(parser->GetOptionsDict());
#if USE_THREAD_AFINITTY
  size_t threads = config->GetThreadCount();
  size_t cores = config->GetCoreCount();
  size_t nodes = config->GetNodeCount();
  CERR << "Detected " << cores << (cores == 1 ? " core and " : " cores and ")
       << threads << (threads == 1 ? " thread in " : " threads in ") << nodes
       << (nodes == 1 ? " node." : " nodes.") << " The highest thread id is "
       << config->GetMaxThread() << ".";
#endif
}

void Numa::BindThread([[maybe_unused]] int id) {
#if USE_THREAD_AFINITTY
  auto config = Config::Lock();
  auto core = config->GetCoreForThread(id);
  CpuSet cpuset(config->GetMaxThread());
  config->ForEachCore(core, [&](unsigned cpu_id) { cpuset.Set(cpu_id); });
  cpuset.SetAffinity();
#endif
}

void Numa::ReserveSearchWorkers(size_t num_workers) {
  if (num_workers == 0) return;
#if USE_THREAD_AFINITTY
  auto config = Config::Lock();
  config->UpdateOptions();
  unsigned socket_id = config->search_socket_id_;
  if (config->CheckReservedCores(socket_id, num_workers)) {
    return;
  }
  for (size_t i = 0; i < num_workers; i++) {
    config->ReserveCoreOnSocket(socket_id);
  }

  std::sort(config->reserved_processors_.begin(),
            config->reserved_processors_.end(),
            [](const auto& a, const auto& b) { return a.core_ < b.core_; });
  CpuSet cpuset(config->GetMaxThread()), emptyset(config->GetMaxThread());
  cpuset.GetAffinity();
  for (const auto& rp : config->reserved_processors_) {
    cpuset.Unset(rp.cpu_);
  }
  if (cpuset == emptyset) {
    return;
  }

  cpuset.SetAffintyAll();
#endif
}

void Numa::BindSearchWorker([[maybe_unused]] size_t id) {
#if USE_THREAD_AFINITTY
  auto config = Config::Lock();
  if (config->reserved_processors_.empty()) return;
  size_t core = -1;
  for (const auto& processor : config->reserved_processors_) {
    if (processor.core_ != core) {
      core = processor.core_;
      if (id-- == 0) break;
    }
  }
  CpuSet cpuset(config->GetMaxThread());
  for (const auto& processor : config->reserved_processors_) {
    if (processor.core_ == core) {
      cpuset.Set(processor.cpu_);
    }
  }
  cpuset.SetAffinity();
#endif
}

void Numa::BindTaskWorkersToSocket() {
#if USE_THREAD_AFINITTY
  auto config = Config::Lock();
  CpuSet cpuset(config->GetMaxThread());
  config->ForEachOnSocket(config->search_socket_id_,
                          [&](unsigned cpu_id) { cpuset.Set(cpu_id); });
  cpuset.SetAffinity();
#endif
}
}  // namespace lczero
