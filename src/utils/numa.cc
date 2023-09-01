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

#include "chess/bitboard.h"
#include "utils/logging.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace lczero {

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
    CERR << "Group " << group_id << " has " << group_cores
         << " core(s) and " << group_threads << " thread(s).";
  }
#endif
}

void Numa::BindThread(int id) {
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
#else
  // Silence warning.
  (void)id;
#endif
}

}  // namespace lczero
