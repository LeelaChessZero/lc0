/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020-2022 The LCZero Authors

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

std::map<uint32_t, Numa::Group> Numa::groups = {};
int Numa::thread_count = 0;
int Numa::core_count = 0;
int Numa::thread_groups = 0;

void Numa::Init() {
#if defined(_WIN64) && _WIN32_WINNT >= 0x0601
  int group_count = 0;
  thread_count = 0;
  core_count = 0;
  thread_groups = 0;

  SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* buffer;
  DWORD len = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
  buffer = static_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(malloc(len));
  GetLogicalProcessorInformationEx(RelationProcessorCore, buffer, &len);
  for (int offset = 0; offset < len;) {
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* info =
        (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)((char*)buffer + offset);
    int efficiency_class = info->Processor.EfficiencyClass;
    int group_id = info->Processor.GroupMask[0].Group;
    int threads = BitBoard(info->Processor.GroupMask[0].Mask).count();

    uint32_t key =
        ((255 - efficiency_class) << 24) | (threads << 16) | group_id;
    if (groups.find(key) == groups.end()) {
      Group tmp = {};
      groups.emplace(key, tmp);
      group_count++;
      if (threads > 1) thread_groups++;
    }
    thread_count += threads;
    core_count++;
    groups[key].efficiency_class = efficiency_class;
    groups[key].cores++;
    groups[key].threads += threads;
    groups[key].group_id = group_id;
    groups[key].mask |= info->Processor.GroupMask[0].Mask;
    offset += info->Size;
  }
  free(buffer);

  CERR << "Detected " << core_count << " core(s) and " << thread_count
       << " thread(s)";

  for (auto const & [ _, grp ] : groups) {
    CERR << "Group " << grp.group_id << " has " << grp.cores << " core(s) and "
         << grp.threads << " thread(s) in efficincy class "
         << grp.efficiency_class << ".";
  }
#endif
}

void Numa::BindThread(int id) {
#if defined(_WIN64) && _WIN32_WINNT >= 0x0601
  GROUP_AFFINITY affinity = {};
  // Allocate up to core_count threads in cores in order of efficiency, then
  // distribute evenly, up
  // to thread_count across all groups with threads > cores and the rest across
  // all groups.
  if (id < core_count) {
    int core_id = id;
    for (auto const & [ _, grp ] : groups) {
      if (core_id < grp.cores) {
        affinity.Group = grp.group_id;
        affinity.Mask = grp.mask;
        SetThreadGroupAffinity(GetCurrentThread(), &affinity, NULL);
        break;
      }
      core_id -= grp.cores;
    }
  } else {
    int idx = id - core_count;
    if (id < thread_count) {
      idx %= thread_groups;
    } else {
      idx %= groups.size();
    }
    int count = 0;
    for (auto const & [ _, grp ] : groups) {
      if (id < thread_count && grp.cores == grp.threads) continue;
      if (idx == count) {
        affinity.Group = grp.group_id;
        affinity.Mask = grp.mask;
        SetThreadGroupAffinity(GetCurrentThread(), &affinity, NULL);
        break;
      }
      count++;
    }
  }
#else
  // Silence warning.
  (void)id;
#endif
}

}  // namespace lczero
