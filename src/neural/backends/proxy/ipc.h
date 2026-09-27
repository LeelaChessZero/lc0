/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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

#include <cstddef>
#include <cstdint>
#include <string>

namespace lczero {
namespace proxy {

// Named shared memory, visible to other processes of the same user session.
class SharedMemory {
 public:
  // Attaches to an existing region of that name if there is one, so a
  // restarted backend process keeps the engines that are still mapped.
  static SharedMemory CreateOrAttach(const std::string& name, size_t size);
  static SharedMemory Open(const std::string& name);

  SharedMemory(SharedMemory&& other) noexcept;
  SharedMemory& operator=(SharedMemory&& other) noexcept;
  SharedMemory(const SharedMemory&) = delete;
  SharedMemory& operator=(const SharedMemory&) = delete;
  ~SharedMemory();

  void* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  SharedMemory() = default;
  void Release();

  void* data_ = nullptr;
  size_t size_ = 0;
  intptr_t handle_ = -1;
};

class NamedSemaphore {
 public:
  static NamedSemaphore CreateOrAttach(const std::string& name);
  static NamedSemaphore Open(const std::string& name);

  NamedSemaphore(NamedSemaphore&& other) noexcept;
  NamedSemaphore& operator=(NamedSemaphore&& other) noexcept;
  NamedSemaphore(const NamedSemaphore&) = delete;
  NamedSemaphore& operator=(const NamedSemaphore&) = delete;
  ~NamedSemaphore();

  void Post();
  // Negative timeout waits forever. Returns false on timeout.
  bool Wait(int timeout_ms);

 private:
  NamedSemaphore() = default;
  void Release();

  void* handle_ = nullptr;
};

uint32_t CurrentProcessId();
bool IsProcessAlive(uint32_t pid);

}  // namespace proxy
}  // namespace lczero
