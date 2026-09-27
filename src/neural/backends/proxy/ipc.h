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
#include <vector>

namespace lczero {
namespace proxy {

// Named shared memory, visible to other processes of the same user session.
class SharedMemory {
 public:
  // Creates a zero-filled region. On POSIX the creator removes the name again
  // when it is destroyed.
  static SharedMemory Create(const std::string& name, size_t size);
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
  std::string unlink_name_;
};

class NamedSemaphore {
 public:
  // On POSIX the creator removes the name again when it is destroyed.
  static NamedSemaphore Create(const std::string& name);
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
  std::string unlink_name_;
};

// A child process. It is killed when this object is destroyed, and on Windows
// also when this process dies.
class ChildProcess {
 public:
  ChildProcess() = default;
  // args[0] is the executable. The child's stdout goes to this process's
  // stderr, as stdout carries UCI, and it gets no stdin.
  static ChildProcess Spawn(const std::vector<std::string>& args);

  ChildProcess(ChildProcess&& other) noexcept;
  ChildProcess& operator=(ChildProcess&& other) noexcept;
  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;
  ~ChildProcess();

  uint32_t pid() const { return pid_; }
  // Not thread safe: on POSIX it reaps the child once it has exited.
  bool IsRunning();
  // Waits up to timeout_ms for the child to exit, then kills it.
  void Stop(int timeout_ms);

 private:
  uint32_t pid_ = 0;
  void* process_ = nullptr;
  void* job_ = nullptr;
};

uint32_t CurrentProcessId();
// Path of the running executable, to start another copy of it.
std::string ExecutablePath();
// Blocks until the parent process, `parent_pid`, exits.
void WaitForParentExit(uint32_t parent_pid);

}  // namespace proxy
}  // namespace lczero
