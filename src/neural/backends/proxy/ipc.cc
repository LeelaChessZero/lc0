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

#include "neural/backends/proxy/ipc.h"

#include <utility>

#include "utils/exception.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <semaphore.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <ctime>
#include <thread>
#endif

namespace lczero {
namespace proxy {
namespace {

#ifdef _WIN32
std::string ObjectName(const std::string& name) { return "Local\\lc0-" + name; }
#else
// macOS limits these names to 31 characters, so keep server names short.
std::string ObjectName(const std::string& name) { return "/lc0-" + name; }
#endif

}  // namespace

SharedMemory SharedMemory::CreateOrAttach(const std::string& name,
                                          size_t size) {
  SharedMemory shm;
  const std::string object_name = ObjectName(name);
#ifdef _WIN32
  const uint64_t size64 = size;
  HANDLE handle = CreateFileMappingA(
      INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
      static_cast<DWORD>(size64 >> 32), static_cast<DWORD>(size64),
      object_name.c_str());
  if (!handle) throw Exception("CreateFileMapping failed: " + object_name);
  shm.handle_ = reinterpret_cast<intptr_t>(handle);
  shm.data_ = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, size);
#else
  const int fd = shm_open(object_name.c_str(), O_CREAT | O_RDWR, 0600);
  if (fd < 0) throw Exception("shm_open failed: " + object_name);
  shm.handle_ = fd;
  struct stat st;
  if (fstat(fd, &st) != 0 ||
      (static_cast<size_t>(st.st_size) < size &&
       ftruncate(fd, static_cast<off_t>(size)) != 0)) {
    throw Exception("Cannot size shared memory: " + object_name);
  }
  void* data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  shm.data_ = data == MAP_FAILED ? nullptr : data;
#endif
  if (!shm.data_) throw Exception("Cannot map shared memory: " + object_name);
  shm.size_ = size;
  return shm;
}

SharedMemory SharedMemory::Open(const std::string& name) {
  SharedMemory shm;
  const std::string object_name = ObjectName(name);
#ifdef _WIN32
  HANDLE handle =
      OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, object_name.c_str());
  if (!handle) throw Exception("No backend server named " + name);
  shm.handle_ = reinterpret_cast<intptr_t>(handle);
  shm.data_ = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
  MEMORY_BASIC_INFORMATION info;
  if (shm.data_ && VirtualQuery(shm.data_, &info, sizeof(info))) {
    shm.size_ = info.RegionSize;
  }
#else
  const int fd = shm_open(object_name.c_str(), O_RDWR, 0);
  if (fd < 0) throw Exception("No backend server named " + name);
  shm.handle_ = fd;
  struct stat st;
  if (fstat(fd, &st) == 0 && st.st_size > 0) {
    void* data = mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                      fd, 0);
    shm.data_ = data == MAP_FAILED ? nullptr : data;
    shm.size_ = st.st_size;
  }
#endif
  if (!shm.data_) throw Exception("Cannot map shared memory: " + object_name);
  return shm;
}

SharedMemory::SharedMemory(SharedMemory&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      handle_(std::exchange(other.handle_, -1)) {}

SharedMemory& SharedMemory::operator=(SharedMemory&& other) noexcept {
  if (this != &other) {
    Release();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    handle_ = std::exchange(other.handle_, -1);
  }
  return *this;
}

SharedMemory::~SharedMemory() { Release(); }

void SharedMemory::Release() {
#ifdef _WIN32
  if (data_) UnmapViewOfFile(data_);
  if (handle_ != -1) CloseHandle(reinterpret_cast<HANDLE>(handle_));
#else
  if (data_) munmap(data_, size_);
  if (handle_ != -1) close(static_cast<int>(handle_));
#endif
  data_ = nullptr;
  handle_ = -1;
}

NamedSemaphore NamedSemaphore::CreateOrAttach(const std::string& name) {
  NamedSemaphore sem;
  const std::string object_name = ObjectName(name);
#ifdef _WIN32
  sem.handle_ = CreateSemaphoreA(nullptr, 0, 0x7fffffff, object_name.c_str());
#else
  sem_t* handle = sem_open(object_name.c_str(), O_CREAT, 0600, 0);
  sem.handle_ = handle == SEM_FAILED ? nullptr : handle;
#endif
  if (!sem.handle_) throw Exception("Cannot create semaphore " + object_name);
  return sem;
}

NamedSemaphore NamedSemaphore::Open(const std::string& name) {
  NamedSemaphore sem;
  const std::string object_name = ObjectName(name);
#ifdef _WIN32
  sem.handle_ = OpenSemaphoreA(SEMAPHORE_MODIFY_STATE | SYNCHRONIZE, FALSE,
                               object_name.c_str());
#else
  sem_t* handle = sem_open(object_name.c_str(), 0);
  sem.handle_ = handle == SEM_FAILED ? nullptr : handle;
#endif
  if (!sem.handle_) throw Exception("Cannot open semaphore " + object_name);
  return sem;
}

NamedSemaphore::NamedSemaphore(NamedSemaphore&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)) {}

NamedSemaphore& NamedSemaphore::operator=(NamedSemaphore&& other) noexcept {
  if (this != &other) {
    Release();
    handle_ = std::exchange(other.handle_, nullptr);
  }
  return *this;
}

NamedSemaphore::~NamedSemaphore() { Release(); }

void NamedSemaphore::Release() {
  if (!handle_) return;
#ifdef _WIN32
  CloseHandle(handle_);
#else
  sem_close(static_cast<sem_t*>(handle_));
#endif
  handle_ = nullptr;
}

void NamedSemaphore::Post() {
#ifdef _WIN32
  ReleaseSemaphore(handle_, 1, nullptr);
#else
  sem_post(static_cast<sem_t*>(handle_));
#endif
}

bool NamedSemaphore::Wait(int timeout_ms) {
#ifdef _WIN32
  return WaitForSingleObject(handle_, timeout_ms < 0 ? INFINITE : timeout_ms) ==
         WAIT_OBJECT_0;
#else
  sem_t* sem = static_cast<sem_t*>(handle_);
  if (timeout_ms < 0) {
    while (sem_wait(sem) != 0) {
      if (errno != EINTR) return false;
    }
    return true;
  }
#ifdef __APPLE__
  // macOS has no sem_timedwait.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (sem_trywait(sem) != 0) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return true;
#else
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_sec += timeout_ms / 1000;
  ts.tv_nsec += static_cast<long>(timeout_ms % 1000) * 1000000;
  if (ts.tv_nsec >= 1000000000) {
    ts.tv_sec++;
    ts.tv_nsec -= 1000000000;
  }
  while (sem_timedwait(sem, &ts) != 0) {
    if (errno != EINTR) return false;
  }
  return true;
#endif
#endif
}

uint32_t CurrentProcessId() {
#ifdef _WIN32
  return GetCurrentProcessId();
#else
  return static_cast<uint32_t>(getpid());
#endif
}

bool IsProcessAlive(uint32_t pid) {
  if (pid == 0) return false;
#ifdef _WIN32
  HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, pid);
  if (!process) return false;
  const bool alive = WaitForSingleObject(process, 0) == WAIT_TIMEOUT;
  CloseHandle(process);
  return alive;
#else
  return kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM;
#endif
}

}  // namespace proxy
}  // namespace lczero
