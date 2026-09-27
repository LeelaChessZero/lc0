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

#include <chrono>
#include <thread>
#include <utility>

#include "utils/commandline.h"
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
#include <spawn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <climits>
#include <ctime>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

extern char** environ;
#endif

namespace lczero {
namespace proxy {
namespace {

#ifdef _WIN32
std::string ObjectName(const std::string& name) { return "Local\\lc0-" + name; }

// Appends `arg` quoted so that the child's C runtime splits it back unchanged.
void AppendQuoted(const std::string& arg, std::string* command_line) {
  if (!command_line->empty()) command_line->push_back(' ');
  if (!arg.empty() && arg.find_first_of(" \t\n\v\"") == std::string::npos) {
    command_line->append(arg);
    return;
  }
  command_line->push_back('"');
  size_t backslashes = 0;
  for (const char c : arg) {
    if (c == '\\') {
      ++backslashes;
      continue;
    }
    // Backslashes only escape when a quote follows.
    command_line->append(c == '"' ? backslashes * 2 + 1 : backslashes, '\\');
    command_line->push_back(c);
    backslashes = 0;
  }
  command_line->append(backslashes * 2, '\\');
  command_line->push_back('"');
}
#else
// macOS limits these names to 31 characters.
std::string ObjectName(const std::string& name) { return "/lc0-" + name; }
#endif

}  // namespace

SharedMemory SharedMemory::Create(const std::string& name, size_t size) {
  SharedMemory shm;
  const std::string object_name = ObjectName(name);
#ifdef _WIN32
  const uint64_t size64 = size;
  HANDLE handle =
      CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                         static_cast<DWORD>(size64 >> 32),
                         static_cast<DWORD>(size64), object_name.c_str());
  if (handle && GetLastError() == ERROR_ALREADY_EXISTS) {
    CloseHandle(handle);
    throw Exception("Shared memory already exists: " + object_name);
  }
  if (!handle) throw Exception("CreateFileMapping failed: " + object_name);
  shm.handle_ = reinterpret_cast<intptr_t>(handle);
  shm.data_ = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, size);
#else
  // Names carry the pid, so one that exists was left by a killed process.
  shm_unlink(object_name.c_str());
  const int fd = shm_open(object_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd < 0) throw Exception("shm_open failed: " + object_name);
  shm.handle_ = fd;
  shm.unlink_name_ = object_name;
  if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
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
  if (!handle) throw Exception("Cannot open shared memory " + object_name);
  shm.handle_ = reinterpret_cast<intptr_t>(handle);
  shm.data_ = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
  MEMORY_BASIC_INFORMATION info;
  if (shm.data_ && VirtualQuery(shm.data_, &info, sizeof(info))) {
    shm.size_ = info.RegionSize;
  }
#else
  const int fd = shm_open(object_name.c_str(), O_RDWR, 0);
  if (fd < 0) throw Exception("Cannot open shared memory " + object_name);
  shm.handle_ = fd;
  struct stat st;
  if (fstat(fd, &st) == 0 && st.st_size > 0) {
    void* data =
        mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
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
      handle_(std::exchange(other.handle_, -1)),
      unlink_name_(std::move(other.unlink_name_)) {
  other.unlink_name_.clear();
}

SharedMemory& SharedMemory::operator=(SharedMemory&& other) noexcept {
  if (this != &other) {
    Release();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    handle_ = std::exchange(other.handle_, -1);
    unlink_name_ = std::move(other.unlink_name_);
    other.unlink_name_.clear();
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
  if (!unlink_name_.empty()) shm_unlink(unlink_name_.c_str());
#endif
  data_ = nullptr;
  handle_ = -1;
  unlink_name_.clear();
}

NamedSemaphore NamedSemaphore::Create(const std::string& name) {
  NamedSemaphore sem;
  const std::string object_name = ObjectName(name);
#ifdef _WIN32
  sem.handle_ = CreateSemaphoreA(nullptr, 0, 0x7fffffff, object_name.c_str());
  if (sem.handle_ && GetLastError() == ERROR_ALREADY_EXISTS) {
    throw Exception("Semaphore already exists: " + object_name);
  }
#else
  sem_unlink(object_name.c_str());
  sem_t* handle = sem_open(object_name.c_str(), O_CREAT | O_EXCL, 0600, 0);
  sem.handle_ = handle == SEM_FAILED ? nullptr : handle;
  if (sem.handle_) sem.unlink_name_ = object_name;
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
    : handle_(std::exchange(other.handle_, nullptr)),
      unlink_name_(std::move(other.unlink_name_)) {
  other.unlink_name_.clear();
}

NamedSemaphore& NamedSemaphore::operator=(NamedSemaphore&& other) noexcept {
  if (this != &other) {
    Release();
    handle_ = std::exchange(other.handle_, nullptr);
    unlink_name_ = std::move(other.unlink_name_);
    other.unlink_name_.clear();
  }
  return *this;
}

NamedSemaphore::~NamedSemaphore() { Release(); }

void NamedSemaphore::Release() {
  if (handle_) {
#ifdef _WIN32
    CloseHandle(handle_);
#else
    sem_close(static_cast<sem_t*>(handle_));
    if (!unlink_name_.empty()) sem_unlink(unlink_name_.c_str());
#endif
  }
  handle_ = nullptr;
  unlink_name_.clear();
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
  const DWORD timeout =
      timeout_ms < 0 ? INFINITE : static_cast<DWORD>(timeout_ms);
  return WaitForSingleObject(handle_, timeout) == WAIT_OBJECT_0;
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

ChildProcess ChildProcess::Spawn(const std::vector<std::string>& args) {
  ChildProcess child;
#ifdef _WIN32
  std::string command_line;
  for (const std::string& arg : args) AppendQuoted(arg, &command_line);

  // Hand the child our stderr as its stdout and stderr, and nothing else.
  // Without one it writes to NUL, never to our stdout.
  HANDLE std_error = nullptr;
  const HANDLE own_std_error = GetStdHandle(STD_ERROR_HANDLE);
  if (own_std_error && own_std_error != INVALID_HANDLE_VALUE) {
    DuplicateHandle(GetCurrentProcess(), own_std_error, GetCurrentProcess(),
                    &std_error, 0, TRUE, DUPLICATE_SAME_ACCESS);
  }
  if (!std_error) {
    SECURITY_ATTRIBUTES inherit{sizeof(inherit), nullptr, TRUE};
    std_error =
        CreateFileA("NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                    &inherit, OPEN_EXISTING, 0, nullptr);
    if (std_error == INVALID_HANDLE_VALUE) std_error = nullptr;
  }
  STARTUPINFOEXA info{};
  info.StartupInfo.cb = sizeof(info);
  std::vector<char> attribute_buffer;
  if (std_error) {
    SIZE_T attribute_size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_size);
    attribute_buffer.resize(attribute_size);
    info.lpAttributeList =
        reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attribute_buffer.data());
    InitializeProcThreadAttributeList(info.lpAttributeList, 1, 0,
                                      &attribute_size);
    UpdateProcThreadAttribute(info.lpAttributeList, 0,
                              PROC_THREAD_ATTRIBUTE_HANDLE_LIST, &std_error,
                              sizeof(std_error), nullptr, nullptr);
    info.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    info.StartupInfo.hStdOutput = std_error;
    info.StartupInfo.hStdError = std_error;
  }
  // Without a console of our own, a console child would open a window.
  const DWORD flags = CREATE_SUSPENDED |
                      (std_error ? EXTENDED_STARTUPINFO_PRESENT : 0) |
                      (GetConsoleWindow() ? 0 : CREATE_NO_WINDOW);
  PROCESS_INFORMATION process_info{};
  const BOOL started =
      CreateProcessA(args[0].c_str(), command_line.data(), nullptr, nullptr,
                     std_error ? TRUE : FALSE, flags, nullptr, nullptr,
                     &info.StartupInfo, &process_info);
  if (info.lpAttributeList) DeleteProcThreadAttributeList(info.lpAttributeList);
  if (std_error) CloseHandle(std_error);
  if (!started) throw Exception("Cannot start " + args[0]);
  child.pid_ = process_info.dwProcessId;
  child.process_ = process_info.hProcess;

  // The job dies with its last handle, and takes the child with it.
  child.job_ = CreateJobObjectA(nullptr, nullptr);
  if (child.job_) {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags =
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(child.job_, JobObjectExtendedLimitInformation,
                            &limits, sizeof(limits));
    AssignProcessToJobObject(child.job_, child.process_);
  }
  ResumeThread(process_info.hThread);
  CloseHandle(process_info.hThread);
#else
  std::vector<char*> argv;
  for (const std::string& arg : args)
    argv.push_back(const_cast<char*>(arg.c_str()));
  argv.push_back(nullptr);
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null",
                                   O_RDONLY, 0);
  posix_spawn_file_actions_adddup2(&actions, STDERR_FILENO, STDOUT_FILENO);
  pid_t pid = 0;
  const int error =
      posix_spawn(&pid, argv[0], &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  if (error != 0) throw Exception("Cannot start " + args[0]);
  child.pid_ = static_cast<uint32_t>(pid);
#endif
  return child;
}

ChildProcess::ChildProcess(ChildProcess&& other) noexcept
    : pid_(std::exchange(other.pid_, 0)),
      process_(std::exchange(other.process_, nullptr)),
      job_(std::exchange(other.job_, nullptr)) {}

ChildProcess& ChildProcess::operator=(ChildProcess&& other) noexcept {
  if (this != &other) {
    Stop(0);
    pid_ = std::exchange(other.pid_, 0);
    process_ = std::exchange(other.process_, nullptr);
    job_ = std::exchange(other.job_, nullptr);
  }
  return *this;
}

ChildProcess::~ChildProcess() { Stop(0); }

bool ChildProcess::IsRunning() {
#ifdef _WIN32
  return process_ && WaitForSingleObject(process_, 0) == WAIT_TIMEOUT;
#else
  // An exited child stays a zombie until reaped, and kill(pid, 0) still
  // succeeds on zombies.
  if (pid_ == 0) return false;
  if (waitpid(static_cast<pid_t>(pid_), nullptr, WNOHANG) == 0) return true;
  pid_ = 0;
  return false;
#endif
}

void ChildProcess::Stop(int timeout_ms) {
#ifdef _WIN32
  if (process_) {
    if (WaitForSingleObject(process_, static_cast<DWORD>(timeout_ms)) !=
        WAIT_OBJECT_0) {
      TerminateProcess(process_, 1);
      WaitForSingleObject(process_, INFINITE);
    }
    CloseHandle(process_);
  }
  if (job_) CloseHandle(job_);
  process_ = nullptr;
  job_ = nullptr;
  pid_ = 0;
#else
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (IsRunning() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (IsRunning()) {
    kill(static_cast<pid_t>(pid_), SIGKILL);
    waitpid(static_cast<pid_t>(pid_), nullptr, 0);
    pid_ = 0;
  }
#endif
}

uint32_t CurrentProcessId() {
#ifdef _WIN32
  return GetCurrentProcessId();
#else
  return static_cast<uint32_t>(getpid());
#endif
}

std::string ExecutablePath() {
#if defined(_WIN32)
  char path[MAX_PATH];
  const DWORD length = GetModuleFileNameA(nullptr, path, MAX_PATH);
  if (length > 0 && length < MAX_PATH) return std::string(path, length);
#elif defined(__APPLE__)
  char path[PATH_MAX];
  uint32_t size = sizeof(path);
  if (_NSGetExecutablePath(path, &size) == 0) return path;
#elif defined(__linux__)
  char path[PATH_MAX];
  const ssize_t length = readlink("/proc/self/exe", path, sizeof(path));
  if (length > 0 && length < static_cast<ssize_t>(sizeof(path))) {
    return std::string(path, length);
  }
#endif
  return CommandLine::BinaryName();
}

void WaitForParentExit(uint32_t parent_pid) {
#ifdef _WIN32
  HANDLE parent = OpenProcess(SYNCHRONIZE, FALSE, parent_pid);
  if (!parent) return;
  WaitForSingleObject(parent, INFINITE);
  CloseHandle(parent);
#else
  // A child whose parent died is handed to another process.
  while (getppid() == static_cast<pid_t>(parent_pid)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
#endif
}

}  // namespace proxy
}  // namespace lczero
