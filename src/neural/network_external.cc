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

#include "neural/factory.h"
#include "neural/network_legacy.h"
#include "utils/bititer.h"
#include "utils/optionsdict.h"
#include "utils/transpose.h"

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace lczero {

namespace {
#ifdef _WIN32
typedef HANDLE map_t;
#else
typedef size_t map_t;
#endif

void* make_map(std::string file_name, unsigned long long needed_size, map_t *maybeHandle)
{
  void* base_address;
#ifndef _WIN32
  // TODO: create file with specific size, not open readonly.
  int fd = ::open(file_name.c_str(), O_RDONLY);
  if (fd == -1) return nullptr;
  base_address = mmap(nullptr, needed_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  *maybeHandle = needed_size;
  ::close(fd);
  if (base_address == MAP_FAILED) {
    exit(1);
  }
#else
  const HANDLE fd =
      CreateFileA(file_name.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_WRITE, nullptr,
                  CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, nullptr);
  LARGE_INTEGER file_size;
  file_size.QuadPart = needed_size;
  SetFilePointerEx(fd, file_size, nullptr, 0);
  SetEndOfFile(fd);
  if (fd == INVALID_HANDLE_VALUE) return nullptr;
  HANDLE mmap =
      CreateFileMapping(fd, nullptr, PAGE_READWRITE, needed_size >> 32,
                                  needed_size & 0xFFFFFFFFu, nullptr);
  *maybeHandle = mmap;
  CloseHandle(fd);
  if (!mmap) {
    exit(1);
  }
  base_address = MapViewOfFile(mmap, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
  if (!base_address) {
    exit(1);
  }
#endif
  return base_address;
}

class ExternalNetworkComputation;

class ExternalNetwork : public Network {
 public:
  ExternalNetwork(const WeightsFile& file, const OptionsDict& options) {
    // Serialize file to bytes.
    auto data = file.SerializeAsString();
    // Make large memory mapped file big enough to contain plus some extra and
    // also at least max batch size times size of inputs, wdl and policies.
    auto mmap_name = options.GetOrDefault<std::string>(
        "mmap_file", "external_net_transport");
    map_ = make_map(
        mmap_name,
        std::max(data.size() + 24, static_cast<size_t>(16) + 1024 * sizeof(float) * (112 * 8 * 8 + 1858 + 3)),
		&map_handle_);
    source_flag_ = new (map_) std::atomic<size_t>();
    dest_flag_ = new (static_cast<char*>(map_) + 8) std::atomic<size_t>();
    length_ = reinterpret_cast<size_t*>(static_cast<char*>(map_) + 16);
    inputs_ = reinterpret_cast<float*>(static_cast<char*>(map_) + 24);
    policies_ = reinterpret_cast<float*>(static_cast<char*>(map_) + 24 +
                                         1024 * sizeof(float) * (112 * 8 * 8));
    wdls_ = reinterpret_cast<float*>(static_cast<char*>(map_) + 24 +
                                         1024 * sizeof(float) * (112 * 8 * 8 + 1858));
    // TODO: There is a race if external sees file and maps it before we clear
    // this flag and the 'undefined value' happens to be read as a 1.
    *source_flag_ = 0;
    *length_ = data.size();
    // write weights bytes at small offset.
    memcpy(static_cast<char*>(map_) + 24, data.data(), data.size());
    // Write 'weights ready' flag.
    *source_flag_ = 1;
    // Spin Wait for 'dest ready' flag.
    while (*dest_flag_ != 1) {
      // Sleep a ms or two should be fine for this loop.  TODO:
    }
  }

  ~ExternalNetwork()
  {
#ifndef _WIN32
    munmap(map_, map_handle_);
#else
    UnmapViewOfFile(map_);
    CloseHandle(map_handle_);
#endif
    // TODO: delete the file.
  }

  std::unique_ptr<NetworkComputation> NewComputation() override;

  void Compute(const std::vector<InputPlanes>& raw_input,
               std::vector<std::vector<float>>* wdls,
               std::vector<std::vector<float>>* policies) const {
    // Take lock.
    std::lock_guard<std::mutex> lock(lock_);
    // Write raw_input at small offset with length.
    *length_ = raw_input.size();
    int flat_index = 0;
    for (auto inputs : raw_input) {
      for (auto input : inputs) {
        for (int i = 0; i < 64; i++) {
          if ((1ULL << i) & input.mask) {
            inputs_[flat_index] = input.value; 
          } else {
            inputs_[flat_index] = 0.0f;
          }
          flat_index++;
        }
      }
    }
    // Write 'input ready' flag value.
    *source_flag_ = 2;
    // Otherside clears input ready flag.
    // Spin Wait for 'output ready' flag value.
    while (*dest_flag_ != 2) {
      // TODO: no sleep for first n loops, then sleep 0.
    }
    // Clear 'output ready' flag. (Maybe atomic_compare_swap in spin wait?)
    *dest_flag_ = 0;
    // Copy output in wdls/policies.
    for (int i = 0; i < raw_input.size(); i++) {
      std::vector<float> policy(0.0f, 1858);
      for (int j = 0; j < 1858; j++) {
        policy[j] = policies_[i * 1858 + j];
      }
      policies->emplace_back(policy);
    }
    for (int i = 0; i < raw_input.size(); i++) {
      std::vector<float> wdl(0.0f, 3);
      for (int j = 0; j < 3; j++) {
        wdl[j] = wdls_[i * 3 + j];
      }
      wdls->emplace_back(wdl);
    }
  }

  const NetworkCapabilities& GetCapabilities() const override {
    // TODO: use same capabilities as weights file implies.
    static NetworkCapabilities capabilities;
    return capabilities;
  }

 private:
  void* map_;
  map_t map_handle_;
  std::atomic<size_t>* source_flag_;
  std::atomic<size_t>* dest_flag_;
  size_t* length_;
  float* inputs_;
  float* policies_;
  float* wdls_;
  mutable std::mutex lock_;
};

class ExternalNetworkComputation : public NetworkComputation {
 public:
  ExternalNetworkComputation(const ExternalNetwork* network)
      : network_(network) {}
  void AddInput(InputPlanes&& input) override {
    raw_input_.emplace_back(input);
  }
  void ComputeBlocking() override {
    network_->Compute(raw_input_, &wdls_, &policies_);
  }

  int GetBatchSize() const override { return raw_input_.size(); }
  float GetQVal(int sample) const override {
    return wdls_[sample][0] - wdls_[sample][2];
  }
  float GetDVal(int sample) const override { return wdls_[sample][1]; }
  float GetPVal(int sample, int move_id) const override {
    return policies_[sample][move_id];
  }

 private:
  std::vector<InputPlanes> raw_input_;
  std::vector<std::vector<float>> wdls_;
  std::vector<std::vector<float>> policies_;
  const ExternalNetwork* network_;
};

std::unique_ptr<NetworkComputation> ExternalNetwork::NewComputation() {
  return std::make_unique<ExternalNetworkComputation>(this);
}

std::unique_ptr<Network> MakeExternalNetwork(const WeightsFile& weights,
                                             const OptionsDict& options) {
  return std::make_unique<ExternalNetwork>(weights, options);
}

REGISTER_NETWORK("external", MakeExternalNetwork, -999)

}  // namespace
}  // namespace lczero
