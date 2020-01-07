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

void* make_map(std::string file_name, unsigned long long needed_size)
{
  void* base_address;
#ifndef _WIN32
  struct stat statbuf;
  // TODO: create file with specific size, not open readonly.
  int fd = ::open(file_name.c_str(), O_RDONLY);
  if (fd == -1) return nullptr;
  base_address = mmap(nullptr, needed_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
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
        std::max(data.size() + 24, static_cast<size_t>(16) + 1024 * sizeof(float) * (116 * 8 * 8 + 1858 + 3)));
    // TODO: There is a race if external sees file and maps it before we clear this flag and the 'undefined value' happens to be read as a 1.
    static_cast<size_t*>(map_)[0] = 0;
    static_cast<size_t*>(map_)[2] = data.size();
    // write weights bytes at small offset.
    memcpy(static_cast<char*>(map_) + 24, data.data(), data.size());
    // Write 'weights ready' flag.
    static_cast<size_t*>(map_)[0] = 1;
    // Spin Wait for 'dest ready' flag.
    // TODO: use atomic read, not just volatile to get better semantic guarantee.
    while (static_cast<volatile size_t*>(map_)[1] != 1) {
      // Sleep a ms or two should be fine for this loop.  TODO:
    }
  }

  ~ExternalNetwork()
  {
    // unmap map_ and delete.
  }

  std::unique_ptr<NetworkComputation> NewComputation() override;

  void Compute(const std::vector<InputPlanes>& raw_input,
               std::vector<std::vector<float>>* wdls,
               std::vector<std::vector<float>>* policies) const {
    // Take lock.
    // Write raw_input at small offset with length.
    // Write 'input ready' flag value.
    static_cast<size_t*>(map_)[0] = 2;
    // Otherside clears input ready flag.
    // Spin Wait for 'output ready' flag value.
    // TODO: use atomic read, not just volatile to get better semantic
    // guarantee.
    while (static_cast<volatile size_t*>(map_)[1] != 2) {
      // TODO: no sleep for first n loops, then sleep 0.
    }
    // Clear 'output ready' flag. (Maybe atomic_compare_swap in spin wait?)
    static_cast<size_t*>(map_)[1] = 0;
    // Copy output in wdls/policies.
  }

  const NetworkCapabilities& GetCapabilities() const override {
    // TODO: use same capabilities as weights file implies.
    static NetworkCapabilities capabilities;
    return capabilities;
  }

 private:
  void* map_;
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
