/*
  Originally from the Leela Zero project.
  Copyright (C) 2017 Gian-Carlo Pascutto

  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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
*/

#pragma once

#include <map>
#include <string>
#include <vector>

#include "OpenCLParams.h"
#include "neural/opencl/OpenCL.h"
#include "neural/opencl/OpenCLParams.h"

using Configurations = std::pair<std::string, std::vector<size_t>>;
using TuneParameters = std::map<std::string, size_t>;

class OpenCL;

class Tuner {
  OpenCL& m_opencl;
  const OpenCLParams& m_params;
  cl::Context m_context;
  cl::Device m_device;

 public:
  std::string tune_sgemm(const int m, const int n, const int k,
                         const int batch_size, const int runs = 4);
  std::string load_sgemm_tuners(const int m, const int n, const int k,
                                const int batch_size);

  static constexpr auto TUNER_VERSION = 0;
  Tuner(OpenCL& opencl, const OpenCLParams& params, cl::Context context,
        cl::Device device)
      : m_opencl(opencl),
        m_params(params),
        m_context(context),
        m_device(device) {}

 private:
  void store_sgemm_tuners(const int m, const int n, const int k,
                          const int batch_size, std::string tuners);
  bool valid_config_sgemm(TuneParameters p, bool exhaustive);
  std::string parameters_to_defines(const TuneParameters& p);
  std::string parameters_to_string(const TuneParameters& p);
  TuneParameters get_parameters_by_int(const std::vector<Configurations>& opts,
                                       const int n);
  std::string sgemm_tuners_from_line(std::string line, const int m, const int n,
                                     const int k, const int batch_size);
};
