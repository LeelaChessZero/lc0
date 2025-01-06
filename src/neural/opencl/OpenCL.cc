/*
  Originally from the Leela Zero project.
  Copyright (C) 2017 Gian-Carlo Pascutto

  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

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

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include "neural/opencl/OpenCL.h"
#include "neural/opencl/OpenCLParams.h"
#include "neural/opencl/OpenCLTuner.h"
#include "utils/logging.h"

static std::string cl_args =
    "-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros "
    "-cl-denorms-are-zero";

const std::string sourceCode_config =
#include "clsource/config.opencl"
    ;

const std::string sourceCode_convolve1 =
#include "clsource/convolve1.opencl"
    ;

const std::string sourceCode_convolve3 =
#include "clsource/convolve3.opencl"
    ;

const std::string sourceCode_se =
#include "clsource/se.opencl"
    ;

const std::string sourceCode_policymap =
#include "clsource/policymap.opencl"
    ;

const std::string sourceCode_blast_level3_common =
#include "clblast_level3/common.opencl"
    ;

const std::string sourceCode_blast_level3_xgemm_part1 =
#include "clblast_level3/xgemm_part1.opencl"
    ;

const std::string sourceCode_blast_level3_xgemm_part2 =
#include "clblast_level3/xgemm_part2.opencl"
    ;

const std::string sourceCode_blast_level3_xgemm_part3 =
#include "clblast_level3/xgemm_part3.opencl"
    ;

const std::string sourceCode_blast_level3_xgemm_batched =
#include "clblast_level3/xgemm_batched.opencl"
    ;

// Important: Keep the following order (common/part1/part2/part3/batched).
const std::string sourceCode_sgemm =
    sourceCode_blast_level3_common + sourceCode_blast_level3_xgemm_part1 +
    sourceCode_blast_level3_xgemm_part2 + sourceCode_blast_level3_xgemm_part3 +
    sourceCode_blast_level3_xgemm_batched;

const std::string sourceCode_sgemv =
#include "clblast_level3/xgemv.opencl"
    ;

void OpenCL_Network::add_weights(size_t layer, size_t size,
                                 const float* weights) {
  if (layer >= m_layers.size()) {
    m_layers.push_back(Layer());
  }

  auto converted_weights = std::vector<net_t>();
  for (auto i = size_t{0}; i < size; i++) {
    converted_weights.emplace_back(weights[i]);
  }

  auto weightSize = size * sizeof(decltype(converted_weights)::value_type);
  m_layers.back().weights.emplace_back(
      m_opencl.m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, weightSize,
      const_cast<net_t*>(converted_weights.data()));
}

void OpenCL_Network::add_weights_short(size_t layer, size_t size,
                                       const short* weights) {
  if (layer >= m_layers.size()) {
    m_layers.push_back(Layer());
  }

  auto weightSize = size * sizeof(short);
  m_layers.back().weights.emplace_back(m_opencl.m_context,
                                       CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                       weightSize, (void*)weights);
}

template <class T>
static std::string opencl_dev_type_to_string(T type) {
  if (type == CL_DEVICE_TYPE_CPU) {
    return "CPU";
  } else if (type == CL_DEVICE_TYPE_GPU) {
    return "GPU";
  } else if (type == CL_DEVICE_TYPE_ACCELERATOR) {
    return "Accelerator";
  } else {
    return "Unknown";
  }
}

static const char* trim_left(const char* trim_me) {
  while (isspace(*trim_me)) trim_me++;
  return trim_me;
}

void OpenCL::process_tuners(std::string tuners) {
  std::string buf;
  std::stringstream ss(tuners);
  std::size_t found;

  auto mwg = false;
  auto nwg = false;
  auto kwg = false;
  auto ndimc = false;
  auto mdimc = false;
  auto vwm = false;
  auto vwn = false;
  while (ss >> buf) {
    found = buf.find("=");
    if (found == std::string::npos) {
      CERR << "Invalid tuner string: " << tuners << std::endl;
      std::exit(-1);
    }
    std::string name = buf.substr(0, found);
    auto value = std::stoi(buf.substr(found + 1, std::string::npos));
    if (name == "-DMWG") {
      m_sgemm_tuners.mwg = value;
      mwg = true;
    }
    if (name == "-DNWG") {
      m_sgemm_tuners.nwg = value;
      nwg = true;
    }
    if (name == "-DKWG") {
      m_sgemm_tuners.kwg = value;
      kwg = true;
    }
    if (name == "-DMDIMC") {
      m_sgemm_tuners.mdimc = value;
      mdimc = true;
    }
    if (name == "-DNDIMC") {
      m_sgemm_tuners.ndimc = value;
      ndimc = true;
    }
    if (name == "-DVWM") {
      m_sgemm_tuners.vwm = value;
      vwm = true;
    }
    if (name == "-DVWN") {
      m_sgemm_tuners.vwn = value;
      vwn = true;
    }
  }
  if (!mwg || !nwg || !kwg || !mdimc || !ndimc || !vwm || !vwn) {
    CERR << "Missing tuner parameters";
    if (!mwg) {
      CERR << " MWG";
    }
    if (!nwg) {
      CERR << " NWG";
    }
    if (!kwg) {
      CERR << " KWG";
    }
    if (!mdimc) {
      CERR << " MDIMC";
    }
    if (!ndimc) {
      CERR << " NDIMC";
    }
    if (!vwm) {
      CERR << " VWM";
    }
    if (!vwn) {
      CERR << " VWN";
    }
    CERR << std::endl;
    std::exit(-1);
  }
}

std::vector<size_t> OpenCL::get_sgemm_tuners(void) {
  std::vector<size_t> tuners;

  tuners.emplace_back(m_sgemm_tuners.mwg);
  tuners.emplace_back(m_sgemm_tuners.nwg);
  tuners.emplace_back(m_sgemm_tuners.kwg);
  tuners.emplace_back(m_sgemm_tuners.vwm);
  tuners.emplace_back(m_sgemm_tuners.vwn);
  tuners.emplace_back(m_sgemm_tuners.mdimc);
  tuners.emplace_back(m_sgemm_tuners.ndimc);

  return tuners;
}

void OpenCL::initialize(const int channels, const OpenCLParams& params) {
  CERR << "Initializing OpenCL.";
  std::vector<cl::Platform> platforms;
  try {
    cl::Platform::get(&platforms);
  } catch (const cl::Error& e) {
    CERR << "OpenCL: " << e.what();
    throw;
  }

  auto best_version = 0.0f;
  cl::Platform best_platform;
  cl::Device best_device;
  std::string best_vendor;
  auto best_score = 0;
  auto found_device = false;
  auto id = 0;

  CERR << "Detected " << platforms.size() << " OpenCL platforms.";

  for (const auto& p : platforms) {
    std::string platvers = p.getInfo<CL_PLATFORM_VERSION>();
    std::string platprof = p.getInfo<CL_PLATFORM_PROFILE>();
    std::string platname = p.getInfo<CL_PLATFORM_NAME>();
    std::string platvend = p.getInfo<CL_PLATFORM_VENDOR>();
    CERR << "Platform version: " << platvers;
    CERR << "Platform profile: " << platprof;
    CERR << "Platform name:    " << platname;
    CERR << "Platform vendor:  " << platvend;

    std::istringstream versstream(platvers);
    std::string tmp;
    float opencl_version;
    versstream >> tmp >> opencl_version;

    std::vector<cl::Device> devices;
    try {
      p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    } catch (const cl::Error& e) {
      CERR << "Error getting device(s): " << e.what() << ": " << e.err()
           << std::endl;
      devices.clear();
    }
    for (auto& d : devices) {
      CERR << "Device ID:      " << id;
      CERR << "Device name:    "
           << trim_left(d.getInfo<CL_DEVICE_NAME>().c_str());
      CERR << "Device type:    "
           << opencl_dev_type_to_string(d.getInfo<CL_DEVICE_TYPE>());
      CERR << "Device vendor:  " << d.getInfo<CL_DEVICE_VENDOR>();
      CERR << "Device driver:  " << d.getInfo<CL_DRIVER_VERSION>();
      CERR << "Device speed:   " << d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>()
           << " MHZ";
      CERR << "Device cores:   " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
           << " CU";

      // assign score, try to find best device
      int this_score = 0;
      std::string this_vendor = d.getInfo<CL_DEVICE_VENDOR>();
      std::transform(this_vendor.begin(), this_vendor.end(),
                     this_vendor.begin(), ::tolower);
      this_score += 1000 * (this_vendor.find("advanced micro devices") !=
                            std::string::npos);
      this_score += 1000 * (this_vendor.find("amd") != std::string::npos);
      this_score += 1000 * (this_vendor.find("nvidia") != std::string::npos);
      this_score += 500 * (this_vendor.find("intel") != std::string::npos);
      this_score += 100 * (d.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU);
      this_score += opencl_version * 10;
      CERR << "Device score:   " << this_score;

      bool preferred = params.gpuId == id;

      if ((this_score > best_score) || preferred) {
        best_version = opencl_version;
        best_platform = p;
        best_device = d;
        best_vendor = this_vendor;
        if (preferred) {
          best_score = std::numeric_limits<decltype(best_score)>::max();
        } else {
          best_score = this_score;
        }
        found_device = true;
      }
      id++;
    }
  }

  if (!found_device) {
    throw std::runtime_error("No suitable OpenCL device found.");
  }

  CERR << "Selected platform: " << best_platform.getInfo<CL_PLATFORM_NAME>();
  CERR << "Selected device: "
       << trim_left(best_device.getInfo<CL_DEVICE_NAME>().c_str());
  CERR << "with OpenCL " << std::fixed << std::setprecision(1) << best_version
       << " capability.";
  cl::Context context;
  try {
    context = cl::Context(best_device);
  } catch (const cl::Error& e) {
    CERR << "Error creating OpenCL context: " << e.what() << ": " << e.err();
    throw std::runtime_error("Error creating OpenCL context.");
  }
  m_context = context;
  m_device = best_device;

  // Make program of the source code in the context.
  try {
    m_program = cl::Program(
        m_context, sourceCode_config + sourceCode_convolve1 +
                       sourceCode_convolve3 + sourceCode_se + sourceCode_sgemm +
                       sourceCode_sgemv + sourceCode_policymap);
  } catch (const cl::Error& e) {
    CERR << "Error getting kernels: " << e.what() << ": " << e.err();
    throw std::runtime_error("Error getting OpenCL kernels.");
  }

  m_cl_args = cl_args;

  auto t = Tuner(*this, params, m_context, m_device);
  auto sgemm_tuners = t.load_sgemm_tuners(
      channels, params.tune_batch_size * WINOGRAD_P, channels, WINOGRAD_TILE);

  // Build program for these specific devices.
  try {
    std::string args = cl_args;
    args += sgemm_tuners;
    m_program.build(args.c_str());
  } catch (const cl::Error&) {
    CERR << "Error building kernels: "
         << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device) << ".";
    throw std::runtime_error("Error building OpenCL kernels.");
  }

  process_tuners(sgemm_tuners);

  auto sgemm_kernel = cl::Kernel(m_program, "XgemmBatched");

  m_wavefront_size =
      sgemm_kernel
          .getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
              best_device);
  CERR << "Wavefront/Warp size: " << m_wavefront_size << std::endl;

  m_max_workgroup_size = best_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  m_max_workgroup_dims = best_device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

  CERR << "Max workgroup size: " << m_max_workgroup_size;
  std::ostringstream ss;
  for (auto d : m_max_workgroup_dims) ss << ' ' << d;
  CERR << "Max workgroup dimensions:" << ss.str();
  m_init_ok = true;
}

std::unique_ptr<OpenCLBuffers> OpenCL_Network::acquire_buffers() const {
  std::lock_guard<std::mutex> lock(m_pool_mutex);
  if (m_buffers_pool.empty()) return std::make_unique<OpenCLBuffers>(*this);
  auto result = std::move(m_buffers_pool.back());
  m_buffers_pool.pop_back();
  return result;
}

void OpenCL_Network::release_buffers(
    std::unique_ptr<OpenCLBuffers> buffers) const {
  std::lock_guard<std::mutex> lock(m_pool_mutex);
  m_buffers_pool.push_back(std::move(buffers));
}

std::string OpenCL::get_device_name() {
  std::stringstream ss;

  ss << "OpenCL: ";
  ss << m_device.getInfo<CL_DEVICE_VENDOR>() << " ";
  ss << m_device.getInfo<CL_DEVICE_NAME>() << " @ ";
  ss << m_device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "MHz";

  return ss.str();
}
