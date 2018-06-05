/*
 This file is part of Leela Zero.
 Copyright (C) 2017 Gian-Carlo Pascutto

 Leela Zero is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 Leela Zero is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
 */


#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include "neural/CL/OpenCL.h"
#include "neural/CL/OpenCLParams.h"
#include "neural/CL/OpenCLTuner.h"


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

thread_local ThreadData opencl_thread_data;

void OpenCL::ensure_thread_initialized() {
  if (!opencl_thread_data.m_is_initialized) {
    // Make kernels
    opencl_thread_data.m_convolve1_kernel = cl::Kernel(m_program, "convolve1");
    opencl_thread_data.m_merge_kernel = cl::Kernel(m_program, "merge_bn");
    opencl_thread_data.m_in_transform_kernel =
        cl::Kernel(m_program, "in_transform");
    opencl_thread_data.m_sgemm_kernel = cl::Kernel(m_program, "XgemmBatched");
    opencl_thread_data.m_out_transform_bn_kernel =
        cl::Kernel(m_program, "out_transform_fused_bn");
    opencl_thread_data.m_out_transform_bn_in_kernel =
        cl::Kernel(m_program, "out_transform_fused_bn_in");
    opencl_thread_data.m_sgemv_kernel = cl::Kernel(m_program, "Xgemv");
    opencl_thread_data.m_commandqueue = cl::CommandQueue(m_context, m_device);
    opencl_thread_data.m_is_initialized = true;
  }
}

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

void OpenCL_Network::forward(const std::vector<net_t>& input,
                             std::vector<net_t>& output_pol,
                             std::vector<net_t>& output_val) const {
  constexpr auto tiles = WINOGRAD_P;

  auto finalSize_pol =
      m_layers[m_layers.size() - 2].ip_out_size * sizeof(net_t);
  auto finalSize_val = m_layers.back().ip_out_size * sizeof(net_t);

  if (m_layers.back().is_policy) {
    std::swap(finalSize_pol, finalSize_val);
  }

  m_opencl.ensure_thread_initialized();

  if (!opencl_thread_data.m_buffers_allocated) {
    auto max_channels = unsigned{0};
    for (const auto& layer : m_layers) {
      max_channels =
          std::max(max_channels, std::max(layer.channels, layer.outputs));
    }

    const auto mwg = m_opencl.m_sgemm_tuners.mwg;
    const auto nwg = m_opencl.m_sgemm_tuners.nwg;
    const auto vwm = m_opencl.m_sgemm_tuners.vwm;
    const auto vwn = m_opencl.m_sgemm_tuners.vwn;

    const auto m_ceil = ceilMultiple(ceilMultiple(max_channels, mwg), vwm);
    const auto n_ceil = ceilMultiple(ceilMultiple(tiles, nwg), vwn);

    const auto alloc_inSize = m_ceil * m_ceil * max_channels * sizeof(net_t);
    const auto alloc_vm_size = WINOGRAD_TILE * m_ceil * n_ceil * sizeof(net_t);

    auto v_zeros = std::vector<float>(alloc_vm_size);

    opencl_thread_data.m_inBuffer =
        cl::Buffer(m_opencl.m_context, CL_MEM_READ_WRITE, alloc_inSize);
    opencl_thread_data.m_inBuffer2 =
        cl::Buffer(m_opencl.m_context, CL_MEM_READ_WRITE, alloc_inSize);
    opencl_thread_data.m_VBuffer = cl::Buffer(
        m_opencl.m_context,
        CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
        alloc_vm_size, v_zeros.data(), nullptr);
    opencl_thread_data.m_MBuffer =
        cl::Buffer(m_opencl.m_context,
                   CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, alloc_vm_size);

    opencl_thread_data.m_pinnedOutBuffer_pol =
        cl::Buffer(m_opencl.m_context,
                   CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, finalSize_pol);
    opencl_thread_data.m_pinnedOutBuffer_val =
        cl::Buffer(m_opencl.m_context,
                   CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, finalSize_val);

    opencl_thread_data.m_buffers_allocated = true;
  }

  cl::Buffer& inBuffer = opencl_thread_data.m_inBuffer;
  cl::Buffer& inBuffer2 = opencl_thread_data.m_inBuffer2;
  cl::Buffer& VBuffer = opencl_thread_data.m_VBuffer;
  cl::Buffer& MBuffer = opencl_thread_data.m_MBuffer;
  cl::CommandQueue& queue = opencl_thread_data.m_commandqueue;

  const auto inSize = sizeof(net_t) * input.size();
  queue.enqueueWriteBuffer(inBuffer, CL_FALSE, 0, inSize, input.data());

  auto skip_in_trans = false;
  for (auto iter = cbegin(m_layers); iter != cend(m_layers); iter++) {
    const auto& layer = *iter;
    const auto niter = std::next(iter);

    if (layer.is_input_convolution) {
      assert(niter != cend(m_layers));
      auto conv_weights = begin(layer.weights);
      auto bn_weights = begin(layer.weights) + 1;
      auto skip_next_in_trans = false;
      if (niter->is_residual_block) {
        skip_next_in_trans = true;
      }
      convolve3(layer.channels, layer.outputs, inBuffer, inBuffer, VBuffer,
                MBuffer, conv_weights, nullptr, bn_weights, skip_in_trans,
                skip_next_in_trans, true);
      skip_in_trans = skip_next_in_trans;
    } else if (layer.is_residual_block) {
      assert(layer.channels == layer.outputs);
      assert(niter != cend(m_layers));
      auto conv1_weights = begin(layer.weights);
      auto bn1_weights = begin(layer.weights) + 1;
      auto conv2_weights = begin(layer.weights) + 3;
      auto bn2_weights = begin(layer.weights) + 4;
      convolve3(layer.channels, layer.outputs, inBuffer, inBuffer2, VBuffer,
                MBuffer, conv1_weights, nullptr, bn1_weights, skip_in_trans,
                true, false);

      auto skip_next_in_trans = false;
      if (niter->is_residual_block) {
        skip_next_in_trans = true;
      }
      convolve3(layer.channels, layer.outputs, inBuffer2, inBuffer, VBuffer,
                MBuffer, conv2_weights, &inBuffer, bn2_weights, true,
                skip_next_in_trans, true);
      skip_in_trans = skip_next_in_trans;
    } else {
      assert(layer.is_value || layer.is_policy);

      cl::Buffer out_buffer;
      if (layer.is_policy) {
        out_buffer = opencl_thread_data.m_pinnedOutBuffer_pol;
      } else {
        out_buffer = opencl_thread_data.m_pinnedOutBuffer_val;
      }

      auto ip_w = begin(layer.weights) + 3;
      auto ip_b = begin(layer.weights) + 4;

      convolve1(layer.channels, layer.outputs, inBuffer, inBuffer2, VBuffer,
                begin(layer.weights));

      innerproduct(inBuffer2, ip_w, ip_b, out_buffer, layer.ip_in_size,
                   layer.ip_out_size, layer.is_value);
    }
  }

  auto pinnedOutBufferHost_pol =
      queue.enqueueMapBuffer(opencl_thread_data.m_pinnedOutBuffer_pol, CL_FALSE,
                             CL_MAP_READ, 0, finalSize_pol);
  auto pinnedOutBufferHost_val =
      queue.enqueueMapBuffer(opencl_thread_data.m_pinnedOutBuffer_val, CL_FALSE,
                             CL_MAP_READ, 0, finalSize_val);

  {
    // Finish call is usually a busy wait. When using multiple threads
    // use the lock to avoid busy waiting with all threads.
    std::lock_guard<std::mutex> lock(m_queue_finish_mutex);
    queue.finish();
  }

  std::memcpy(output_pol.data(), pinnedOutBufferHost_pol, finalSize_pol);
  std::memcpy(output_val.data(), pinnedOutBufferHost_val, finalSize_val);

  queue.enqueueUnmapMemObject(opencl_thread_data.m_pinnedOutBuffer_pol,
                              pinnedOutBufferHost_pol);
  queue.enqueueUnmapMemObject(opencl_thread_data.m_pinnedOutBuffer_val,
                              pinnedOutBufferHost_val);
}

void OpenCL_Network::convolve3(int channels, int outputs, cl::Buffer& bufferIn,
                               cl::Buffer& bufferOut, cl::Buffer& bufferV,
                               cl::Buffer& bufferM, weight_slice_t weights,
                               cl::Buffer* bufferResidual,
                               weight_slice_t bn_weights,
                               bool skip_in_transform, bool fuse_in_transform,
                               bool store_inout) const {
  cl::Kernel& in_transform_kernel = opencl_thread_data.m_in_transform_kernel;
  cl::Kernel& sgemm_kernel = opencl_thread_data.m_sgemm_kernel;
  cl::Kernel& out_transform_bn_kernel =
      opencl_thread_data.m_out_transform_bn_kernel;
  cl::Kernel& out_transform_bn_in_kernel =
      opencl_thread_data.m_out_transform_bn_in_kernel;

  auto mwg = m_opencl.m_sgemm_tuners.mwg;
  auto nwg = m_opencl.m_sgemm_tuners.nwg;
  auto kwg = m_opencl.m_sgemm_tuners.kwg;
  auto vwm = m_opencl.m_sgemm_tuners.vwm;
  auto vwn = m_opencl.m_sgemm_tuners.vwn;
  auto mdimc = m_opencl.m_sgemm_tuners.mdimc;
  auto ndimc = m_opencl.m_sgemm_tuners.ndimc;
  auto wavefront_size = m_opencl.m_wavefront_size;

  assert(mwg != 0);
  assert(nwg != 0);
  assert(kwg != 0);
  assert(mdimc != 0);
  assert(ndimc != 0);
  assert(vwm != 0);
  assert(vwn != 0);
  assert(wavefront_size != 0);

  constexpr auto tiles = WINOGRAD_P;
  constexpr auto width = 8;
  constexpr auto height = 8;

  auto wgs = ceilMultiple(tiles, wavefront_size);
  auto m_ceil = int(ceilMultiple(ceilMultiple(outputs, mwg), vwm));
  auto n_ceil = int(ceilMultiple(ceilMultiple(tiles, nwg), vwn));
  auto k_ceil = int(ceilMultiple(ceilMultiple(channels, kwg), vwm));

  cl::CommandQueue& queue = opencl_thread_data.m_commandqueue;

  if (!skip_in_transform) {
    try {
      in_transform_kernel.setArg(0, bufferIn);
      in_transform_kernel.setArg(1, bufferV);
      in_transform_kernel.setArg(2, channels);
      in_transform_kernel.setArg(3, k_ceil);
      in_transform_kernel.setArg(4, n_ceil);

      queue.enqueueNDRangeKernel(in_transform_kernel, cl::NullRange,
                                 cl::NDRange(wgs, channels));
    } catch (const cl::Error& e) {
      std::cerr << "Error in convolve3: " << e.what() << ": " << e.err()
                << std::endl;
      throw;
    }
  }

  try {
    sgemm_kernel.setArg(0, m_ceil);
    sgemm_kernel.setArg(1, n_ceil);
    sgemm_kernel.setArg(2, k_ceil);
    sgemm_kernel.setArg(3, weights[0]);
    sgemm_kernel.setArg(4, bufferV);
    sgemm_kernel.setArg(5, bufferM);

    cl::NDRange local_sgemm = {mdimc, ndimc, 1};

    cl::NDRange size_sgemm = {(m_ceil * mdimc) / mwg, (n_ceil * ndimc) / nwg,
                              (cl::size_type)WINOGRAD_TILE};

    queue.enqueueNDRangeKernel(sgemm_kernel, cl::NullRange, size_sgemm,
                               local_sgemm);
  } catch (const cl::Error& e) {
    std::cerr << "Error in convolve3: " << e.what() << ": " << e.err()
              << std::endl;
    throw;
  }

  try {
    if (fuse_in_transform) {
      // TODO : Eventually this might also be something tuneable?
      constexpr auto dim_size = 2;
      out_transform_bn_in_kernel.setArg(0, bufferM);
      if (store_inout) {
        out_transform_bn_in_kernel.setArg(1, bufferOut);
      } else {
        out_transform_bn_in_kernel.setArg(1, nullptr);
      }
      out_transform_bn_in_kernel.setArg(2, bufferV);
      out_transform_bn_in_kernel.setArg(3, outputs);
      out_transform_bn_in_kernel.setArg(4, m_ceil);
      out_transform_bn_in_kernel.setArg(5, n_ceil);
      // k_ceil of the next convolution
      auto k_ceil2 = int(ceilMultiple(ceilMultiple(outputs, kwg), vwm));
      out_transform_bn_in_kernel.setArg(6, k_ceil2);
      if (bufferResidual) {
        out_transform_bn_in_kernel.setArg(7, *bufferResidual);
      } else {
        out_transform_bn_in_kernel.setArg(7, nullptr);
      }
      out_transform_bn_in_kernel.setArg(8, bn_weights[0]);
      out_transform_bn_in_kernel.setArg(9, bn_weights[1]);
      out_transform_bn_in_kernel.setArg(
          10, cl::Local(dim_size * width * height * sizeof(float)));

      queue.enqueueNDRangeKernel(out_transform_bn_in_kernel, cl::NullRange,
                                 cl::NDRange(outputs, wgs),
                                 cl::NDRange(dim_size, wgs));
    } else {
      out_transform_bn_kernel.setArg(0, bufferM);
      out_transform_bn_kernel.setArg(1, bufferOut);
      out_transform_bn_kernel.setArg(2, outputs);
      out_transform_bn_kernel.setArg(3, m_ceil);
      out_transform_bn_kernel.setArg(4, n_ceil);
      if (bufferResidual) {
        out_transform_bn_kernel.setArg(5, *bufferResidual);
      } else {
        out_transform_bn_kernel.setArg(5, nullptr);
      }
      out_transform_bn_kernel.setArg(6, bn_weights[0]);
      out_transform_bn_kernel.setArg(7, bn_weights[1]);

      queue.enqueueNDRangeKernel(out_transform_bn_kernel, cl::NullRange,
                                 cl::NDRange(outputs, wgs));
    }
  } catch (const cl::Error& e) {
    std::cerr << "Error in convolve3: " << e.what() << ": " << e.err()
              << std::endl;
    throw;
  }
}

void OpenCL_Network::convolve1(int channels, int outputs,
                               cl::Buffer& bufferInput,
                               cl::Buffer& bufferOutput,
                               cl::Buffer& bufferMerge,
                               weight_slice_t weights) const {
  // fixed for 8x8
  constexpr int width = 8;
  constexpr int height = 8;
  constexpr int boardsize = width * height;
  constexpr int rowTiles = 8;

  // Input channel grouping in multiples of 8
  constexpr int channelGroup = 8;
  constexpr int channelShift = 3;
  constexpr int rowGroup = 1;
  size_t outputGroup = std::min(outputs, 32);

  auto m_convolve_kernel = &opencl_thread_data.m_convolve1_kernel;

#ifndef NDEBUG
  // Total output size after reducing
  size_t outSize = width * height * outputs * sizeof(net_t);

  // Produce channel * output planes and merge them at the end
  size_t mergeSize = (channels >> channelShift) * outSize;
  assert(mergeSize <= bufferMerge.getInfo<CL_MEM_SIZE>());
#endif

  // Copy the rows locally
  size_t stripSize = width * sizeof(float);

  int rowBuffer = std::min<int>(channelGroup, 7);
  size_t rowSize = channelGroup * outputGroup * rowBuffer * sizeof(float);

  cl::CommandQueue& queue = opencl_thread_data.m_commandqueue;

  try {
    m_convolve_kernel->setArg(0, bufferInput);
    m_convolve_kernel->setArg(1, bufferMerge);
    m_convolve_kernel->setArg(2, weights[0]);
    m_convolve_kernel->setArg(3,
                              cl::Local(stripSize * channelGroup * rowGroup));
    m_convolve_kernel->setArg(4, cl::Local(rowSize));

    queue.enqueueNDRangeKernel(
        *m_convolve_kernel, cl::NullRange,
        cl::NDRange(channels, outputs, rowTiles),
        cl::NDRange(channelGroup, outputGroup, rowGroup));
  } catch (const cl::Error& e) {
    std::cerr << "Error in convolve1: " << e.what() << ": " << e.err()
              << std::endl;
    throw;
  }

  cl::Kernel& merge_kernel = opencl_thread_data.m_merge_kernel;
  assert(channels % (1 << channelShift) == 0);

  try {
    merge_kernel.setArg(0, bufferMerge);
    merge_kernel.setArg(1, bufferOutput);
    merge_kernel.setArg(2, channels >> channelShift);
    merge_kernel.setArg(3, weights[1]);
    merge_kernel.setArg(4, weights[2]);

    queue.enqueueNDRangeKernel(merge_kernel, cl::NullRange,
                               cl::NDRange(outputs, boardsize),
                               cl::NDRange(std::min(8, outputs), 8));
  } catch (const cl::Error& e) {
    std::cerr << "Error in merge: " << e.what() << ": " << e.err() << std::endl;
    throw;
  }
}

void OpenCL_Network::innerproduct(cl::Buffer& input, weight_slice_t weights,
                                  weight_slice_t biases, cl::Buffer& output,
                                  const int inputs, const int outputs,
                                  const int relu) const {
  auto sgemv_kernel = opencl_thread_data.m_sgemv_kernel;
  cl::CommandQueue& queue = opencl_thread_data.m_commandqueue;

  // TODO: Tune these
  size_t wgs1 = 64;
  size_t wpt1 = 1;

  auto m_ceil = int(ceilMultiple(outputs, wgs1 * wpt1));
  auto global_size = m_ceil / wpt1;
  auto local_size = wgs1;

  try {
    // Sets the kernel arguments
    sgemv_kernel.setArg(0, static_cast<int>(outputs));
    sgemv_kernel.setArg(1, static_cast<int>(inputs));
    sgemv_kernel.setArg(2, weights[0]);
    sgemv_kernel.setArg(3, static_cast<int>(0));
    sgemv_kernel.setArg(4, static_cast<int>(inputs));
    sgemv_kernel.setArg(5, input);
    sgemv_kernel.setArg(6, static_cast<int>(0));
    sgemv_kernel.setArg(7, output);
    sgemv_kernel.setArg(8, static_cast<int>(0));
    sgemv_kernel.setArg(9, biases[0]);
    sgemv_kernel.setArg(10, static_cast<int>(relu));

    queue.enqueueNDRangeKernel(sgemv_kernel, cl::NullRange,
                               cl::NDRange(global_size),
                               cl::NDRange(local_size));
  } catch (const cl::Error& e) {
    std::cerr << "Error in innerproduct: " << e.what() << ": " << e.err()
              << std::endl;
    throw;
  }
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

static std::string trim(std::string trim_me) {
  boost::algorithm::trim(trim_me);
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
      std::cerr << "Invalid tuner string: " << tuners << std::endl;
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
    std::cerr << "Missing tuner parameters";
    if (!mwg) {
      std::cerr << " MWG";
    }
    if (!nwg) {
      std::cerr << " NWG";
    }
    if (!kwg) {
      std::cerr << " KWG";
    }
    if (!mdimc) {
      std::cerr << " MDIMC";
    }
    if (!ndimc) {
      std::cerr << " NDIMC";
    }
    if (!vwm) {
      std::cerr << " VWM";
    }
    if (!vwn) {
      std::cerr << " VWN";
    }
    std::cerr << std::endl;
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
  bool verbose = params.verbose;
  if (verbose) {
    fprintf(stderr, "Initializing OpenCL.\n");
  }
  std::vector<cl::Platform> platforms;
  try {
    cl::Platform::get(&platforms);
  } catch (const cl::Error& e) {
    fprintf(stderr, "OpenCL: %s\n", e.what());
    throw;
  }

  auto best_version = 0.0f;
  cl::Platform best_platform;
  cl::Device best_device;
  std::string best_vendor;
  auto best_score = 0;
  auto found_device = false;
  auto id = 0;

  if (verbose) {
    fprintf(stderr, "Detected %zu OpenCL platforms.\n", platforms.size());
  }

  for (const auto& p : platforms) {
    std::string platvers = p.getInfo<CL_PLATFORM_VERSION>();
    if (verbose) {
      std::string platprof = p.getInfo<CL_PLATFORM_PROFILE>();
      std::string platname = p.getInfo<CL_PLATFORM_NAME>();
      std::string platvend = p.getInfo<CL_PLATFORM_VENDOR>();
      fprintf(stderr, "Platform version: %s\n", platvers.c_str());
      ;
      fprintf(stderr, "Platform profile: %s\n", platprof.c_str());
      fprintf(stderr, "Platform name:    %s\n", platname.c_str());
      fprintf(stderr, "Platform vendor:  %s\n", platvend.c_str());
    }

    std::istringstream versstream(platvers);
    std::string tmp;
    float opencl_version;
    versstream >> tmp >> opencl_version;

    std::vector<cl::Device> devices;
    try {
      p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    } catch (const cl::Error& e) {
      fprintf(stderr, "Error getting device(s): %s: %d\n", e.what(), e.err());
      devices.clear();
    }
    for (auto& d : devices) {
      if (verbose) {
        fprintf(stderr, "Device ID:     %d\n", id);
        fprintf(stderr, "Device name:   %s\n",
               trim(d.getInfo<CL_DEVICE_NAME>()).c_str());
        fprintf(stderr, "Device type:   %s\n",
               opencl_dev_type_to_string(d.getInfo<CL_DEVICE_TYPE>()).c_str());
        fprintf(stderr, "Device vendor: %s\n", d.getInfo<CL_DEVICE_VENDOR>().c_str());
        fprintf(stderr, "Device driver: %s\n", d.getInfo<CL_DRIVER_VERSION>().c_str());
        fprintf(stderr, "Device speed:  %u MHz\n",
               d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>());
        fprintf(stderr, "Device cores:  %u CU\n",
               d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());
      }

      // assign score, try to find best device
      int this_score = 0;
      std::string this_vendor = d.getInfo<CL_DEVICE_VENDOR>();
      this_score +=
          1000 * boost::icontains(this_vendor, "advanced micro devices");
      this_score += 1000 * boost::icontains(this_vendor, "amd");
      this_score += 1000 * boost::icontains(this_vendor, "nvidia");
      this_score += 500 * boost::icontains(this_vendor, "intel");
      this_score += 100 * (d.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU);
      this_score += opencl_version * 10;
      if (verbose) {
        fprintf(stderr, "Device score:  %d\n", this_score);
      }

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

  if (verbose) {
    fprintf(stderr, "Selected platform: %s\n",
           best_platform.getInfo<CL_PLATFORM_NAME>().c_str());
    fprintf(stderr, "Selected device: %s\n",
           trim(best_device.getInfo<CL_DEVICE_NAME>()).c_str());
    fprintf(stderr, "with OpenCL %2.1f capability.\n", best_version);
  }
  cl::Context context;
  try {
    context = cl::Context(best_device);
  } catch (const cl::Error& e) {
    fprintf(stderr, "Error creating OpenCL context: %s: %d", e.what(), e.err());
    throw std::runtime_error("Error creating OpenCL context.");
  }
  m_context = context;
  m_device = best_device;

  // Make program of the source code in the context
  try {
    m_program =
        cl::Program(m_context, sourceCode_config + sourceCode_convolve1 +
                                   sourceCode_convolve3 + sourceCode_sgemm +
                                   sourceCode_sgemv);
  } catch (const cl::Error& e) {
    fprintf(stderr, "Error getting kernels: %s: %d", e.what(), e.err());
    throw std::runtime_error("Error getting OpenCL kernels.");
  }

  m_cl_args = cl_args;

  auto t = Tuner(*this, params, m_context, m_device);
  auto sgemm_tuners =
      t.load_sgemm_tuners(channels, WINOGRAD_P, channels, WINOGRAD_TILE);

  // Build program for these specific devices
  try {
    std::string args = cl_args;
    args += sgemm_tuners;
    m_program.build(args.c_str());
  } catch (const cl::Error&) {
    fprintf(stderr, "Error building kernels: %s\n",
           m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device).c_str());
    throw std::runtime_error("Error building OpenCL kernels.");
  }

  ensure_thread_initialized();
  process_tuners(sgemm_tuners);

  m_wavefront_size =
      opencl_thread_data.m_sgemm_kernel
          .getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
              best_device);
  if (verbose) {
    fprintf(stderr, "Wavefront/Warp size: %zu\n", m_wavefront_size);
  }

  m_max_workgroup_size = best_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  m_max_workgroup_dims = best_device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

  if (verbose) {
    fprintf(stderr, "Max workgroup size: %zu\n", m_max_workgroup_size);
    fprintf(stderr, "Max workgroup dimensions: ");
    for (auto d : m_max_workgroup_dims) {
      fprintf(stderr, "%zu ", d);
    }
    fprintf(stderr, "\n");
  }
  m_init_ok = true;
}

std::string OpenCL::get_device_name() {
  std::stringstream ss;

  ss << "OpenCL: ";
  ss << m_device.getInfo<CL_DEVICE_VENDOR>() << " ";
  ss << m_device.getInfo<CL_DEVICE_NAME>() << " @ ";
  ss << m_device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "MHz";

  return ss.str();
}
