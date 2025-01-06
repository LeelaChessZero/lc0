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

#pragma once

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

class OpenCL_Network;
class OpenCL;

class OpenCLBuffers {
  friend class OpenCL;
  friend class OpenCL_Network;

 public:
  OpenCLBuffers(const OpenCL_Network& opencl_net);

  void forward(const std::vector<net_t>& input, std::vector<net_t>& output_pol,
               std::vector<net_t>& output_val, std::vector<net_t>& output_mov,
               const int batch_size);

 private:
  using weight_slice_t = std::vector<cl::Buffer>::const_iterator;

  void convolve3(int channels, int outputs, cl::Buffer& bufferIn,
                 cl::Buffer& bufferOut, cl::Buffer& bufferV,
                 cl::Buffer& bufferM, weight_slice_t weights,
                 cl::Buffer* bufferResidual, weight_slice_t biases,
                 bool skip_in_transform, bool fuse_in_transform,
                 bool store_inout, bool relu, int batch_size);

  void convolve1(int channels, int outputs, cl::Buffer& bufferInput,
                 cl::Buffer& bufferOutput, cl::Buffer& bufferMerge,
                 weight_slice_t weights, weight_slice_t biases, int batch_size);

  void innerproduct(cl::Buffer& input, weight_slice_t weights,
                    weight_slice_t biases, cl::Buffer& output, const int inputs,
                    const int outputs, const int relu, int batch_size);

  void squeeze_excitation(int channels, int fc_outputs, cl::Buffer& bufferIn,
                          cl::Buffer& bufferTemp1, cl::Buffer& bufferTemp2,
                          weight_slice_t weights, cl::Buffer& bufferResidual,
                          int batch_size);

  void policymap(int N, const cl::Buffer& input, cl::Buffer& output,
                 const cl::Buffer& indices, int inputSize, int usedSize,
                 int outputSize);

  const OpenCL_Network& m_opencl_net;
  const OpenCL& m_opencl;

  cl::CommandQueue m_commandqueue;
  cl::Kernel m_convolve1_kernel;
  cl::Kernel m_merge_kernel;
  cl::Kernel m_in_transform_kernel;
  cl::Kernel m_sgemm_kernel;
  cl::Kernel m_sgemv_kernel;
  cl::Kernel m_out_transform_bn_kernel;
  cl::Kernel m_out_transform_bn_in_kernel;
  cl::Kernel m_global_avg_pooling_kernel;
  cl::Kernel m_apply_se_kernel;
  cl::Kernel m_policymap_kernel;
  cl::Buffer m_inBuffer;
  cl::Buffer m_inBuffer2;
  cl::Buffer m_VBuffer;
  cl::Buffer m_MBuffer;
  cl::Buffer m_pool_buffer;
  cl::Buffer m_pinnedOutBuffer_pol;
  cl::Buffer m_pinnedOutBuffer_val;
  cl::Buffer m_pinnedOutBuffer_mov;
  size_t m_finalSize_pol;
  size_t m_finalSize_val;
  size_t m_finalSize_mov;
};
