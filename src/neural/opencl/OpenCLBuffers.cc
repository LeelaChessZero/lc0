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

#include "neural/opencl/OpenCLBuffers.h"

OpenCLBuffers::OpenCLBuffers(OpenCL_Network& opencl_net)
    : m_opencl_net(opencl_net), m_opencl(opencl_net.getOpenCL()) {
  auto& program = m_opencl.m_program;
  auto& context = m_opencl.m_context;
  auto& device = m_opencl.m_device;

  m_convolve1_kernel = cl::Kernel(program, "convolve1");
  m_merge_kernel = cl::Kernel(program, "merge_bn");
  m_in_transform_kernel = cl::Kernel(program, "in_transform");
  m_sgemm_kernel = cl::Kernel(program, "XgemmBatched");
  m_out_transform_bn_kernel = cl::Kernel(program, "out_transform_fused_bn");
  m_out_transform_bn_in_kernel =
      cl::Kernel(program, "out_transform_fused_bn_in");
  m_sgemv_kernel = cl::Kernel(program, "Xgemv");
  m_commandqueue = cl::CommandQueue(context, device);

  auto& m_layers = m_opencl_net.m_layers;

  constexpr auto tiles = WINOGRAD_P;
  constexpr auto width = 8;
  constexpr auto height = 8;

  auto finalSize_pol =
      m_layers[m_layers.size() - 2].ip_out_size * sizeof(net_t);
  auto finalSize_val = m_layers.back().ip_out_size * sizeof(net_t);

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

  const auto max_batch_size = m_opencl_net.getMaxMatchSize();
  const auto alloc_inSize =
      max_batch_size * width * height * max_channels * sizeof(net_t);
  const auto alloc_vm_size =
      max_batch_size * WINOGRAD_TILE * m_ceil * n_ceil * sizeof(net_t);

  auto v_zeros = std::vector<float>(alloc_vm_size);

  m_inBuffer = cl::Buffer(m_opencl.m_context, CL_MEM_READ_WRITE, alloc_inSize);
  m_inBuffer2 = cl::Buffer(m_opencl.m_context, CL_MEM_READ_WRITE, alloc_inSize);
  m_VBuffer = cl::Buffer(
      m_opencl.m_context,
      CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
      alloc_vm_size, v_zeros.data(), nullptr);
  m_MBuffer =
      cl::Buffer(m_opencl.m_context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                 alloc_vm_size);

  m_pinnedOutBuffer_pol =
      cl::Buffer(m_opencl.m_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                 max_batch_size * finalSize_pol);
  m_pinnedOutBuffer_val =
      cl::Buffer(m_opencl.m_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                 max_batch_size * finalSize_val);
}

void OpenCLBuffers::forward(const std::vector<net_t>& input,
                            std::vector<net_t>& output_pol,
                            std::vector<net_t>& output_val,
                            const int batch_size) {
 
  auto& m_layers = m_opencl_net.m_layers;

  auto finalSize_pol =
      m_layers[m_layers.size() - 2].ip_out_size * sizeof(net_t);
  auto finalSize_val = m_layers.back().ip_out_size * sizeof(net_t);

  const auto inSize = sizeof(net_t) * input.size();
  m_commandqueue.enqueueWriteBuffer(m_inBuffer, CL_FALSE, 0, inSize,
                                    input.data());

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
      convolve3(layer.channels, layer.outputs, m_inBuffer, m_inBuffer,
                m_VBuffer, m_MBuffer, conv_weights, nullptr, bn_weights,
                skip_in_trans, skip_next_in_trans, true, batch_size);
      skip_in_trans = skip_next_in_trans;
    } else if (layer.is_residual_block) {
      assert(layer.channels == layer.outputs);
      assert(niter != cend(m_layers));
      auto conv1_weights = begin(layer.weights);
      auto bn1_weights = begin(layer.weights) + 1;
      auto conv2_weights = begin(layer.weights) + 3;
      auto bn2_weights = begin(layer.weights) + 4;
      convolve3(layer.channels, layer.outputs, m_inBuffer, m_inBuffer2,
                m_VBuffer, m_MBuffer, conv1_weights, nullptr, bn1_weights,
                skip_in_trans, true, false, batch_size);

      auto skip_next_in_trans = false;
      if (niter->is_residual_block) {
        skip_next_in_trans = true;
      }
      convolve3(layer.channels, layer.outputs, m_inBuffer2, m_inBuffer,
                m_VBuffer, m_MBuffer, conv2_weights, &m_inBuffer, bn2_weights,
                true, skip_next_in_trans, true, batch_size);
      skip_in_trans = skip_next_in_trans;
    } else {
      assert(layer.is_value || layer.is_policy);

      cl::Buffer out_buffer;
      if (layer.is_policy) {
        out_buffer = m_pinnedOutBuffer_pol;
      } else {
        out_buffer = m_pinnedOutBuffer_val;
      }

      auto ip_w = begin(layer.weights) + 3;
      auto ip_b = begin(layer.weights) + 4;

      convolve1(layer.channels, layer.outputs, m_inBuffer, m_inBuffer2,
                m_VBuffer, begin(layer.weights), batch_size);

      innerproduct(m_inBuffer2, ip_w, ip_b, out_buffer, layer.ip_in_size,
                   layer.ip_out_size, layer.is_value, batch_size);
    }
  }

  auto pinnedOutBufferHost_pol = m_commandqueue.enqueueMapBuffer(
      m_pinnedOutBuffer_pol, CL_FALSE, CL_MAP_READ, 0,
      batch_size * finalSize_pol);
  auto pinnedOutBufferHost_val = m_commandqueue.enqueueMapBuffer(
      m_pinnedOutBuffer_val, CL_FALSE, CL_MAP_READ, 0,
      batch_size * finalSize_val);

  {
    // Finish call is usually a busy wait. When using multiple threads,
    // use the lock to avoid busy waiting with all threads.
    std::lock_guard<std::mutex> lock(m_queue_finish_mutex);
    m_commandqueue.finish();
  }

  std::memcpy(output_pol.data(), pinnedOutBufferHost_pol,
              batch_size * finalSize_pol);
  std::memcpy(output_val.data(), pinnedOutBufferHost_val,
              batch_size * finalSize_val);

  m_commandqueue.enqueueUnmapMemObject(m_pinnedOutBuffer_pol,
                                       pinnedOutBufferHost_pol);
  m_commandqueue.enqueueUnmapMemObject(m_pinnedOutBuffer_val,
                                       pinnedOutBufferHost_val);
}

void OpenCLBuffers::convolve3(int channels, int outputs, cl::Buffer& bufferIn,
                              cl::Buffer& bufferOut, cl::Buffer& bufferV,
                              cl::Buffer& bufferM, weight_slice_t weights,
                              cl::Buffer* bufferResidual,
                              weight_slice_t bn_weights, bool skip_in_transform,
                              bool fuse_in_transform, bool store_inout,
                              int batch_size) {
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
  auto n_ceil = int(ceilMultiple(ceilMultiple(batch_size * tiles, nwg), vwn));
  auto k_ceil = int(ceilMultiple(ceilMultiple(channels, kwg), vwm));

  if (!skip_in_transform) {
    try {
      m_in_transform_kernel.setArg(0, bufferIn);
      m_in_transform_kernel.setArg(1, bufferV);
      m_in_transform_kernel.setArg(2, channels);
      m_in_transform_kernel.setArg(3, k_ceil);
      m_in_transform_kernel.setArg(4, n_ceil);

      m_commandqueue.enqueueNDRangeKernel(
          m_in_transform_kernel, cl::NullRange,
          cl::NDRange(wgs, channels, batch_size));
    } catch (const cl::Error& e) {
      CERR << "Error in convolve3: " << e.what() << ": " << e.err()
           << std::endl;
      throw;
    }
  }

  try {
    m_sgemm_kernel.setArg(0, m_ceil);
    m_sgemm_kernel.setArg(1, n_ceil);
    m_sgemm_kernel.setArg(2, k_ceil);
    m_sgemm_kernel.setArg(3, weights[0]);
    m_sgemm_kernel.setArg(4, bufferV);
    m_sgemm_kernel.setArg(5, bufferM);

    cl::NDRange local_sgemm = {mdimc, ndimc, 1};

    cl::NDRange size_sgemm = {(m_ceil * mdimc) / mwg, (n_ceil * ndimc) / nwg,
                              (cl::size_type)WINOGRAD_TILE};

    m_commandqueue.enqueueNDRangeKernel(m_sgemm_kernel, cl::NullRange,
                                        size_sgemm, local_sgemm);
  } catch (const cl::Error& e) {
    CERR << "Error in convolve3: " << e.what() << ": " << e.err() << std::endl;
    throw;
  }

  try {
    if (fuse_in_transform) {
      // TODO : Eventually this might also be something tuneable?
      constexpr auto dim_size = 2;
      m_out_transform_bn_in_kernel.setArg(0, bufferM);
      if (store_inout) {
        m_out_transform_bn_in_kernel.setArg(1, bufferOut);
      } else {
        m_out_transform_bn_in_kernel.setArg(1, nullptr);
      }
      m_out_transform_bn_in_kernel.setArg(2, bufferV);
      m_out_transform_bn_in_kernel.setArg(3, outputs);
      m_out_transform_bn_in_kernel.setArg(4, m_ceil);
      m_out_transform_bn_in_kernel.setArg(5, n_ceil);
      // k_ceil of the next convolution
      auto k_ceil2 = int(ceilMultiple(ceilMultiple(outputs, kwg), vwm));
      m_out_transform_bn_in_kernel.setArg(6, k_ceil2);
      if (bufferResidual) {
        m_out_transform_bn_in_kernel.setArg(7, *bufferResidual);
      } else {
        m_out_transform_bn_in_kernel.setArg(7, nullptr);
      }
      m_out_transform_bn_in_kernel.setArg(8, bn_weights[0]);
      m_out_transform_bn_in_kernel.setArg(9, bn_weights[1]);
      m_out_transform_bn_in_kernel.setArg(
          10, cl::Local(dim_size * width * height * sizeof(float)));

      m_commandqueue.enqueueNDRangeKernel(
          m_out_transform_bn_in_kernel, cl::NullRange,
          cl::NDRange(outputs, wgs, batch_size), cl::NDRange(dim_size, wgs, 1));
    } else {
      m_out_transform_bn_kernel.setArg(0, bufferM);
      m_out_transform_bn_kernel.setArg(1, bufferOut);
      m_out_transform_bn_kernel.setArg(2, outputs);
      m_out_transform_bn_kernel.setArg(3, m_ceil);
      m_out_transform_bn_kernel.setArg(4, n_ceil);
      if (bufferResidual) {
        m_out_transform_bn_kernel.setArg(5, *bufferResidual);
      } else {
        m_out_transform_bn_kernel.setArg(5, nullptr);
      }
      m_out_transform_bn_kernel.setArg(6, bn_weights[0]);
      m_out_transform_bn_kernel.setArg(7, bn_weights[1]);

      m_commandqueue.enqueueNDRangeKernel(
          m_out_transform_bn_kernel, cl::NullRange,
          cl::NDRange(outputs, wgs, batch_size));
    }
  } catch (const cl::Error& e) {
    CERR << "Error in convolve3: " << e.what() << ": " << e.err() << std::endl;
    throw;
  }
}

void OpenCLBuffers::convolve1(int channels, int outputs,
                              cl::Buffer& bufferInput, cl::Buffer& bufferOutput,
                              cl::Buffer& bufferMerge, weight_slice_t weights,
                              int batch_size) {
  // fixed for 8x8.
  constexpr int width = 8;
  constexpr int height = 8;
  constexpr int boardsize = width * height;
  constexpr int rowTiles = 8;

  // Input channel grouping in multiples of 8.
  constexpr int channelGroup = 8;
  constexpr int channelShift = 3;
  constexpr int rowGroup = 1;
  size_t outputGroup = std::min(outputs, 32);

#ifndef NDEBUG
  // Total output size after reducing.
  size_t outSize = width * height * outputs * sizeof(net_t);

  // Produce channel * output planes and merge them at the end.
  size_t mergeSize = (channels >> channelShift) * outSize;
  assert(mergeSize <= bufferMerge.getInfo<CL_MEM_SIZE>());
#endif

  // Copy the rows locally.
  size_t stripSize = width * sizeof(float);

  int rowBuffer = std::min<int>(channelGroup, 7);
  size_t rowSize = channelGroup * outputGroup * rowBuffer * sizeof(float);

  try {
    m_convolve1_kernel.setArg(0, bufferInput);
    m_convolve1_kernel.setArg(1, bufferMerge);
    m_convolve1_kernel.setArg(2, weights[0]);
    m_convolve1_kernel.setArg(3,
                              cl::Local(stripSize * channelGroup * rowGroup));
    m_convolve1_kernel.setArg(4, cl::Local(rowSize));

    m_commandqueue.enqueueNDRangeKernel(
        m_convolve1_kernel, cl::NullRange,
        cl::NDRange(channels, outputs, batch_size * rowTiles),
        cl::NDRange(channelGroup, outputGroup, rowGroup));
  } catch (const cl::Error& e) {
    CERR << "Error in convolve1: " << e.what() << ": " << e.err() << std::endl;
    throw;
  }

  assert(channels % (1 << channelShift) == 0);

  try {
    m_merge_kernel.setArg(0, bufferMerge);
    m_merge_kernel.setArg(1, bufferOutput);
    m_merge_kernel.setArg(2, channels >> channelShift);
    m_merge_kernel.setArg(3, weights[1]);
    m_merge_kernel.setArg(4, weights[2]);

    m_commandqueue.enqueueNDRangeKernel(
        m_merge_kernel, cl::NullRange,
        cl::NDRange(outputs, boardsize, batch_size),
        cl::NDRange(std::min(8, outputs), 8, 1));
  } catch (const cl::Error& e) {
    CERR << "Error in merge: " << e.what() << ": " << e.err() << std::endl;
    throw;
  }
}

void OpenCLBuffers::innerproduct(cl::Buffer& input, weight_slice_t weights,
                                 weight_slice_t biases, cl::Buffer& output,
                                 const int inputs, const int outputs,
                                 const int relu, int batch_size) {
  // TODO: Tune these.
  size_t wgs1 = 64;
  size_t wpt1 = 1;

  auto m_ceil = int(ceilMultiple(outputs, wgs1 * wpt1));
  auto global_size = m_ceil / wpt1;
  auto local_size = wgs1;

  try {
    // Sets the kernel arguments.
    m_sgemv_kernel.setArg(0, static_cast<int>(outputs));
    m_sgemv_kernel.setArg(1, static_cast<int>(inputs));
    m_sgemv_kernel.setArg(2, weights[0]);
    m_sgemv_kernel.setArg(3, static_cast<int>(0));
    m_sgemv_kernel.setArg(4, static_cast<int>(inputs));
    m_sgemv_kernel.setArg(5, input);
    m_sgemv_kernel.setArg(6, static_cast<int>(0));
    m_sgemv_kernel.setArg(7, output);
    m_sgemv_kernel.setArg(8, static_cast<int>(0));
    m_sgemv_kernel.setArg(9, biases[0]);
    m_sgemv_kernel.setArg(10, static_cast<int>(relu));

    m_commandqueue.enqueueNDRangeKernel(m_sgemv_kernel, cl::NullRange,
                                        cl::NDRange(global_size, batch_size),
                                        cl::NDRange(local_size, 1));
  } catch (const cl::Error& e) {
    CERR << "Error in innerproduct: " << e.what() << ": " << e.err()
         << std::endl;
    throw;
  }
}
