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

using net_t = float;

#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#if defined(_WIN32) && !defined(_WIN64)
#define CL_HPP_TARGET_OPENCL_VERSION 110
#else
#define CL_HPP_TARGET_OPENCL_VERSION 120
#endif
#define CL_HPP_ENABLE_EXCEPTIONS
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "cl2.hpp"
#include "neural/opencl/OpenCLBuffers.h"
#include "neural/opencl/OpenCLParams.h"

inline size_t ceilMultiple(size_t a, size_t b) {
  if (a % b == 0) return a;
  return a + (b - a % b);
}

static constexpr auto WINOGRAD_P = 8 * 8 / 4;
static constexpr auto WINOGRAD_TILE = 4 * 4;

class OpenCL;
class OpenCLBuffers;

class Layer {
  friend class OpenCL_Network;
  friend class OpenCLBuffers;

 private:
  unsigned int channels{0};
  unsigned int outputs{0};
  unsigned int se_fc_outputs{0};
  unsigned int filter_size{0};
  unsigned int ip_in_size{0};
  unsigned int ip_out_size{0};
  bool is_input_convolution{false};
  bool is_residual_block{false};
  bool is_se_unit{false};
  bool is_policy{false};
  bool is_conv_policy{false};
  bool is_value{false};
  bool is_moves_left{false};
  std::vector<cl::Buffer> weights;
};

class OpenCL_Network {
  friend class OpenCLBuffers;

 public:
  OpenCL_Network(OpenCL& opencl) : m_opencl(opencl), m_max_batch_size(1) {}

  std::unique_ptr<OpenCLBuffers> acquire_buffers() const;
  void release_buffers(std::unique_ptr<OpenCLBuffers>) const;

  OpenCL& getOpenCL() const { return m_opencl; }

  size_t getMaxMatchSize() const { return m_max_batch_size; }

  void setMaxMatchSize(size_t new_value) { m_max_batch_size = new_value; }

  void push_input_convolution(unsigned int filter_size, unsigned int channels,
                              unsigned int outputs,
                              const std::vector<float>& weights,
                              const std::vector<float>& biases) {
    size_t layer = get_layer_count();
    push_weights(layer, weights);
    push_weights(layer, biases);
    m_layers[layer].is_input_convolution = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].filter_size = filter_size;
    m_layers[layer].channels = channels;
  }

  void push_residual(unsigned int filter_size, unsigned int channels,
                     unsigned int outputs, const std::vector<float>& weights_1,
                     const std::vector<float>& biases_1,
                     const std::vector<float>& weights_2,
                     const std::vector<float>& biases_2) {
    size_t layer = get_layer_count();
    push_weights(layer, weights_1);
    push_weights(layer, biases_1);
    push_weights(layer, weights_2);
    push_weights(layer, biases_2);
    m_layers[layer].is_residual_block = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].filter_size = filter_size;
    m_layers[layer].channels = channels;
  }

  void push_se(unsigned int channels, unsigned int se_fc_outputs,
               const std::vector<float>& weights_1,
               const std::vector<float>& biases_1,
               const std::vector<float>& weights_2,
               const std::vector<float>& biases_2) {
    size_t layer = get_layer_count();
    push_weights(layer, weights_1);
    push_weights(layer, biases_1);
    push_weights(layer, weights_2);
    push_weights(layer, biases_2);
    m_layers[layer].is_se_unit = true;
    m_layers[layer].channels = channels;
    m_layers[layer].se_fc_outputs = se_fc_outputs;
    m_layers[layer].outputs = channels;
  }

  void push_policy(unsigned int channels, unsigned int outputs,
                   unsigned int ip_in, unsigned int ip_out,
                   const std::vector<float>& weights,
                   const std::vector<float>& biases,
                   const std::vector<float>& fc_w,
                   const std::vector<float>& fc_b) {
    size_t layer = get_layer_count();
    push_weights(layer, weights);
    push_weights(layer, biases);
    push_weights(layer, fc_w);
    push_weights(layer, fc_b);
    m_layers[layer].is_policy = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].channels = channels;
    m_layers[layer].ip_in_size = ip_in;
    m_layers[layer].ip_out_size = ip_out;
  }

  void push_conv_policy(unsigned int channels, unsigned int outputs,
                        unsigned int ip_in, unsigned int ip_out,
                        const std::vector<float>& weights_1,
                        const std::vector<float>& biases_1,
                        const std::vector<float>& weights_2,
                        const std::vector<float>& biases_2,
                        const std::vector<short>& indices) {
    size_t layer = get_layer_count();
    push_weights(layer, weights_1);
    push_weights(layer, biases_1);
    push_weights(layer, weights_2);
    push_weights(layer, biases_2);
    push_weights_short(layer, indices);
    m_layers[layer].is_conv_policy = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].channels = channels;
    m_layers[layer].ip_in_size = ip_in;
    m_layers[layer].ip_out_size = ip_out;
  }

  void push_value(unsigned int channels, unsigned int outputs,
                  unsigned int ip_in, unsigned int ip_out,
                  const std::vector<float>& weights,
                  const std::vector<float>& biases,
                  const std::vector<float>& fc_w,
                  const std::vector<float>& fc_b) {
    size_t layer = get_layer_count();
    push_weights(layer, weights);
    push_weights(layer, biases);
    push_weights(layer, fc_w);
    push_weights(layer, fc_b);
    m_layers[layer].is_value = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].channels = channels;
    m_layers[layer].ip_in_size = ip_in;
    m_layers[layer].ip_out_size = ip_out;
  }

  void push_moves_left(unsigned int channels, unsigned int outputs,
                       unsigned int ip_in, unsigned int ip_out,
                       const std::vector<float>& weights,
                       const std::vector<float>& biases,
                       const std::vector<float>& fc_w,
                       const std::vector<float>& fc_b) {
    size_t layer = get_layer_count();
    push_weights(layer, weights);
    push_weights(layer, biases);
    push_weights(layer, fc_w);
    push_weights(layer, fc_b);
    m_layers[layer].is_moves_left = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].channels = channels;
    m_layers[layer].ip_in_size = ip_in;
    m_layers[layer].ip_out_size = ip_out;
  }

  size_t get_layer_count() const { return m_layers.size(); }

 private:
  void push_weights(size_t layer, const std::vector<float>& weights) {
    add_weights(layer, weights.size(), weights.data());
  }
  void add_weights(size_t layer, size_t size, const float* weights);

  void push_weights_short(size_t layer, const std::vector<short>& weights) {
    add_weights_short(layer, weights.size(), weights.data());
  }
  void add_weights_short(size_t layer, size_t size, const short* weights);

  OpenCL& m_opencl;
  size_t m_max_batch_size;

  std::vector<Layer> m_layers;

  mutable std::mutex m_pool_mutex;
  mutable std::vector<std::unique_ptr<OpenCLBuffers>> m_buffers_pool;
};

class OpenCL {
  friend class OpenCL_Network;
  friend class OpenCLBuffers;
  friend class Tuner;

 public:
  void initialize(const int channels, const OpenCLParams& params);
  std::string get_device_name();

  std::vector<size_t> get_sgemm_tuners(void);

  cl::Device m_device;
  cl::Context m_context;

 private:
  void tune_sgemm(void);
  void process_tuners(std::string tuners);

  cl::Program m_program;
  std::string m_cl_args;

  struct sgemm_tuners {
    size_t mwg, nwg, kwg;
    size_t vwm, vwn;
    size_t mdimc, ndimc;
  };
  sgemm_tuners m_sgemm_tuners;
  size_t m_wavefront_size{0};
  size_t m_max_workgroup_size{0};
  std::vector<size_t> m_max_workgroup_dims;
  bool m_init_ok{false};
};

extern const std::string sourceCode_sgemm;
