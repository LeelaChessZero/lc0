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

#include "layers.h"
#include <cassert>
#include <cstring>
#include <vector>

namespace lczero {

namespace dnnl_backend {

BaseLayer::BaseLayer(int c, int h, int w, BaseLayer* ip)
    : input_(ip), C(c), H(h), W(w) {}

ConvLayer::ConvLayer(BaseLayer* ip, int C, int H, int W, int filter, int Cin,
                     bool relu, bool skip)
    : BaseLayer(C, H, W, ip),
      c_input_(Cin),
      filter_size_(filter),
      use_relu_(relu),
      use_skip_(skip) {}

void ConvLayer::LoadWeights(float* pfilter, float* pBias, dnnl::engine& eng) {
  auto filter_md = dnnl::memory::desc({C, c_input_, filter_size_, filter_size_},
                                      dnnl::memory::data_type::f32,
                                      dnnl::memory::format_tag::oihw);
  filter_mem = dnnl::memory(filter_md, eng);
  memcpy(filter_mem.get_data_handle(), pfilter,
         sizeof(float) * C * c_input_ * filter_size_ * filter_size_);

  auto bias_md = dnnl::memory::desc({C}, dnnl::memory::data_type::f32,
                                    dnnl::memory::format_tag::a);
  bias_mem = dnnl::memory(bias_md, eng);
  memcpy(bias_mem.get_data_handle(), pBias, sizeof(float) * C);
}

void ConvLayer::Eval(int N, dnnl::memory& output, dnnl::memory& input,
                     dnnl::engine& eng, dnnl::stream& stream) {
  if (last_batch_ != N) {
    auto t_in_md =
        dnnl::memory::desc({N, c_input_, H, W}, dnnl::memory::data_type::f32,
                           dnnl::memory::format_tag::any);

    auto t_filter_md = dnnl::memory::desc(
        {C, c_input_, filter_size_, filter_size_}, dnnl::memory::data_type::f32,
        dnnl::memory::format_tag::any);

    auto t_out_md =
        dnnl::memory::desc({N, C, H, W}, dnnl::memory::data_type::f32,
                           dnnl::memory::format_tag::any);

    const int padding = filter_size_ / 2;
    auto conv_d = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_auto,
        t_in_md, t_filter_md, bias_mem.get_desc(), t_out_md, {1, 1},
        {padding, padding}, {padding, padding});
    dnnl::post_ops conv_ops;
    if (use_skip_) {
      conv_ops.append_sum();
    }
    if (use_relu_) {
      conv_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    }
    dnnl::primitive_attr conv_attr;
    conv_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    conv_attr.set_post_ops(conv_ops);
    auto conv_pd =
        dnnl::convolution_forward::primitive_desc(conv_d, conv_attr, eng);
    conv_ = dnnl::convolution_forward(conv_pd);
    scratchpad_mem = dnnl::memory(conv_pd.scratchpad_desc(), eng);

    in_md = conv_pd.src_desc();
    out_md = conv_pd.dst_desc();
    if (conv_pd.weights_desc() != filter_mem.get_desc()) {
      auto tmp = dnnl::memory(conv_pd.weights_desc(), eng);
      dnnl::reorder(filter_mem, tmp).execute(stream, filter_mem, tmp);
      filter_mem = tmp;
    }

    last_batch_ = N;
  }

  auto conv_input = input;
  auto conv_output = output;

  if (in_md != input.get_desc()) {
    conv_input = dnnl::memory(in_md, eng);
    dnnl::reorder(input, conv_input).execute(stream, input, conv_input);
  }

  if (out_md != output.get_desc()) {
    conv_output = dnnl::memory(out_md, eng);
    if (use_skip_) {
      dnnl::reorder(output, conv_output).execute(stream, output, conv_output);
    }
  }

  conv_.execute(stream, {{DNNL_ARG_SRC, conv_input},
                         {DNNL_ARG_WEIGHTS, filter_mem},
                         {DNNL_ARG_BIAS, bias_mem},
                         {DNNL_ARG_DST, conv_output},
                         {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});

  output = conv_output;
}

SELayer::SELayer(BaseLayer* ip, int fc1Outputs)
    : BaseLayer(ip->GetC(), ip->GetH(), ip->GetW(), ip),
      numFc1Out_(fc1Outputs) {}

void SELayer::LoadWeights(float* w1, float* b1, float* w2, float* b2,
                          dnnl::engine& eng) {
  const size_t weight_size = sizeof(float) * C * numFc1Out_;

  auto filter_md =
      dnnl::memory::desc({numFc1Out_, C}, dnnl::memory::data_type::f32,
                         dnnl::memory::format_tag::ab);
  filter_mem = dnnl::memory(filter_md, eng);
  memcpy(filter_mem.get_data_handle(), w1, weight_size);

  auto bias_md = dnnl::memory::desc({numFc1Out_}, dnnl::memory::data_type::f32,
                                    dnnl::memory::format_tag::a);
  bias_mem = dnnl::memory(bias_md, eng);
  memcpy(bias_mem.get_data_handle(), b1, numFc1Out_ * sizeof(float));

  auto filter2a_md =
      dnnl::memory::desc({C, numFc1Out_}, dnnl::memory::data_type::f32,
                         dnnl::memory::format_tag::ab);
  filter2a_mem = dnnl::memory(filter2a_md, eng);
  memcpy(filter2a_mem.get_data_handle(), w2, weight_size);

  auto filter2b_md =
      dnnl::memory::desc({C, numFc1Out_}, dnnl::memory::data_type::f32,
                         dnnl::memory::format_tag::ab);
  filter2b_mem = dnnl::memory(filter2b_md, eng);
  memcpy(filter2b_mem.get_data_handle(), w2 + C * numFc1Out_, weight_size);

  auto bias2a_md = dnnl::memory::desc({C}, dnnl::memory::data_type::f32,
                                      dnnl::memory::format_tag::a);
  bias2a_mem = dnnl::memory(bias2a_md, eng);
  memcpy(bias2a_mem.get_data_handle(), b2, C * sizeof(float));

  auto bias2b_md = dnnl::memory::desc({C}, dnnl::memory::data_type::f32,
                                      dnnl::memory::format_tag::a);
  bias2b_mem = dnnl::memory(bias2b_md, eng);
  memcpy(bias2b_mem.get_data_handle(), b2 + C, C * sizeof(float));
}

void SELayer::Eval(int N, dnnl::memory& output, dnnl::memory& input,
                   dnnl::engine& eng, dnnl::stream& stream) {
  auto in_md = dnnl::memory::desc({N, C, H, W}, dnnl::memory::data_type::f32,
                                  dnnl::memory::format_tag::nhwc);

  auto conv_input = input;
  if (in_md != input.get_desc()) {
    conv_input = dnnl::memory(in_md, eng);
    dnnl::reorder(input, conv_input).execute(stream, input, conv_input);
  }

  if (last_batch_ != N) {
    auto t_out_md =
        dnnl::memory::desc({N, C, H, W}, dnnl::memory::data_type::f32,
                           dnnl::memory::format_tag::any);

    auto t_pool_out_md =
        dnnl::memory::desc({N, C, 1, 1}, dnnl::memory::data_type::f32,
                           dnnl::memory::format_tag::any);

    auto t_fc1_out_md =
        dnnl::memory::desc({N, numFc1Out_}, dnnl::memory::data_type::f32,
                           dnnl::memory::format_tag::any);

    auto t_fc2a_out_md = dnnl::memory::desc(
        {N, C}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

    auto t_fc2b_out_md = dnnl::memory::desc(
        {N, C}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

    auto t_filter_md =
        dnnl::memory::desc({numFc1Out_, C}, dnnl::memory::data_type::f32,
                           dnnl::memory::format_tag::any);

    auto t_filter2a_md =
        dnnl::memory::desc({C, numFc1Out_}, dnnl::memory::data_type::f32,
                           dnnl::memory::format_tag::any);

    auto t_filter2b_md =
        dnnl::memory::desc({C, numFc1Out_}, dnnl::memory::data_type::f32,
                           dnnl::memory::format_tag::any);

    auto pooling_d = dnnl::pooling_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::pooling_avg,
        conv_input.get_desc(), t_pool_out_md, {1, 1}, {H, W}, {0, 0}, {0, 0});
    dnnl::primitive_attr pooling_attr;
    pooling_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto pooling_pd =
        dnnl::pooling_forward::primitive_desc(pooling_d, pooling_attr, eng);
    if (pooling_pd.scratchpad_desc().get_size() > 0) {
      throw Exception("SELayer does not support scparchpad memory");
    }
    pooling_ = dnnl::pooling_forward(pooling_pd);

    pool_out_md = pooling_pd.dst_desc();

    auto fc_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, pool_out_md.reshape({N, C}),
        t_filter_md, bias_mem.get_desc(), t_fc1_out_md);
    dnnl::post_ops fc_ops;
    fc_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    dnnl::primitive_attr fc_attr;
    fc_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    fc_attr.set_post_ops(fc_ops);
    auto fc_pd =
        dnnl::inner_product_forward::primitive_desc(fc_d, fc_attr, eng);
    if (fc_pd.scratchpad_desc().get_size() > 0) {
      throw Exception("SELayer does not support scparchpad memory");
    }
    fc_ = dnnl::inner_product_forward(fc_pd);

    fc1_out_md = fc_pd.dst_desc();
    if (fc_pd.weights_desc() != filter_mem.get_desc()) {
      auto tmp = dnnl::memory(fc_pd.weights_desc(), eng);
      dnnl::reorder(filter_mem, tmp).execute(stream, filter_mem, tmp);
      filter_mem = tmp;
    }

    auto fc2a_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, fc1_out_md, t_filter2a_md,
        bias2a_mem.get_desc(), t_fc2a_out_md);
    dnnl::post_ops fc2a_ops;
    fc2a_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_logistic, 0.0f,
                            0.0f);
    dnnl::primitive_attr fc2a_attr;
    fc2a_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    fc2a_attr.set_post_ops(fc2a_ops);
    auto fc2a_pd =
        dnnl::inner_product_forward::primitive_desc(fc2a_d, fc2a_attr, eng);
    if (fc2a_pd.scratchpad_desc().get_size() > 0) {
      throw Exception("SELayer does not support scparchpad memory");
    }
    fc2a_ = dnnl::inner_product_forward(fc2a_pd);

    fc2a_out_md = fc2a_pd.dst_desc();
    if (fc2a_pd.weights_desc() != filter2a_mem.get_desc()) {
      auto tmp = dnnl::memory(fc2a_pd.weights_desc(), eng);
      dnnl::reorder(filter2a_mem, tmp).execute(stream, filter2a_mem, tmp);
      filter2a_mem = tmp;
    }

    auto fc2b_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, fc1_out_md, t_filter2b_md,
        bias2b_mem.get_desc(), t_fc2b_out_md);
    dnnl::primitive_attr fc2b_attr;
    fc2b_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto fc2b_pd =
        dnnl::inner_product_forward::primitive_desc(fc2b_d, fc2b_attr, eng);
    if (fc2b_pd.scratchpad_desc().get_size() > 0) {
      throw Exception("SELayer does not support scparchpad memory");
    }
    fc2b_ = dnnl::inner_product_forward(fc2b_pd);

    fc2b_out_md = fc2b_pd.dst_desc();
    if (fc2b_pd.weights_desc() != filter2b_mem.get_desc()) {
      auto tmp = dnnl::memory(fc2b_pd.weights_desc(), eng);
      dnnl::reorder(filter2b_mem, tmp).execute(stream, filter2b_mem, tmp);
      filter2b_mem = tmp;
    }

    auto mul_d =
        dnnl::binary::desc(dnnl::algorithm::binary_mul, conv_input.get_desc(),
                           fc2a_out_md.reshape({N, C, 1, 1}), t_out_md);
    dnnl::post_ops mul_ops;
    mul_ops.append_sum();
    dnnl::primitive_attr mul_attr;
    mul_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    mul_attr.set_post_ops(mul_ops);
    auto mul_pd = dnnl::binary::primitive_desc(mul_d, mul_attr, eng);
    if (mul_pd.scratchpad_desc().get_size() > 0) {
      throw Exception("SELayer does not support scparchpad memory");
    }
    mul_ = dnnl::binary(mul_pd);
    out_md = mul_pd.dst_desc();

    auto add_d = dnnl::binary::desc(dnnl::algorithm::binary_add, out_md,
                                    fc2b_out_md.reshape({N, C, 1, 1}), out_md);
    dnnl::post_ops add_ops;
    add_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    dnnl::primitive_attr add_attr;
    add_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    add_attr.set_post_ops(add_ops);
    auto add_pd = dnnl::binary::primitive_desc(add_d, add_attr, eng);
    if (add_pd.scratchpad_desc().get_size() > 0) {
      throw Exception("SELayer does not support scparchpad memory");
    }
    add_ = dnnl::binary(add_pd);

    last_batch_ = N;
  }

  auto conv_output = output;

  if (out_md != output.get_desc()) {
    conv_output = dnnl::memory(out_md, eng);
    dnnl::reorder(output, conv_output).execute(stream, output, conv_output);
  }

  auto pool_out_mem = dnnl::memory(pool_out_md, eng);
  auto fc1_out_mem = dnnl::memory(fc1_out_md, eng);
  auto fc2a_out_mem = dnnl::memory(fc2a_out_md, eng);
  auto fc2b_out_mem = dnnl::memory(fc2b_out_md, eng);

  pooling_.execute(stream,
                   {{DNNL_ARG_SRC, conv_input}, {DNNL_ARG_DST, pool_out_mem}});

  fc_.execute(stream, {{DNNL_ARG_SRC, pool_out_mem},
                       {DNNL_ARG_WEIGHTS, filter_mem},
                       {DNNL_ARG_BIAS, bias_mem},
                       {DNNL_ARG_DST, fc1_out_mem}});

  fc2a_.execute(stream, {{DNNL_ARG_SRC, fc1_out_mem},
                         {DNNL_ARG_WEIGHTS, filter2a_mem},
                         {DNNL_ARG_BIAS, bias2a_mem},
                         {DNNL_ARG_DST, fc2a_out_mem}});

  fc2b_.execute(stream, {{DNNL_ARG_SRC, fc1_out_mem},
                         {DNNL_ARG_WEIGHTS, filter2b_mem},
                         {DNNL_ARG_BIAS, bias2b_mem},
                         {DNNL_ARG_DST, fc2b_out_mem}});

  mul_.execute(stream, {{DNNL_ARG_SRC_0, conv_input},
                        {DNNL_ARG_SRC_1, fc2a_out_mem},
                        {DNNL_ARG_DST, conv_output}});

  add_.execute(stream, {{DNNL_ARG_SRC_0, conv_output},
                        {DNNL_ARG_SRC_1, fc2b_out_mem},
                        {DNNL_ARG_DST, conv_output}});

  output = conv_output;
}

FCLayer::FCLayer(BaseLayer* ip, int C, int H, int W, bool relu, bool tanh)
    : BaseLayer(C, H, W, ip), use_relu_(relu), use_tanh_(tanh) {}

void FCLayer::LoadWeights(float* cpuWeight, float* cpuBias, dnnl::engine& eng) {
  const int num_outputs = C * H * W;
  const int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();
  const size_t weight_size = sizeof(float) * num_inputs * num_outputs;
  const size_t bias_size = sizeof(float) * num_outputs;

  auto filter_md = dnnl::memory::desc({num_outputs, num_inputs},
                                      dnnl::memory::data_type::f32,
                                      dnnl::memory::format_tag::ab);
  filter_mem = dnnl::memory(filter_md, eng);
  memcpy(filter_mem.get_data_handle(), cpuWeight, weight_size);

  auto bias_md = dnnl::memory::desc({num_outputs}, dnnl::memory::data_type::f32,
                                    dnnl::memory::format_tag::a);
  bias_mem = dnnl::memory(bias_md, eng);
  memcpy(bias_mem.get_data_handle(), cpuBias, bias_size);
}

void FCLayer::Eval(int N, dnnl::memory& output, dnnl::memory& input,
                   dnnl::engine& eng, dnnl::stream& stream) {
  if (last_batch_ != N) {
    const int num_outputs = C * H * W;
    const int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

    auto t_in_md = dnnl::memory::desc(
        {N, input_->GetC(), input_->GetH(), input_->GetW()},
        dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

    auto t_filter_md = dnnl::memory::desc({num_outputs, num_inputs},
                                          dnnl::memory::data_type::f32,
                                          dnnl::memory::format_tag::any);

    auto t_out_md =
        dnnl::memory::desc({N, C, H, W}, dnnl::memory::data_type::f32,
                           dnnl::memory::format_tag::any);
    auto fc_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, t_in_md.reshape({N, num_inputs}),
        t_filter_md, bias_mem.get_desc(), t_out_md.reshape({N, num_outputs}));
    dnnl::post_ops fc_ops;
    if (use_relu_) {
      fc_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    }
    if (use_tanh_) {
      fc_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_tanh, 0.0f, 0.0f);
    }
    dnnl::primitive_attr fc_attr;
    fc_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    fc_attr.set_post_ops(fc_ops);
    auto fc_pd =
        dnnl::inner_product_forward::primitive_desc(fc_d, fc_attr, eng);
    if (fc_pd.scratchpad_desc().get_size() > 0) {
      throw Exception("FClayer does not support scparchpad memory");
    }
    fc_ = dnnl::inner_product_forward(fc_pd);

    in_md = fc_pd.src_desc().reshape(
        {N, input_->GetC(), input_->GetH(), input_->GetW()});
    out_md = fc_pd.dst_desc().reshape({N, C, H, W});
    if (fc_pd.weights_desc() != filter_mem.get_desc()) {
      auto tmp = dnnl::memory(fc_pd.weights_desc(), eng);
      dnnl::reorder(filter_mem, tmp).execute(stream, filter_mem, tmp);
      filter_mem = tmp;
    }

    last_batch_ = N;
  }

  auto conv_input = input;
  auto conv_output = output;

  if (in_md != input.get_desc()) {
    conv_input = dnnl::memory(in_md, eng);
    dnnl::reorder(input, conv_input).execute(stream, input, conv_input);
  }

  if (out_md != output.get_desc()) {
    conv_output = dnnl::memory(out_md, eng);
  }

  fc_.execute(stream, {{DNNL_ARG_SRC, conv_input},
                       {DNNL_ARG_WEIGHTS, filter_mem},
                       {DNNL_ARG_BIAS, bias_mem},
                       {DNNL_ARG_DST, conv_output}});

  output = conv_output;
}

}  // namespace dnnl_backend
}  // namespace lczero
