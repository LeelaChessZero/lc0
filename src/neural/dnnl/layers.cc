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
    : input_(ip), C(c), H(h), W(w) {
  if (ip) {
    data_type_ = ip->data_type_;
  } else {
    data_type_ = dnnl::memory::data_type::undef;
  }
}

ConvLayer::ConvLayer(BaseLayer* ip, int C, int H, int W, int filter, int Cin,
                     bool relu, bool skip)
    : BaseLayer(C, H, W, ip),
      c_input_(Cin),
      filter_size_(filter),
      use_relu_(relu),
      use_skip_(skip) {}

void ConvLayer::LoadWeights(float* pfilter, float* pBias, dnnl::engine& eng,
                            dnnl::stream& stream) {
  dnnl::engine cpu_eng;
  if (eng.get_kind() == dnnl::engine::kind::cpu) {
    cpu_eng = eng;
  } else {
    cpu_eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
  }

  auto t_filter_md = dnnl::memory::desc(
      {C, c_input_, filter_size_, filter_size_}, dnnl::memory::data_type::f32,
      dnnl::memory::format_tag::oihw);
  auto t_filter_mem = dnnl::memory(t_filter_md, cpu_eng, pfilter);
  auto filter_md =
      dnnl::memory::desc({C, c_input_, filter_size_, filter_size_}, data_type_,
                         dnnl::memory::format_tag::oihw);
  filter_mem = dnnl::memory(filter_md, eng);
  dnnl::reorder(t_filter_mem, filter_mem)
      .execute(stream, t_filter_mem, filter_mem);

  auto t_bias_md = dnnl::memory::desc({C}, dnnl::memory::data_type::f32,
                                      dnnl::memory::format_tag::a);
  auto t_bias_mem = dnnl::memory(t_bias_md, cpu_eng, pBias);
  auto bias_md =
      dnnl::memory::desc({C}, data_type_, dnnl::memory::format_tag::a);
  bias_mem = dnnl::memory(bias_md, eng);
  dnnl::reorder(t_bias_mem, bias_mem).execute(stream, t_bias_mem, bias_mem);
}

void ConvLayer::Eval(int N, dnnl::memory& output, dnnl::memory& input,
                     dnnl::engine& eng, dnnl::stream& stream) {
  if (last_batch_ != N) {
    auto t_in_md = dnnl::memory::desc({N, c_input_, H, W}, data_type_,
                                      dnnl::memory::format_tag::any);

    auto t_filter_md =
        dnnl::memory::desc({C, c_input_, filter_size_, filter_size_},
                           data_type_, dnnl::memory::format_tag::any);

    auto t_out_md = dnnl::memory::desc({N, C, H, W}, data_type_,
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
    if (!conv_filter_mem ||
        conv_pd.weights_desc() != conv_filter_mem.get_desc()) {
      // This may be a transformation for Winograd convolution, so keep the
      // original weights.
      conv_filter_mem = dnnl::memory(conv_pd.weights_desc(), eng);
      dnnl::reorder(filter_mem, conv_filter_mem)
          .execute(stream, filter_mem, conv_filter_mem);
    }

    auto in_reorder_pd =
        dnnl::reorder::primitive_desc(eng, input.get_desc(), eng, in_md);
    in_reorder_ = dnnl::reorder(in_reorder_pd);

    if (use_skip_) {
      auto skip_reorder_pd =
          dnnl::reorder::primitive_desc(eng, output.get_desc(), eng, out_md);
      skip_reorder_ = dnnl::reorder(skip_reorder_pd);
    }

    last_batch_ = N;
  }

  if (in_md != input.get_desc()) {
    auto tmp = dnnl::memory(in_md, eng);
    in_reorder_.execute(stream, input, tmp);
    input = tmp;
  }

  if (!output || out_md != output.get_desc()) {
    if (use_skip_) {
      auto tmp = dnnl::memory(out_md, eng);
      skip_reorder_.execute(stream, output, tmp);
      output = tmp;
    } else {
      output = dnnl::memory(out_md, eng);
    }
  }

  conv_.execute(stream, {{DNNL_ARG_SRC, input},
                         {DNNL_ARG_WEIGHTS, conv_filter_mem},
                         {DNNL_ARG_BIAS, bias_mem},
                         {DNNL_ARG_DST, output},
                         {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});
}

SELayer::SELayer(BaseLayer* ip, int fc1Outputs)
    : BaseLayer(ip->GetC(), ip->GetH(), ip->GetW(), ip),
      numFc1Out_(fc1Outputs) {}

void SELayer::LoadWeights(float* w1, float* b1, float* w2, float* b2,
                          dnnl::engine& eng, dnnl::stream& stream) {
  dnnl::engine cpu_eng;
  if (eng.get_kind() == dnnl::engine::kind::cpu) {
    cpu_eng = eng;
  } else {
    cpu_eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
  }

  auto t_filter_md =
      dnnl::memory::desc({numFc1Out_, C}, dnnl::memory::data_type::f32,
                         dnnl::memory::format_tag::ab);
  auto t_filter_mem = dnnl::memory(t_filter_md, cpu_eng, w1);
  auto filter_md = dnnl::memory::desc({numFc1Out_, C}, data_type_,
                                      dnnl::memory::format_tag::ab);
  filter_mem = dnnl::memory(filter_md, eng);
  dnnl::reorder(t_filter_mem, filter_mem)
      .execute(stream, t_filter_mem, filter_mem);

  auto t_bias_md = dnnl::memory::desc(
      {numFc1Out_}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::a);
  auto t_bias_mem = dnnl::memory(t_bias_md, cpu_eng, b1);
  auto bias_md =
      dnnl::memory::desc({numFc1Out_}, data_type_, dnnl::memory::format_tag::a);
  bias_mem = dnnl::memory(bias_md, eng);
  dnnl::reorder(t_bias_mem, bias_mem).execute(stream, t_bias_mem, bias_mem);

  auto t_filter2a_md =
      dnnl::memory::desc({C, numFc1Out_}, dnnl::memory::data_type::f32,
                         dnnl::memory::format_tag::ab);
  auto t_filter2a_mem = dnnl::memory(t_filter2a_md, cpu_eng, w2);
  auto filter2a_md = dnnl::memory::desc({C, numFc1Out_}, data_type_,
                                        dnnl::memory::format_tag::ab);
  filter2a_mem = dnnl::memory(filter2a_md, eng);
  dnnl::reorder(t_filter2a_mem, filter2a_mem)
      .execute(stream, t_filter2a_mem, filter2a_mem);

  auto t_filter2b_md =
      dnnl::memory::desc({C, numFc1Out_}, dnnl::memory::data_type::f32,
                         dnnl::memory::format_tag::ab);
  auto t_filter2b_mem =
      dnnl::memory(t_filter2b_md, cpu_eng, w2 + C * numFc1Out_);
  auto filter2b_md = dnnl::memory::desc({C, numFc1Out_}, data_type_,
                                        dnnl::memory::format_tag::ab);
  filter2b_mem = dnnl::memory(filter2b_md, eng);
  dnnl::reorder(t_filter2b_mem, filter2b_mem)
      .execute(stream, t_filter2b_mem, filter2b_mem);

  auto t_bias2a_md = dnnl::memory::desc({C}, dnnl::memory::data_type::f32,
                                        dnnl::memory::format_tag::a);
  auto t_bias2a_mem = dnnl::memory(t_bias2a_md, cpu_eng, b2);
  auto bias2a_md =
      dnnl::memory::desc({C}, data_type_, dnnl::memory::format_tag::a);
  bias2a_mem = dnnl::memory(bias2a_md, eng);
  dnnl::reorder(t_bias2a_mem, bias2a_mem)
      .execute(stream, t_bias2a_mem, bias2a_mem);

  auto t_bias2b_md = dnnl::memory::desc({C}, dnnl::memory::data_type::f32,
                                        dnnl::memory::format_tag::a);
  auto t_bias2b_mem = dnnl::memory(t_bias2b_md, cpu_eng, b2 + C);
  auto bias2b_md =
      dnnl::memory::desc({C}, data_type_, dnnl::memory::format_tag::a);
  bias2b_mem = dnnl::memory(bias2b_md, eng);
  dnnl::reorder(t_bias2b_mem, bias2b_mem)
      .execute(stream, t_bias2b_mem, bias2b_mem);
}

void SELayer::Eval(int N, dnnl::memory& output, dnnl::memory& input,
                   dnnl::engine& eng, dnnl::stream& stream) {
  if (last_batch_ != N) {
    // Also the broadcast input memory format for the binary primitives.
    auto t_pool_out_md = dnnl::memory::desc({N, C, 1, 1}, data_type_,
                                            dnnl::memory::format_tag::any);

    // Also the output memory format for the fc2 inner products.
    auto t_fc1_in_md =
        dnnl::memory::desc({N, C}, data_type_, dnnl::memory::format_tag::any);

    auto t_fc1_out_md = dnnl::memory::desc({N, numFc1Out_}, data_type_,
                                           dnnl::memory::format_tag::any);

    auto t_filter_md = dnnl::memory::desc({numFc1Out_, C}, data_type_,
                                          dnnl::memory::format_tag::any);

    auto t_filter2_md = dnnl::memory::desc({C, numFc1Out_}, data_type_,
                                           dnnl::memory::format_tag::any);

    auto pooling_d = dnnl::pooling_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::pooling_avg,
        input.get_desc(), t_pool_out_md, {1, 1}, {H, W}, {0, 0}, {0, 0});
    dnnl::primitive_attr pooling_attr;
    pooling_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto pooling_pd =
        dnnl::pooling_forward::primitive_desc(pooling_d, pooling_attr, eng);
    pooling_scratchpad_mem = dnnl::memory(pooling_pd.scratchpad_desc(), eng);
    pooling_ = dnnl::pooling_forward(pooling_pd);

    // This is also the optimized memory format descriptor for the binary
    // primitives.
    pool_out_md = pooling_pd.dst_desc();

    auto fc_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, t_fc1_in_md, t_filter_md,
        bias_mem.get_desc(), t_fc1_out_md);
    dnnl::post_ops fc_ops;
    fc_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    dnnl::primitive_attr fc_attr;
    fc_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    fc_attr.set_post_ops(fc_ops);
    auto fc_pd =
        dnnl::inner_product_forward::primitive_desc(fc_d, fc_attr, eng);
    fc_scratchpad_mem = dnnl::memory(fc_pd.scratchpad_desc(), eng);
    fc_ = dnnl::inner_product_forward(fc_pd);

    fc1_in_md = fc_pd.src_desc().reshape({N, C, 1, 1});
    fc1_out_md = fc_pd.dst_desc();
    if (fc_pd.weights_desc() != filter_mem.get_desc()) {
      auto tmp = dnnl::memory(fc_pd.weights_desc(), eng);
      dnnl::reorder(filter_mem, tmp).execute(stream, filter_mem, tmp);
      filter_mem = tmp;
    }

    auto fc2a_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, fc1_out_md, t_filter2_md,
        bias2a_mem.get_desc(), t_fc1_in_md);
    dnnl::post_ops fc2a_ops;
    fc2a_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_logistic, 0.0f,
                            0.0f);
    dnnl::primitive_attr fc2a_attr;
    fc2a_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    fc2a_attr.set_post_ops(fc2a_ops);
    auto fc2a_pd =
        dnnl::inner_product_forward::primitive_desc(fc2a_d, fc2a_attr, eng);
    fc2a_scratchpad_mem = dnnl::memory(fc2a_pd.scratchpad_desc(), eng);
    fc2a_ = dnnl::inner_product_forward(fc2a_pd);

    fc2a_out_md = fc2a_pd.dst_desc().reshape({N, C, 1, 1});
    if (fc2a_pd.weights_desc() != filter2a_mem.get_desc()) {
      auto tmp = dnnl::memory(fc2a_pd.weights_desc(), eng);
      dnnl::reorder(filter2a_mem, tmp).execute(stream, filter2a_mem, tmp);
      filter2a_mem = tmp;
    }

    auto fc2b_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, fc1_out_md, t_filter2_md,
        bias2b_mem.get_desc(), t_fc1_in_md);
    dnnl::primitive_attr fc2b_attr;
    fc2b_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto fc2b_pd =
        dnnl::inner_product_forward::primitive_desc(fc2b_d, fc2b_attr, eng);
    fc2b_scratchpad_mem = dnnl::memory(fc2b_pd.scratchpad_desc(), eng);
    fc2b_ = dnnl::inner_product_forward(fc2b_pd);

    fc2b_out_md = fc2b_pd.dst_desc().reshape({N, C, 1, 1});
    if (fc2b_pd.weights_desc() != filter2b_mem.get_desc()) {
      auto tmp = dnnl::memory(fc2b_pd.weights_desc(), eng);
      dnnl::reorder(filter2b_mem, tmp).execute(stream, filter2b_mem, tmp);
      filter2b_mem = tmp;
    }

    auto mul_d =
        dnnl::binary::desc(dnnl::algorithm::binary_mul, input.get_desc(),
                           pool_out_md, output.get_desc());
    dnnl::post_ops mul_ops;
    mul_ops.append_sum();
    dnnl::primitive_attr mul_attr;
    mul_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    mul_attr.set_post_ops(mul_ops);
    auto mul_pd = dnnl::binary::primitive_desc(mul_d, mul_attr, eng);
    mul_scratchpad_mem = dnnl::memory(mul_pd.scratchpad_desc(), eng);
    mul_ = dnnl::binary(mul_pd);

    auto add_d =
        dnnl::binary::desc(dnnl::algorithm::binary_add, output.get_desc(),
                           pool_out_md, output.get_desc());
    dnnl::post_ops add_ops;
    add_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    dnnl::primitive_attr add_attr;
    add_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    add_attr.set_post_ops(add_ops);
    auto add_pd = dnnl::binary::primitive_desc(add_d, add_attr, eng);
    add_scratchpad_mem = dnnl::memory(add_pd.scratchpad_desc(), eng);
    add_ = dnnl::binary(add_pd);

    auto fc1_reorder_pd =
        dnnl::reorder::primitive_desc(eng, pool_out_md, eng, fc1_in_md);
    fc1_reorder_ = dnnl::reorder(fc1_reorder_pd);

    auto mul_reorder_pd =
        dnnl::reorder::primitive_desc(eng, fc2a_out_md, eng, pool_out_md);
    mul_reorder_ = dnnl::reorder(mul_reorder_pd);

    auto add_reorder_pd =
        dnnl::reorder::primitive_desc(eng, fc2b_out_md, eng, pool_out_md);
    add_reorder_ = dnnl::reorder(add_reorder_pd);

    last_batch_ = N;
  }

  auto pool_out_mem = dnnl::memory(pool_out_md, eng);
  auto fc1_out_mem = dnnl::memory(fc1_out_md, eng);
  auto fc2a_out_mem = dnnl::memory(fc2a_out_md, eng);
  auto fc2b_out_mem = dnnl::memory(fc2b_out_md, eng);

  pooling_.execute(stream, {{DNNL_ARG_SRC, input},
                            {DNNL_ARG_DST, pool_out_mem},
                            {DNNL_ARG_SCRATCHPAD, pooling_scratchpad_mem}});

  dnnl::memory fc1_in_mem;
  if (fc1_in_md != pool_out_md) {
    fc1_in_mem = dnnl::memory(fc1_in_md, eng);
    fc1_reorder_.execute(stream, pool_out_mem, fc1_in_mem);
  } else {
    fc1_in_mem = pool_out_mem;
  }

  fc_.execute(stream, {{DNNL_ARG_SRC, fc1_in_mem},
                       {DNNL_ARG_WEIGHTS, filter_mem},
                       {DNNL_ARG_BIAS, bias_mem},
                       {DNNL_ARG_DST, fc1_out_mem},
                       {DNNL_ARG_SCRATCHPAD, fc_scratchpad_mem}});

  fc2a_.execute(stream, {{DNNL_ARG_SRC, fc1_out_mem},
                         {DNNL_ARG_WEIGHTS, filter2a_mem},
                         {DNNL_ARG_BIAS, bias2a_mem},
                         {DNNL_ARG_DST, fc2a_out_mem},
                         {DNNL_ARG_SCRATCHPAD, fc2a_scratchpad_mem}});

  fc2b_.execute(stream, {{DNNL_ARG_SRC, fc1_out_mem},
                         {DNNL_ARG_WEIGHTS, filter2b_mem},
                         {DNNL_ARG_BIAS, bias2b_mem},
                         {DNNL_ARG_DST, fc2b_out_mem},
                         {DNNL_ARG_SCRATCHPAD, fc2b_scratchpad_mem}});

  dnnl::memory mul_in_mem;
  if (pool_out_md != fc2a_out_md) {
    mul_in_mem = dnnl::memory(pool_out_md, eng);
    mul_reorder_.execute(stream, fc2a_out_mem, mul_in_mem);
  } else {
    mul_in_mem = fc2a_out_mem;
  }

  mul_.execute(stream, {{DNNL_ARG_SRC_0, input},
                        {DNNL_ARG_SRC_1, mul_in_mem},
                        {DNNL_ARG_DST, output},
                        {DNNL_ARG_SCRATCHPAD, mul_scratchpad_mem}});

  dnnl::memory add_in_mem;
  if (pool_out_md != fc2b_out_md) {
    add_in_mem = dnnl::memory(pool_out_md, eng);
    add_reorder_.execute(stream, fc2b_out_mem, add_in_mem);
  } else {
    add_in_mem = fc2b_out_mem;
  }

  add_.execute(stream, {{DNNL_ARG_SRC_0, output},
                        {DNNL_ARG_SRC_1, add_in_mem},
                        {DNNL_ARG_DST, output},
                        {DNNL_ARG_SCRATCHPAD, add_scratchpad_mem}});
}

FCLayer::FCLayer(BaseLayer* ip, int C, int H, int W, bool relu, bool tanh)
    : BaseLayer(C, H, W, ip), use_relu_(relu), use_tanh_(tanh) {}

void FCLayer::LoadWeights(float* cpuWeight, float* cpuBias, dnnl::engine& eng,
                          dnnl::stream& stream) {
  dnnl::engine cpu_eng;
  if (eng.get_kind() == dnnl::engine::kind::cpu) {
    cpu_eng = eng;
  } else {
    cpu_eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
  }

  const int num_outputs = C * H * W;

  auto t_filter_md = dnnl::memory::desc(
      {num_outputs, input_->GetC(), input_->GetH(), input_->GetW()},
      dnnl::memory::data_type::f32, dnnl::memory::format_tag::abcd);
  auto t_filter_mem = dnnl::memory(t_filter_md, cpu_eng, cpuWeight);
  auto filter_md = dnnl::memory::desc(
      {num_outputs, input_->GetC(), input_->GetH(), input_->GetW()}, data_type_,
      dnnl::memory::format_tag::abcd);
  filter_mem = dnnl::memory(filter_md, eng);
  dnnl::reorder(t_filter_mem, filter_mem)
      .execute(stream, t_filter_mem, filter_mem);

  auto t_bias_md = dnnl::memory::desc(
      {num_outputs}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::a);
  auto t_bias_mem = dnnl::memory(t_bias_md, cpu_eng, cpuBias);
  auto bias_md = dnnl::memory::desc({num_outputs}, data_type_,
                                    dnnl::memory::format_tag::a);
  bias_mem = dnnl::memory(bias_md, eng);
  dnnl::reorder(t_bias_mem, bias_mem).execute(stream, t_bias_mem, bias_mem);
}

void FCLayer::Eval(int N, dnnl::memory& output, dnnl::memory& input,
                   dnnl::engine& eng, dnnl::stream& stream) {
  if (last_batch_ != N) {
    const int num_outputs = C * H * W;

    auto t_in_md =
        dnnl::memory::desc({N, input_->GetC(), input_->GetH(), input_->GetW()},
                           data_type_, dnnl::memory::format_tag::any);

    auto t_filter_md = dnnl::memory::desc(
        {num_outputs, input_->GetC(), input_->GetH(), input_->GetW()},
        data_type_, dnnl::memory::format_tag::any);

    auto t_out_md = dnnl::memory::desc({N, C, H, W}, data_type_,
                                       dnnl::memory::format_tag::any);

    auto fc_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, t_in_md, t_filter_md,
        bias_mem.get_desc(), t_out_md.reshape({N, num_outputs}));
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
    scratchpad_mem = dnnl::memory(fc_pd.scratchpad_desc(), eng);
    fc_ = dnnl::inner_product_forward(fc_pd);

    in_md = fc_pd.src_desc();
    out_md = fc_pd.dst_desc().reshape({N, C, H, W});
    if (fc_pd.weights_desc() != filter_mem.get_desc()) {
      auto tmp = dnnl::memory(fc_pd.weights_desc(), eng);
      dnnl::reorder(filter_mem, tmp).execute(stream, filter_mem, tmp);
      filter_mem = tmp;
    }

    auto in_reorder_pd =
        dnnl::reorder::primitive_desc(eng, input.get_desc(), eng, in_md);
    in_reorder_ = dnnl::reorder(in_reorder_pd);

    last_batch_ = N;
  }

  if (in_md != input.get_desc()) {
    auto tmp = dnnl::memory(in_md, eng);
    in_reorder_.execute(stream, input, tmp);
    input = tmp;
  }

  if (!output || out_md != output.get_desc()) {
    output = dnnl::memory(out_md, eng);
  }

  fc_.execute(stream, {{DNNL_ARG_SRC, input},
                       {DNNL_ARG_WEIGHTS, filter_mem},
                       {DNNL_ARG_BIAS, bias_mem},
                       {DNNL_ARG_DST, output},
                       {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});
}

}  // namespace dnnl_backend
}  // namespace lczero
