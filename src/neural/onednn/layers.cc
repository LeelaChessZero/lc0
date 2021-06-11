/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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

namespace onednn_backend {

BaseLayer::BaseLayer(int c, int h, int w, BaseLayer* ip)
    : input_(ip), C(c), H(h), W(w) {
  if (ip) {
    data_type_ = ip->data_type_;
    convolution_type_ = ip->convolution_type_;
  } else {
    data_type_ = dnnl::memory::data_type::undef;
    convolution_type_ = dnnl::algorithm::convolution_auto;
  }
}

ConvLayer::ConvLayer(BaseLayer* ip, int C, int H, int W, int filter, int Cin,
                     bool relu, bool skip)
    : BaseLayer(C, H, W, ip),
      c_input_(Cin),
      filter_size_(filter),
      use_relu_(relu),
      use_skip_(skip) {}

void ConvLayer::LoadWeights(dnnl::memory& w1, dnnl::memory& b1,
                            dnnl::engine& eng, dnnl::stream& stream) {
  auto filter_md =
      dnnl::memory::desc({C, c_input_, filter_size_, filter_size_}, data_type_,
                         dnnl::memory::format_tag::oihw);
  filter_mem = dnnl::memory(filter_md, eng);
  dnnl::reorder(w1, filter_mem).execute(stream, w1, filter_mem);

  auto bias_md =
      dnnl::memory::desc({C}, data_type_, dnnl::memory::format_tag::a);
  bias_mem = dnnl::memory(bias_md, eng);
  dnnl::reorder(b1, bias_mem).execute(stream, b1, bias_mem);
}

void ConvLayer::Eval(int N, dnnl::memory& output, dnnl::memory& input,
                     dnnl::engine& eng, dnnl::stream& stream) {
  std::lock_guard<std::mutex> lock(lock_);
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
        dnnl::prop_kind::forward_inference,
        filter_size_ == 3 ? convolution_type_
                          : dnnl::algorithm::convolution_auto,
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

void SELayer::LoadWeights(dnnl::memory& w1, dnnl::memory& b1, dnnl::memory& w2,
                          dnnl::memory& b2, dnnl::engine& eng,
                          dnnl::stream& stream) {
  auto filter_md = dnnl::memory::desc({numFc1Out_, C}, data_type_,
                                      dnnl::memory::format_tag::ab);
  filter_mem = dnnl::memory(filter_md, eng);
  dnnl::reorder(w1, filter_mem).execute(stream, w1, filter_mem);

  auto bias_md =
      dnnl::memory::desc({numFc1Out_}, data_type_, dnnl::memory::format_tag::a);
  bias_mem = dnnl::memory(bias_md, eng);
  dnnl::reorder(b1, bias_mem).execute(stream, b1, bias_mem);

  auto filter2_md = dnnl::memory::desc({2 * C, numFc1Out_}, data_type_,
                                       dnnl::memory::format_tag::ab);

  filter2_mem = dnnl::memory(filter2_md, eng);
  dnnl::reorder(w2, filter2_mem).execute(stream, w2, filter2_mem);

  auto bias2_md =
      dnnl::memory::desc({2 * C}, data_type_, dnnl::memory::format_tag::a);

  bias2_mem = dnnl::memory(bias2_md, eng);
  dnnl::reorder(b2, bias2_mem).execute(stream, b2, bias2_mem);
}

void SELayer::Eval(int N, dnnl::memory& output, dnnl::memory& input,
                   dnnl::engine& eng, dnnl::stream& stream) {
  std::lock_guard<std::mutex> lock(lock_);
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

    auto t_filter2_md = dnnl::memory::desc({2 * C, numFc1Out_}, data_type_,
                                           dnnl::memory::format_tag::any);

    auto t_fc2_out_md = dnnl::memory::desc({N, 2 * C}, data_type_,
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

    auto fc2_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, fc1_out_md, t_filter2_md,
        bias2_mem.get_desc(), t_fc2_out_md);
    dnnl::primitive_attr fc2_attr;
    fc2_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto fc2_pd =
        dnnl::inner_product_forward::primitive_desc(fc2_d, fc2_attr, eng);
    fc2_scratchpad_mem = dnnl::memory(fc2_pd.scratchpad_desc(), eng);
    fc2_ = dnnl::inner_product_forward(fc2_pd);

    if (fc2_pd.weights_desc() != filter2_mem.get_desc()) {
      auto tmp = dnnl::memory(fc2_pd.weights_desc(), eng);
      dnnl::reorder(filter2_mem, tmp).execute(stream, filter2_mem, tmp);
      filter2_mem = tmp;
    }

    fc2_out_md = fc2_pd.dst_desc();

    auto sigmoid_d = dnnl::eltwise_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_logistic,
        pool_out_md, 0.f, 0.f);
    dnnl::primitive_attr sigmoid_attr;
    sigmoid_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto sigmoid_pd =
        dnnl::eltwise_forward::primitive_desc(sigmoid_d, sigmoid_attr, eng);
    sigmoid_scratchpad_mem = dnnl::memory(sigmoid_pd.scratchpad_desc(), eng);
    sigmoid_ = dnnl::eltwise_forward(sigmoid_pd);

    auto mul_d =
        dnnl::binary::desc(dnnl::algorithm::binary_mul, input.get_desc(),
                           pool_out_md, output.get_desc());
    dnnl::post_ops mul_ops;
    mul_ops.append_sum();
    if (eng.get_kind() == dnnl::engine::kind::gpu) {
      // Using binary post-ops is a gain on gpu but a huge loss on cpu.
      mul_ops.append_binary(dnnl::algorithm::binary_add, pool_out_md);
      mul_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    }
    dnnl::primitive_attr mul_attr;
    mul_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    mul_attr.set_post_ops(mul_ops);
    auto mul_pd = dnnl::binary::primitive_desc(mul_d, mul_attr, eng);
    mul_scratchpad_mem = dnnl::memory(mul_pd.scratchpad_desc(), eng);
    mul_ = dnnl::binary(mul_pd);

    if (eng.get_kind() != dnnl::engine::kind::gpu) {
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
    }
    auto fc1_reorder_pd =
        dnnl::reorder::primitive_desc(eng, pool_out_md, eng, fc1_in_md);
    fc1_reorder_ = dnnl::reorder(fc1_reorder_pd);

    auto mul_reorder_pd = dnnl::reorder::primitive_desc(
        eng, fc2_out_md.submemory_desc({N, C}, {0, 0}).reshape({N, C, 1, 1}),
        eng, pool_out_md);
    mul_reorder_ = dnnl::reorder(mul_reorder_pd);

    auto add_reorder_pd = dnnl::reorder::primitive_desc(
        eng, fc2_out_md.submemory_desc({N, C}, {0, C}).reshape({N, C, 1, 1}),
        eng, pool_out_md);
    add_reorder_ = dnnl::reorder(add_reorder_pd);

    last_batch_ = N;
  }

  auto pool_out_mem = dnnl::memory(pool_out_md, eng);
  auto fc1_out_mem = dnnl::memory(fc1_out_md, eng);
  auto fc2_out_mem = dnnl::memory(fc2_out_md, eng);

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

  fc2_.execute(stream, {{DNNL_ARG_SRC, fc1_out_mem},
                        {DNNL_ARG_WEIGHTS, filter2_mem},
                        {DNNL_ARG_BIAS, bias2_mem},
                        {DNNL_ARG_DST, fc2_out_mem},
                        {DNNL_ARG_SCRATCHPAD, fc2_scratchpad_mem}});

  dnnl::memory mul_in_mem;
  mul_in_mem = dnnl::memory(pool_out_md, eng);
  mul_reorder_.execute(stream, fc2_out_mem, mul_in_mem);

  sigmoid_.execute(stream, {{DNNL_ARG_SRC, mul_in_mem},
                            {DNNL_ARG_DST, mul_in_mem},
                            {DNNL_ARG_SCRATCHPAD, sigmoid_scratchpad_mem}});

  dnnl::memory add_in_mem;
  add_in_mem = dnnl::memory(pool_out_md, eng);
  add_reorder_.execute(stream, fc2_out_mem, add_in_mem);

  if (eng.get_kind() == dnnl::engine::kind::gpu) {
    mul_.execute(stream, {{DNNL_ARG_SRC_0, input},
                          {DNNL_ARG_SRC_1, mul_in_mem},
                          {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                           add_in_mem},
                          {DNNL_ARG_DST, output},
                          {DNNL_ARG_SCRATCHPAD, mul_scratchpad_mem}});
  } else {
    mul_.execute(stream, {{DNNL_ARG_SRC_0, input},
                          {DNNL_ARG_SRC_1, mul_in_mem},
                          {DNNL_ARG_DST, output},
                          {DNNL_ARG_SCRATCHPAD, mul_scratchpad_mem}});

    add_.execute(stream, {{DNNL_ARG_SRC_0, output},
                          {DNNL_ARG_SRC_1, add_in_mem},
                          {DNNL_ARG_DST, output},
                          {DNNL_ARG_SCRATCHPAD, add_scratchpad_mem}});
  }
}

FCLayer::FCLayer(BaseLayer* ip, int C, int H, int W, bool relu, bool tanh)
    : BaseLayer(C, H, W, ip), use_relu_(relu), use_tanh_(tanh) {}

void FCLayer::LoadWeights(dnnl::memory& w1, dnnl::memory& b1, dnnl::engine& eng,
                          dnnl::stream& stream) {
  const int num_outputs = C * H * W;

  auto filter_md = dnnl::memory::desc(
      {num_outputs, input_->GetC(), input_->GetH(), input_->GetW()}, data_type_,
      dnnl::memory::format_tag::abcd);
  filter_mem = dnnl::memory(filter_md, eng);
  dnnl::reorder(w1, filter_mem).execute(stream, w1, filter_mem);

  auto bias_md = dnnl::memory::desc({num_outputs}, data_type_,
                                    dnnl::memory::format_tag::a);
  bias_mem = dnnl::memory(bias_md, eng);
  dnnl::reorder(b1, bias_mem).execute(stream, b1, bias_mem);
}

void FCLayer::Eval(int N, dnnl::memory& output, dnnl::memory& input,
                   dnnl::engine& eng, dnnl::stream& stream) {
  std::lock_guard<std::mutex> lock(lock_);
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

}  // namespace onednn_backend
}  // namespace lczero
