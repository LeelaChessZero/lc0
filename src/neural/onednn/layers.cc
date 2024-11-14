/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021-2022 The LCZero Authors

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
#include <cmath>
#include <cstring>
#include <vector>

namespace lczero {

namespace onednn_backend {

#if DNNL_VERSION_MAJOR < 3
#define APPEND_ELTWISE(A, B, C) append_eltwise(1.0f, A, B, C)
#else
#define APPEND_ELTWISE(A, B, C) append_eltwise(A, B, C)
#endif

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
                     ActivationFunction activation, bool skip)
    : BaseLayer(C, H, W, ip),
      c_input_(Cin),
      filter_size_(filter),
      activation_(activation),
      use_skip_(skip) {}

void ConvLayer::LoadWeights(dnnl::memory& w1, dnnl::memory& b1,
                            const dnnl::engine& eng,
                            const dnnl::stream& stream) {
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

void ConvLayer::Eval(int N, dnnl::memory& output, const dnnl::memory& input,
                     const dnnl::memory& scratch, const dnnl::engine& eng,
                     const dnnl::stream& stream, dnnl::memory& scratchpad_mem) {
  if (last_batch_ != N) {
    auto t_in_md = dnnl::memory::desc({N, c_input_, H, W}, data_type_,
                                      dnnl::memory::format_tag::any);

    auto t_filter_md =
        dnnl::memory::desc({C, c_input_, filter_size_, filter_size_},
                           data_type_, dnnl::memory::format_tag::any);

    auto t_out_md = use_skip_
                        ? output.get_desc()
                        : dnnl::memory::desc({N, C, H, W}, data_type_,
                                             dnnl::memory::format_tag::any);

    const int padding = filter_size_ / 2;
    dnnl::post_ops conv_ops;
    if (use_skip_) {
      conv_ops.append_sum();
    }
    if (activation_ == ACTIVATION_RELU) {
      conv_ops.APPEND_ELTWISE(dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    } else if (activation_ == ACTIVATION_TANH) {
      conv_ops.APPEND_ELTWISE(dnnl::algorithm::eltwise_tanh, 0.0f, 0.0f);
    }
    dnnl::primitive_attr conv_attr;
    conv_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    conv_attr.set_post_ops(conv_ops);
#if DNNL_VERSION_MAJOR < 3
    auto conv_d = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference,
        filter_size_ == 3 ? convolution_type_
                          : dnnl::algorithm::convolution_auto,
        t_in_md, t_filter_md, bias_mem.get_desc(), t_out_md, {1, 1},
        {padding, padding}, {padding, padding});
    auto conv_pd =
        dnnl::convolution_forward::primitive_desc(conv_d, conv_attr, eng);
#else
    auto conv_pd = dnnl::convolution_forward::primitive_desc(
        eng, dnnl::prop_kind::forward_inference,
        filter_size_ == 3 ? convolution_type_
                          : dnnl::algorithm::convolution_auto,
        t_in_md, t_filter_md, bias_mem.get_desc(), t_out_md, {1, 1},
        {padding, padding}, {padding, padding}, conv_attr);
#endif
    scratchpad_md = conv_pd.scratchpad_desc();
    conv_ = dnnl::convolution_forward(conv_pd);

    in_md = conv_pd.src_desc();
    out_md = use_skip_ ? output.get_desc() : conv_pd.dst_desc();

    // Apparently convolution doesn't go well with mish post op.
    if (activation_ == ACTIVATION_MISH) {
      dnnl::primitive_attr mish_attr;
      mish_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#if DNNL_VERSION_MAJOR < 3
      auto mish_d = dnnl::eltwise_forward::desc(
          dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_mish,
          out_md, 0.f, 0.f);
      auto mish_pd =
          dnnl::eltwise_forward::primitive_desc(mish_d, mish_attr, eng);
#else
      auto mish_pd = dnnl::eltwise_forward::primitive_desc(
          eng, dnnl::prop_kind::forward_inference,
          dnnl::algorithm::eltwise_mish, out_md, out_md, 0.f, 0.f, mish_attr);
#endif
      mish_ = dnnl::eltwise_forward(mish_pd);
      if (scratchpad_md.get_size() < mish_pd.scratchpad_desc().get_size()) {
        scratchpad_md = mish_pd.scratchpad_desc();
      }
    }

    if (!conv_filter_mem ||
        conv_pd.weights_desc() != conv_filter_mem.get_desc()) {
      // This may be a transformation for Winograd convolution, so keep the
      // original weights.
      conv_filter_mem = dnnl::memory(conv_pd.weights_desc(), eng);
      dnnl::reorder(filter_mem, conv_filter_mem)
          .execute(stream, filter_mem, conv_filter_mem);
    }

    dnnl::primitive_attr reorder_attr;
    reorder_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto in_reorder_pd = dnnl::reorder::primitive_desc(
        eng, input.get_desc(), eng, in_md, reorder_attr);
    in_reorder_ = dnnl::reorder(in_reorder_pd);
    if (scratchpad_md.get_size() < in_reorder_pd.scratchpad_desc().get_size()) {
      scratchpad_md = in_reorder_pd.scratchpad_desc();
    }

    last_batch_ = N;
  }

  if (!scratchpad_mem ||
      scratchpad_mem.get_desc().get_size() < scratchpad_md.get_size()) {
    scratchpad_mem = dnnl::memory(scratchpad_md, eng);
  }

  dnnl::memory in;
  if (in_md != input.get_desc()) {
    in_reorder_.execute(stream, {{DNNL_ARG_SRC, input},
                                 {DNNL_ARG_DST, scratch},
                                 {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});
    in = scratch;
  } else {
    in = input;
  }

  if (out_md != output.get_desc()) {
    output = dnnl::memory(out_md, eng, output.get_data_handle());
  }

  conv_.execute(stream, {{DNNL_ARG_SRC, in},
                         {DNNL_ARG_WEIGHTS, conv_filter_mem},
                         {DNNL_ARG_BIAS, bias_mem},
                         {DNNL_ARG_DST, output},
                         {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});

  if (activation_ == ACTIVATION_MISH) {
    mish_.execute(stream, {{DNNL_ARG_SRC, output},
                           {DNNL_ARG_DST, output},
                           {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});
  }
}

SELayer::SELayer(BaseLayer* ip, int fc1Outputs, ActivationFunction activation)
    : BaseLayer(ip->GetC(), ip->GetH(), ip->GetW(), ip),
      numFc1Out_(fc1Outputs),
      activation_(activation) {}

void SELayer::LoadWeights(dnnl::memory& w1, dnnl::memory& b1, dnnl::memory& w2,
                          dnnl::memory& b2, const dnnl::engine& eng,
                          const dnnl::stream& stream) {
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

void SELayer::Eval(int N, dnnl::memory& output, const dnnl::memory& input,
                   const dnnl::memory& scratch, const dnnl::engine& eng,
                   const dnnl::stream& stream, dnnl::memory& scratchpad_mem) {
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

    dnnl::primitive_attr pooling_attr;
    pooling_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#if DNNL_VERSION_MAJOR < 3
    auto pooling_d = dnnl::pooling_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::pooling_avg,
        input.get_desc(), t_pool_out_md, {1, 1}, {H, W}, {0, 0}, {0, 0});
    auto pooling_pd =
        dnnl::pooling_forward::primitive_desc(pooling_d, pooling_attr, eng);
#else
    auto pooling_pd = dnnl::pooling_forward::primitive_desc(
        eng, dnnl::prop_kind::forward_inference,
        dnnl::algorithm::pooling_avg_exclude_padding, input.get_desc(),
        t_pool_out_md, {1, 1}, {H, W}, {0, 0}, {0, 0}, {0, 0}, pooling_attr);
#endif
    pooling_ = dnnl::pooling_forward(pooling_pd);
    scratchpad_md = pooling_pd.scratchpad_desc();

    // This is also the optimized memory format descriptor for the binary
    // primitives.
    pool_out_md = pooling_pd.dst_desc();

    dnnl::post_ops fc_ops;
    if (activation_ == ACTIVATION_RELU) {
      fc_ops.APPEND_ELTWISE(dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    } else if (activation_ == ACTIVATION_MISH) {
      fc_ops.APPEND_ELTWISE(dnnl::algorithm::eltwise_mish, 0.0f, 0.0f);
    } else if (activation_ == ACTIVATION_TANH) {
      fc_ops.APPEND_ELTWISE(dnnl::algorithm::eltwise_tanh, 0.0f, 0.0f);
    }
    dnnl::primitive_attr fc_attr;
    fc_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    fc_attr.set_post_ops(fc_ops);
#if DNNL_VERSION_MAJOR < 3
    auto fc_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, t_fc1_in_md, t_filter_md,
        bias_mem.get_desc(), t_fc1_out_md);
    auto fc_pd =
        dnnl::inner_product_forward::primitive_desc(fc_d, fc_attr, eng);
#else
    auto fc_pd = dnnl::inner_product_forward::primitive_desc(
        eng, dnnl::prop_kind::forward_inference, t_fc1_in_md, t_filter_md,
        bias_mem.get_desc(), t_fc1_out_md, fc_attr);
#endif
    fc_ = dnnl::inner_product_forward(fc_pd);
    if (scratchpad_md.get_size() < fc_pd.scratchpad_desc().get_size()) {
      scratchpad_md = fc_pd.scratchpad_desc();
    }

    fc1_in_md = fc_pd.src_desc().reshape({N, C, 1, 1});
    fc1_out_md = fc_pd.dst_desc();
    if (fc_pd.weights_desc() != filter_mem.get_desc()) {
      auto tmp = dnnl::memory(fc_pd.weights_desc(), eng);
      dnnl::reorder(filter_mem, tmp).execute(stream, filter_mem, tmp);
      filter_mem = tmp;
    }

    dnnl::primitive_attr fc2_attr;
    fc2_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#if DNNL_VERSION_MAJOR < 3
    auto fc2_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, fc1_out_md, t_filter2_md,
        bias2_mem.get_desc(), t_fc2_out_md);
    auto fc2_pd =
        dnnl::inner_product_forward::primitive_desc(fc2_d, fc2_attr, eng);
#else
    auto fc2_pd = dnnl::inner_product_forward::primitive_desc(
        eng, dnnl::prop_kind::forward_inference, fc1_out_md, t_filter2_md,
        bias2_mem.get_desc(), t_fc2_out_md, fc2_attr);
#endif
    fc2_ = dnnl::inner_product_forward(fc2_pd);
    if (scratchpad_md.get_size() < fc2_pd.scratchpad_desc().get_size()) {
      scratchpad_md = fc2_pd.scratchpad_desc();
    }

    if (fc2_pd.weights_desc() != filter2_mem.get_desc()) {
      auto tmp = dnnl::memory(fc2_pd.weights_desc(), eng);
      dnnl::reorder(filter2_mem, tmp).execute(stream, filter2_mem, tmp);
      filter2_mem = tmp;
    }

    fc2_out_md = fc2_pd.dst_desc();

    dnnl::primitive_attr sigmoid_attr;
    sigmoid_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#if DNNL_VERSION_MAJOR < 3
    auto sigmoid_d = dnnl::eltwise_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_logistic,
        pool_out_md, 0.f, 0.f);
    auto sigmoid_pd =
        dnnl::eltwise_forward::primitive_desc(sigmoid_d, sigmoid_attr, eng);
#else
    auto sigmoid_pd = dnnl::eltwise_forward::primitive_desc(
        eng, dnnl::prop_kind::forward_inference,
        dnnl::algorithm::eltwise_logistic, pool_out_md, pool_out_md, 0.f, 0.f,
        sigmoid_attr);
#endif
    sigmoid_ = dnnl::eltwise_forward(sigmoid_pd);
    if (scratchpad_md.get_size() < sigmoid_pd.scratchpad_desc().get_size()) {
      scratchpad_md = sigmoid_pd.scratchpad_desc();
    }

    dnnl::post_ops mul_ops;
    mul_ops.append_sum();
    if (eng.get_kind() == dnnl::engine::kind::gpu) {
      // Using binary post-ops is a gain on gpu but a huge loss on cpu.
      mul_ops.append_binary(dnnl::algorithm::binary_add, pool_out_md);
      if (activation_ == ACTIVATION_RELU) {
        mul_ops.APPEND_ELTWISE(dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
      } else if (activation_ == ACTIVATION_MISH) {
        mul_ops.APPEND_ELTWISE(dnnl::algorithm::eltwise_mish, 0.0f, 0.0f);
      } else if (activation_ == ACTIVATION_TANH) {
        mul_ops.APPEND_ELTWISE(dnnl::algorithm::eltwise_tanh, 0.0f, 0.0f);
      }
    }
    dnnl::primitive_attr mul_attr;
    mul_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    mul_attr.set_post_ops(mul_ops);
#if DNNL_VERSION_MAJOR < 3
    auto mul_d =
        dnnl::binary::desc(dnnl::algorithm::binary_mul, input.get_desc(),
                           pool_out_md, output.get_desc());
    auto mul_pd = dnnl::binary::primitive_desc(mul_d, mul_attr, eng);
#else
    auto mul_pd = dnnl::binary::primitive_desc(eng, dnnl::algorithm::binary_mul,
                                               input.get_desc(), pool_out_md,
                                               output.get_desc(), mul_attr);
#endif
    mul_ = dnnl::binary(mul_pd);
    if (scratchpad_md.get_size() < mul_pd.scratchpad_desc().get_size()) {
      scratchpad_md = mul_pd.scratchpad_desc();
    }

    if (eng.get_kind() != dnnl::engine::kind::gpu) {
      dnnl::post_ops add_ops;
      if (activation_ == ACTIVATION_RELU) {
        add_ops.APPEND_ELTWISE(dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
      } else if (activation_ == ACTIVATION_MISH) {
        add_ops.APPEND_ELTWISE(dnnl::algorithm::eltwise_mish, 0.0f, 0.0f);
      } else if (activation_ == ACTIVATION_TANH) {
        add_ops.APPEND_ELTWISE(dnnl::algorithm::eltwise_tanh, 0.0f, 0.0f);
      }
      dnnl::primitive_attr add_attr;
      add_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      add_attr.set_post_ops(add_ops);
#if DNNL_VERSION_MAJOR < 3
      auto add_d =
          dnnl::binary::desc(dnnl::algorithm::binary_add, output.get_desc(),
                             pool_out_md, output.get_desc());
      auto add_pd = dnnl::binary::primitive_desc(add_d, add_attr, eng);
#else
      auto add_pd = dnnl::binary::primitive_desc(
          eng, dnnl::algorithm::binary_add, output.get_desc(), pool_out_md,
          output.get_desc(), add_attr);
#endif
      add_ = dnnl::binary(add_pd);
      if (scratchpad_md.get_size() < add_pd.scratchpad_desc().get_size()) {
        scratchpad_md = add_pd.scratchpad_desc();
      }
    }

    dnnl::primitive_attr reorder_attr;
    reorder_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto fc1_reorder_pd = dnnl::reorder::primitive_desc(
        eng, pool_out_md, eng, fc1_in_md, reorder_attr);
    fc1_reorder_ = dnnl::reorder(fc1_reorder_pd);
    if (scratchpad_md.get_size() <
        fc1_reorder_pd.scratchpad_desc().get_size()) {
      scratchpad_md = fc1_reorder_pd.scratchpad_desc();
    }

    auto mul_reorder_pd = dnnl::reorder::primitive_desc(
        eng, fc2_out_md.submemory_desc({N, C}, {0, 0}).reshape({N, C, 1, 1}),
        eng, pool_out_md, reorder_attr);
    mul_reorder_ = dnnl::reorder(mul_reorder_pd);
    if (scratchpad_md.get_size() <
        mul_reorder_pd.scratchpad_desc().get_size()) {
      scratchpad_md = mul_reorder_pd.scratchpad_desc();
    }

    auto add_reorder_pd = dnnl::reorder::primitive_desc(
        eng, fc2_out_md.submemory_desc({N, C}, {0, C}).reshape({N, C, 1, 1}),
        eng, pool_out_md, reorder_attr);
    add_reorder_ = dnnl::reorder(add_reorder_pd);
    if (scratchpad_md.get_size() <
        add_reorder_pd.scratchpad_desc().get_size()) {
      scratchpad_md = add_reorder_pd.scratchpad_desc();
    }

    buf1 = dnnl::memory(pool_out_md, eng);
    if (fc1_out_md.get_size() > pool_out_md.get_size()) {
      buf2 = dnnl::memory(fc1_out_md, eng);
    } else {
      buf2 = dnnl::memory(pool_out_md, eng);
    }

    last_batch_ = N;
  }

  if (!scratchpad_mem ||
      scratchpad_mem.get_desc().get_size() < scratchpad_md.get_size()) {
    scratchpad_mem = dnnl::memory(scratchpad_md, eng);
  }

  auto& pool_out_mem = buf1;
  auto& fc1_out_mem = buf2;
  auto& fc2_out_mem = scratch;

  pooling_.execute(stream, {{DNNL_ARG_SRC, input},
                            {DNNL_ARG_DST, pool_out_mem},
                            {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});

  dnnl::memory fc1_in_mem;
  if (fc1_in_md != pool_out_md) {
    fc1_reorder_.execute(stream, {{DNNL_ARG_SRC, pool_out_mem},
                                  {DNNL_ARG_DST, scratch},
                                  {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});
    fc1_in_mem = scratch;
  } else {
    fc1_in_mem = pool_out_mem;
  }

  fc_.execute(stream, {{DNNL_ARG_SRC, fc1_in_mem},
                       {DNNL_ARG_WEIGHTS, filter_mem},
                       {DNNL_ARG_BIAS, bias_mem},
                       {DNNL_ARG_DST, fc1_out_mem},
                       {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});

  fc2_.execute(stream, {{DNNL_ARG_SRC, fc1_out_mem},
                        {DNNL_ARG_WEIGHTS, filter2_mem},
                        {DNNL_ARG_BIAS, bias2_mem},
                        {DNNL_ARG_DST, fc2_out_mem},
                        {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});

  auto& mul_in_mem = buf1;
  mul_reorder_.execute(stream, {{DNNL_ARG_SRC, fc2_out_mem},
                                {DNNL_ARG_DST, mul_in_mem},
                                {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});

  sigmoid_.execute(stream, {{DNNL_ARG_SRC, mul_in_mem},
                            {DNNL_ARG_DST, mul_in_mem},
                            {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});

  auto& add_in_mem = buf2;
  add_reorder_.execute(stream, {{DNNL_ARG_SRC, fc2_out_mem},
                                {DNNL_ARG_DST, add_in_mem},
                                {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});

  if (eng.get_kind() == dnnl::engine::kind::gpu) {
    mul_.execute(stream, {{DNNL_ARG_SRC_0, input},
                          {DNNL_ARG_SRC_1, mul_in_mem},
                          {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                           add_in_mem},
                          {DNNL_ARG_DST, output},
                          {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});
  } else {
    mul_.execute(stream, {{DNNL_ARG_SRC_0, input},
                          {DNNL_ARG_SRC_1, mul_in_mem},
                          {DNNL_ARG_DST, output},
                          {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});

    add_.execute(stream, {{DNNL_ARG_SRC_0, output},
                          {DNNL_ARG_SRC_1, add_in_mem},
                          {DNNL_ARG_DST, output},
                          {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});
  }
}

FCLayer::FCLayer(BaseLayer* ip, int C, int H, int W,
                 ActivationFunction activation)
    : BaseLayer(C, H, W, ip), activation_(activation) {}

void FCLayer::LoadWeights(dnnl::memory& w1, dnnl::memory& b1,
                          const dnnl::engine& eng, const dnnl::stream& stream) {
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

void FCLayer::Eval(int N, dnnl::memory& output, const dnnl::memory& input,
                   const dnnl::memory& scratch, const dnnl::engine& eng,
                   const dnnl::stream& stream, dnnl::memory& scratchpad_mem) {
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

    dnnl::post_ops fc_ops;
    if (activation_ == ACTIVATION_RELU) {
      fc_ops.APPEND_ELTWISE(dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    } else if (activation_ == ACTIVATION_MISH) {
      fc_ops.APPEND_ELTWISE(dnnl::algorithm::eltwise_mish, 0.0f, 0.0f);
    } else if (activation_ == ACTIVATION_TANH) {
      fc_ops.APPEND_ELTWISE(dnnl::algorithm::eltwise_tanh, 0.0f, 0.0f);
    }
    dnnl::primitive_attr fc_attr;
    fc_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    fc_attr.set_post_ops(fc_ops);
#if DNNL_VERSION_MAJOR < 3
    auto fc_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, t_in_md, t_filter_md,
        bias_mem.get_desc(), t_out_md.reshape({N, num_outputs}));
    auto fc_pd =
        dnnl::inner_product_forward::primitive_desc(fc_d, fc_attr, eng);
#else
    auto fc_pd = dnnl::inner_product_forward::primitive_desc(
        eng, dnnl::prop_kind::forward_inference, t_in_md, t_filter_md,
        bias_mem.get_desc(), t_out_md.reshape({N, num_outputs}), fc_attr);
#endif
    fc_ = dnnl::inner_product_forward(fc_pd);
    scratchpad_md = fc_pd.scratchpad_desc();

    in_md = fc_pd.src_desc();
    out_md = fc_pd.dst_desc().reshape({N, C, H, W});
    if (fc_pd.weights_desc() != filter_mem.get_desc()) {
      auto tmp = dnnl::memory(fc_pd.weights_desc(), eng);
      dnnl::reorder(filter_mem, tmp).execute(stream, filter_mem, tmp);
      filter_mem = tmp;
    }

    dnnl::primitive_attr reorder_attr;
    reorder_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto in_reorder_pd = dnnl::reorder::primitive_desc(
        eng, input.get_desc(), eng, in_md, reorder_attr);
    in_reorder_ = dnnl::reorder(in_reorder_pd);
    if (scratchpad_md.get_size() < in_reorder_pd.scratchpad_desc().get_size()) {
      scratchpad_md = in_reorder_pd.scratchpad_desc();
    }

    last_batch_ = N;
  }

  if (!scratchpad_mem ||
      scratchpad_mem.get_desc().get_size() < scratchpad_md.get_size()) {
    scratchpad_mem = dnnl::memory(scratchpad_md, eng);
  }

  dnnl::memory in;
  if (in_md != input.get_desc()) {
    in_reorder_.execute(stream, {{DNNL_ARG_SRC, input},
                                 {DNNL_ARG_DST, scratch},
                                 {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});
    in = scratch;
  } else {
    in = input;
  }

  if (out_md != output.get_desc()) {
    output = dnnl::memory(out_md, eng, output.get_data_handle());
  }

  fc_.execute(stream, {{DNNL_ARG_SRC, in},
                       {DNNL_ARG_WEIGHTS, filter_mem},
                       {DNNL_ARG_BIAS, bias_mem},
                       {DNNL_ARG_DST, output},
                       {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});
}

void AttentionPolicyHead::LoadWeights(dnnl::memory& w1, dnnl::memory& b1,
                                      dnnl::memory& w2, dnnl::memory& b2,
                                      dnnl::memory& w3, dnnl::memory& b3,
                                      dnnl::memory& w4, const dnnl::engine& eng,
                                      const dnnl::stream& stream) {
  auto fc_filter_md = dnnl::memory::desc({C, embedding_size_}, data_type_,
                                         dnnl::memory::format_tag::ab);
  fc_filter_mem = dnnl::memory(fc_filter_md, eng);
  dnnl::reorder(w1, fc_filter_mem).execute(stream, w1, fc_filter_mem);
  auto fc_bias_md = dnnl::memory::desc({embedding_size_}, data_type_,
                                       dnnl::memory::format_tag::a);
  fc_bias_mem = dnnl::memory(fc_bias_md, eng);
  dnnl::reorder(b1, fc_bias_mem).execute(stream, b1, fc_bias_mem);

  auto fcQK_filter_md =
      dnnl::memory::desc({embedding_size_, policy_d_model_}, data_type_,
                         dnnl::memory::format_tag::ab);

  fcQ_filter_mem = dnnl::memory(fcQK_filter_md, eng);
  dnnl::reorder(w2, fcQ_filter_mem).execute(stream, w2, fcQ_filter_mem);
  auto fcQK_bias_md = dnnl::memory::desc({policy_d_model_}, data_type_,
                                         dnnl::memory::format_tag::a);
  fcQ_bias_mem = dnnl::memory(fcQK_bias_md, eng);
  dnnl::reorder(b2, fcQ_bias_mem).execute(stream, b2, fcQ_bias_mem);

  fcK_filter_mem = dnnl::memory(fcQK_filter_md, eng);
  dnnl::reorder(w3, fcK_filter_mem).execute(stream, w3, fcK_filter_mem);
  fcK_bias_mem = dnnl::memory(fcQK_bias_md, eng);
  dnnl::reorder(b3, fcK_bias_mem).execute(stream, b3, fcK_bias_mem);

  auto pmul_md = dnnl::memory::desc({1, 4, policy_d_model_}, data_type_,
                                    dnnl::memory::format_tag::abc);
  pmul_mem = dnnl::memory(pmul_md, eng);
  dnnl::reorder(w4, pmul_mem).execute(stream, w4, pmul_mem);
}

void AttentionPolicyHead::Eval(int N, dnnl::memory& output,
                               const dnnl::memory& input,
                               const dnnl::memory& scratch,
                               const dnnl::engine& eng,
                               const dnnl::stream& stream,
                               dnnl::memory& scratchpad_mem) {
  if (last_batch_ != N) {
    in_md = dnnl::memory::desc({N, C, H, W}, data_type_,
                               dnnl::memory::format_tag::nhwc);
    out_md = dnnl::memory::desc({N, 67, 8, 8}, data_type_,
                                dnnl::memory::format_tag::nchw);

    auto fc_out_md = dnnl::memory::desc({N * 64, embedding_size_}, data_type_,
                                        dnnl::memory::format_tag::ab);
    fc_out_mem = dnnl::memory(fc_out_md, eng);
    auto foo_md = dnnl::memory::desc({N, H, W, C}, data_type_,
                                     dnnl::memory::format_tag::nchw);
    dnnl::post_ops fc_ops;
    // SELU activation.
    fc_ops.APPEND_ELTWISE(dnnl::algorithm::eltwise_elu, 1.67326324f, 0.0f);
    fc_ops.APPEND_ELTWISE(dnnl::algorithm::eltwise_linear, 1.05070098f, 0.0f);
    dnnl::primitive_attr fc_attr;
    fc_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    fc_attr.set_post_ops(fc_ops);
#if DNNL_VERSION_MAJOR < 3
    auto fc_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, foo_md.reshape({N * H * W, C}),
        fc_filter_mem.get_desc(), fc_bias_mem.get_desc(), fc_out_md);
    auto fc_pd =
        dnnl::inner_product_forward::primitive_desc(fc_d, fc_attr, eng);
#else
    auto fc_pd = dnnl::inner_product_forward::primitive_desc(
        eng, dnnl::prop_kind::forward_inference, foo_md.reshape({N * H * W, C}),
        fc_filter_mem.get_desc(), fc_bias_mem.get_desc(), fc_out_md, fc_attr);
#endif
    fc_ = dnnl::inner_product_forward(fc_pd);
    scratchpad_md = fc_pd.scratchpad_desc();

    // Q
    auto fcQK_out_md = dnnl::memory::desc({N * 64, policy_d_model_}, data_type_,
                                          dnnl::memory::format_tag::ab);
    fcQ_out_mem = dnnl::memory(fcQK_out_md, eng);

    dnnl::primitive_attr common_attr;
    common_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#if DNNL_VERSION_MAJOR < 3
    auto fcQK_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, fc_out_md,
        fcQ_filter_mem.get_desc(), fcQ_bias_mem.get_desc(), fcQK_out_md);
    auto fcQK_pd =
        dnnl::inner_product_forward::primitive_desc(fcQK_d, common_attr, eng);
#else
    auto fcQK_pd = dnnl::inner_product_forward::primitive_desc(
        eng, dnnl::prop_kind::forward_inference, fc_out_md,
        fcQ_filter_mem.get_desc(), fcQ_bias_mem.get_desc(), fcQK_out_md,
        common_attr);
#endif
    fcQK_ = dnnl::inner_product_forward(fcQK_pd);
    if (scratchpad_md.get_size() < fcQK_pd.scratchpad_desc().get_size()) {
      scratchpad_md = fcQK_pd.scratchpad_desc();
    }

    // K
    fcK_out_mem = dnnl::memory(fcQK_out_md, eng);
    const float scaling = sqrtf(policy_d_model_);
    auto mul_A_md = dnnl::memory::desc({N, 64, policy_d_model_}, data_type_,
                                       dnnl::memory::format_tag::abc);
    auto mul_B_md = dnnl::memory::desc({N, policy_d_model_, 64}, data_type_,
                                       dnnl::memory::format_tag::acb);
    auto mul_C_md =
        dnnl::memory::desc({N, 64, 64}, data_type_, {64 * 67, 64, 1});
    dnnl::primitive_attr mul_attr;
    mul_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#if DNNL_VERSION_MAJOR < 3
    mul_attr.set_output_scales(0, {1.0f / scaling});
    auto mul_d = dnnl::matmul::desc(mul_A_md, mul_B_md, mul_C_md);
    auto mul_pd = dnnl::matmul::primitive_desc(mul_d, mul_attr, eng);
#else
    dnnl::post_ops mul_ops;
    mul_ops.append_eltwise(dnnl::algorithm::eltwise_linear, 1.0f / scaling,
                           0.0f);
    mul_attr.set_post_ops(mul_ops);
    auto mul_pd = dnnl::matmul::primitive_desc(eng, mul_A_md, mul_B_md,
                                               mul_C_md, mul_attr);
#endif
    mul_ = dnnl::matmul(mul_pd);
    if (scratchpad_md.get_size() < mul_pd.scratchpad_desc().get_size()) {
      scratchpad_md = mul_pd.scratchpad_desc();
    }

    // The promotion offsets are stored in the output array for now.
    auto promo_md =
        out_md.submemory_desc({N, 1, 4, 8}, {0, 64, 0, 0}).reshape({N, 4, 8});

    if (eng.get_kind() == dnnl::engine::kind::gpu) {
      // The gpu matmul primitive ignores memory offsets, so a copy is needed.
      auto reorder_pd = dnnl::reorder::primitive_desc(
          eng, mul_B_md.submemory_desc({N, policy_d_model_, 8}, {0, 0, 56}),
          eng, mul_B_md.submemory_desc({N, policy_d_model_, 8}, {0, 0, 0}),
          common_attr);
      hack_reorder_ = dnnl::reorder(reorder_pd);
      if (scratchpad_md.get_size() < reorder_pd.scratchpad_desc().get_size()) {
        scratchpad_md = reorder_pd.scratchpad_desc();
      }
#if DNNL_VERSION_MAJOR < 3
      auto pmul_d = dnnl::matmul::desc(
          pmul_mem.get_desc(),
          mul_B_md.submemory_desc({N, policy_d_model_, 8}, {0, 0, 0}),
          mul_A_md.submemory_desc({N, 4, 8}, {0, 0, 0}));
      auto pmul_pd = dnnl::matmul::primitive_desc(pmul_d, common_attr, eng);
#else
      auto pmul_pd = dnnl::matmul::primitive_desc(
          eng, pmul_mem.get_desc(),
          mul_B_md.submemory_desc({N, policy_d_model_, 8}, {0, 0, 0}),
          mul_A_md.submemory_desc({N, 4, 8}, {0, 0, 0}), common_attr);
#endif
      pmul_ = dnnl::matmul(pmul_pd);
      if (scratchpad_md.get_size() < pmul_pd.scratchpad_desc().get_size()) {
        scratchpad_md = pmul_pd.scratchpad_desc();
      }

      reorder_pd = dnnl::reorder::primitive_desc(
          eng, mul_A_md.submemory_desc({N, 4, 8}, {0, 0, 0}), eng, promo_md,
          common_attr);
      hack_reorder_2_ = dnnl::reorder(reorder_pd);
      if (scratchpad_md.get_size() < reorder_pd.scratchpad_desc().get_size()) {
        scratchpad_md = reorder_pd.scratchpad_desc();
      }

    } else {
#if DNNL_VERSION_MAJOR < 3
      auto pmul_d = dnnl::matmul::desc(
          pmul_mem.get_desc(),
          mul_B_md.submemory_desc({N, policy_d_model_, 8}, {0, 0, 56}),
          promo_md);
      auto pmul_pd = dnnl::matmul::primitive_desc(pmul_d, common_attr, eng);
#else
      auto pmul_pd = dnnl::matmul::primitive_desc(
          eng, pmul_mem.get_desc(),
          mul_B_md.submemory_desc({N, policy_d_model_, 8}, {0, 0, 56}),
          promo_md, common_attr);
#endif
      pmul_ = dnnl::matmul(pmul_pd);
      if (scratchpad_md.get_size() < pmul_pd.scratchpad_desc().get_size()) {
        scratchpad_md = pmul_pd.scratchpad_desc();
      }
    }

    auto in_reorder_pd = dnnl::reorder::primitive_desc(eng, input.get_desc(),
                                                       eng, in_md, common_attr);
    in_reorder_ = dnnl::reorder(in_reorder_pd);
    if (scratchpad_md.get_size() < in_reorder_pd.scratchpad_desc().get_size()) {
      scratchpad_md = in_reorder_pd.scratchpad_desc();
    }

    last_batch_ = N;
  }

  if (!scratchpad_mem ||
      scratchpad_mem.get_desc().get_size() < scratchpad_md.get_size()) {
    scratchpad_mem = dnnl::memory(scratchpad_md, eng);
  }

  dnnl::memory in;
  // Convert to NHWC.
  if (in_md != input.get_desc()) {
    in_reorder_.execute(stream, {{DNNL_ARG_SRC, input},
                                 {DNNL_ARG_DST, scratch},
                                 {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});
    in = scratch;
  } else {
    in = input;
  }

  if (out_md != output.get_desc()) {
    output = dnnl::memory(out_md, eng, output.get_data_handle());
  }

  fc_.execute(stream, {{DNNL_ARG_SRC, in},
                       {DNNL_ARG_WEIGHTS, fc_filter_mem},
                       {DNNL_ARG_BIAS, fc_bias_mem},
                       {DNNL_ARG_DST, fc_out_mem},
                       {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});

  fcQK_.execute(stream, {{DNNL_ARG_SRC, fc_out_mem},
                         {DNNL_ARG_WEIGHTS, fcQ_filter_mem},
                         {DNNL_ARG_BIAS, fcQ_bias_mem},
                         {DNNL_ARG_DST, fcQ_out_mem},
                         {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});

  fcQK_.execute(stream, {{DNNL_ARG_SRC, fc_out_mem},
                         {DNNL_ARG_WEIGHTS, fcK_filter_mem},
                         {DNNL_ARG_BIAS, fcK_bias_mem},
                         {DNNL_ARG_DST, fcK_out_mem},
                         {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});

  mul_.execute(stream, {{DNNL_ARG_SRC, fcQ_out_mem},
                        {DNNL_ARG_WEIGHTS, fcK_out_mem},
                        {DNNL_ARG_DST, output},
                        {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});

  if (eng.get_kind() == dnnl::engine::kind::gpu) {
    hack_reorder_.execute(stream, {{DNNL_ARG_SRC, fcK_out_mem},
                                   {DNNL_ARG_DST, fcQ_out_mem},
                                   {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});
    pmul_.execute(stream, {{DNNL_ARG_SRC, pmul_mem},
                           {DNNL_ARG_WEIGHTS, fcQ_out_mem},
                           {DNNL_ARG_DST, fcK_out_mem},
                           {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});
    hack_reorder_2_.execute(stream, {{DNNL_ARG_SRC, fcK_out_mem},
                                     {DNNL_ARG_DST, output},
                                     {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});

  } else {
    pmul_.execute(stream, {{DNNL_ARG_SRC, pmul_mem},
                           {DNNL_ARG_WEIGHTS, fcK_out_mem},
                           {DNNL_ARG_DST, output},
                           {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});
  }
}

}  // namespace onednn_backend
}  // namespace lczero
