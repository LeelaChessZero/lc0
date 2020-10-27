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
      use_skip_(skip) {
  biases = (float*)malloc(sizeof(float) * C);
  weights = (float*)malloc(C * c_input_ * filter_size_ * filter_size_ *
                           sizeof(float));
}

void ConvLayer::LoadWeights(float* pfilter, float* pBias, void* /*scratch*/) {
  memcpy(biases, pBias, sizeof(float) * C);
  memcpy(weights, pfilter,
         sizeof(float) * C * c_input_ * filter_size_ * filter_size_);
}

void ConvLayer::Eval(int N, float* output, const float* input,
                     void* /*scratch*/, size_t /*scratch_size*/,
                     dnnl::engine& eng, dnnl::stream& stream) {
  auto in_md =
      dnnl::memory::desc({N, c_input_, H, W}, dnnl::memory::data_type::f32,
                         dnnl::memory::format_tag::nchw);
  auto in_mem = dnnl::memory(in_md, eng, (void*)input);

  auto filter_md = dnnl::memory::desc({C, c_input_, filter_size_, filter_size_},
                                      dnnl::memory::data_type::f32,
                                      dnnl::memory::format_tag::oihw);
  auto filter_mem = dnnl::memory(filter_md, eng, weights);

  auto out_md = dnnl::memory::desc({N, C, H, W}, dnnl::memory::data_type::f32,
                                   dnnl::memory::format_tag::nchw);
  auto out_mem = dnnl::memory(out_md, eng, output);

  auto bias_md = dnnl::memory::desc({C}, dnnl::memory::data_type::f32,
                                    dnnl::memory::format_tag::a);
  auto bias_mem = dnnl::memory(bias_md, eng, biases);

  if (last_batch_ != N) {
    const int padding = filter_size_ / 2;
    auto conv_d = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_auto,
        in_md, filter_md, bias_md, out_md, {1, 1}, {padding, padding},
        {padding, padding});
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
    dnnl::memory::desc scratchpad_md = conv_pd.scratchpad_desc();
    size_t new_scratchpad_size = scratchpad_md.get_size();
    if (new_scratchpad_size > scratchpad_size) {
      scratchpad_size = new_scratchpad_size;
      free(scratchpad_ptr);
      scratchpad_ptr = malloc(new_scratchpad_size);
    }
    scratchpad_mem = dnnl::memory(scratchpad_md, eng, scratchpad_ptr);
    last_batch_ = N;
  }

  conv_.execute(stream, {{DNNL_ARG_SRC, in_mem},
                         {DNNL_ARG_WEIGHTS, filter_mem},
                         {DNNL_ARG_BIAS, bias_mem},
                         {DNNL_ARG_DST, out_mem},
                         {DNNL_ARG_SCRATCHPAD, scratchpad_mem}});
}

ConvLayer::~ConvLayer() {
  free(biases);
  free(weights);
  free(scratchpad_ptr);
}

SELayer::SELayer(BaseLayer* ip, int fc1Outputs)
    : BaseLayer(ip->GetC(), ip->GetH(), ip->GetW(), ip),
      numFc1Out_(fc1Outputs) {
  w1_ = (float*)malloc(C * numFc1Out_ * sizeof(float));
  w2_ = (float*)malloc(2 * C * numFc1Out_ * sizeof(float));

  b1_ = (float*)malloc(numFc1Out_ * sizeof(float));
  b2_ = (float*)malloc(2 * C * sizeof(float));
}

SELayer::~SELayer() {
  free(w1_);
  free(w2_);
  free(b1_);
  free(b2_);
}

void SELayer::LoadWeights(float* w1, float* b1, float* w2, float* b2,
                          void* /*scratch*/) {
  const size_t num_weights1 = C * numFc1Out_;
  const size_t weight_size1 = sizeof(float) * num_weights1;
  const size_t weight_size2 = 2 * weight_size1;

  memcpy(w1_, w1, weight_size1);
  memcpy(b1_, b1, numFc1Out_ * sizeof(float));

  memcpy(w2_, w2, weight_size2);
  memcpy(b2_, b2, 2 * C * sizeof(float));
}

void SELayer::Eval(int N, float* output, const float* input, void* scratch,
                   size_t /*scratch_size*/, dnnl::engine& eng,
                   dnnl::stream& stream) {
  float* pool = (float*)scratch;
  float* fc_out1 = (float*)scratch + 2 * C * N;

  auto in_md = dnnl::memory::desc({N, C, H, W}, dnnl::memory::data_type::f32,
                                  dnnl::memory::format_tag::nchw);
  auto in_mem = dnnl::memory(in_md, eng, (void*)input);
  auto out_mem = dnnl::memory(in_md, eng, (void*)output);

  auto pool_md = dnnl::memory::desc({N, C}, dnnl::memory::data_type::f32,
                                    dnnl::memory::format_tag::ab);
  auto pool_mem = dnnl::memory(pool_md, eng, (void*)pool);
  auto pool2_mem = dnnl::memory(pool_md, eng, (void*)(pool + N * C));

  auto filter_md =
      dnnl::memory::desc({numFc1Out_, C}, dnnl::memory::data_type::f32,
                         dnnl::memory::format_tag::ab);
  auto filter_mem = dnnl::memory(filter_md, eng, w1_);

  auto fc1out_md =
      dnnl::memory::desc({N, numFc1Out_}, dnnl::memory::data_type::f32,
                         dnnl::memory::format_tag::ab);
  auto fc1out_mem = dnnl::memory(fc1out_md, eng, fc_out1);

  auto bias_md = dnnl::memory::desc({numFc1Out_}, dnnl::memory::data_type::f32,
                                    dnnl::memory::format_tag::a);
  auto bias_mem = dnnl::memory(bias_md, eng, b1_);

  auto filter2_md =
      dnnl::memory::desc({C, numFc1Out_}, dnnl::memory::data_type::f32,
                         dnnl::memory::format_tag::ab);
  auto filter2a_mem = dnnl::memory(filter2_md, eng, w2_);
  auto filter2b_mem = dnnl::memory(filter2_md, eng, w2_ + C * numFc1Out_);

  auto bias2_md = dnnl::memory::desc({C}, dnnl::memory::data_type::f32,
                                     dnnl::memory::format_tag::a);
  auto bias2a_mem = dnnl::memory(bias2_md, eng, b2_);
  auto bias2b_mem = dnnl::memory(bias2_md, eng, b2_ + C);

  if (last_batch_ != N) {
    auto pooling_d = dnnl::pooling_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::pooling_avg, in_md,
        pool_md.reshape({N, C, 1, 1}), {1, 1}, {H, W}, {0, 0}, {0, 0});
    dnnl::primitive_attr pooling_attr;
    pooling_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto pooling_pd =
        dnnl::pooling_forward::primitive_desc(pooling_d, pooling_attr, eng);
    if (pooling_pd.scratchpad_desc().get_size() > 0) {
      throw Exception("SELayer does not support scparchpad memory");
    }
    pooling_ = dnnl::pooling_forward(pooling_pd);

    auto fc_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, pool_md, filter_md, bias_md,
        fc1out_md);
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

    auto fc2a_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, fc1out_md, filter2_md, bias2_md,
        pool_md);
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

    auto fc2b_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, fc1out_md, filter2_md, bias2_md,
        pool_md);
    dnnl::primitive_attr fc2b_attr;
    fc2b_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto fc2b_pd =
        dnnl::inner_product_forward::primitive_desc(fc2b_d, fc2b_attr, eng);
    if (fc2b_pd.scratchpad_desc().get_size() > 0) {
      throw Exception("SELayer does not support scparchpad memory");
    }
    fc2b_ = dnnl::inner_product_forward(fc2b_pd);

    auto mul_d = dnnl::binary::desc(dnnl::algorithm::binary_mul, in_md,
                                    pool_md.reshape({N, C, 1, 1}), in_md);
    dnnl::post_ops mul_ops;
    mul_ops.append_sum();
#if defined(DNNL_ARG_ATTR_MULTIPLE_POST_OP)
    mul_ops.append_binary(dnnl::algorithm::binary_add,
                          pool_md.reshape({N, C, 1, 1}));
    mul_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
#endif
    dnnl::primitive_attr mul_attr;
    mul_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    mul_attr.set_post_ops(mul_ops);
    auto mul_pd = dnnl::binary::primitive_desc(mul_d, mul_attr, eng);
    if (mul_pd.scratchpad_desc().get_size() > 0) {
      throw Exception("SELayer does not support scparchpad memory");
    }
    mul_ = dnnl::binary(mul_pd);

#if !defined(DNNL_ARG_ATTR_MULTIPLE_POST_OP)
    auto bias_add_d = dnnl::binary::desc(dnnl::algorithm::binary_add, in_md,
                                         pool_md.reshape({N, C, 1, 1}), in_md);
    dnnl::post_ops bias_add_ops;
    bias_add_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_relu, 0.0f,
                                0.0f);
    dnnl::primitive_attr bias_add_attr;
    bias_add_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    bias_add_attr.set_post_ops(bias_add_ops);
    auto bias_add_pd =
        dnnl::binary::primitive_desc(bias_add_d, bias_add_attr, eng);
    if (bias_add_pd.scratchpad_desc().get_size() > 0) {
      throw Exception("SELayer does not support scparchpad memory");
    }
    bias_add_ = dnnl::binary(bias_add_pd);
#endif
    last_batch_ = N;
  }

  pooling_.execute(stream, {{DNNL_ARG_SRC, in_mem}, {DNNL_ARG_DST, pool_mem}});

  fc_.execute(stream, {{DNNL_ARG_SRC, pool_mem},
                       {DNNL_ARG_WEIGHTS, filter_mem},
                       {DNNL_ARG_BIAS, bias_mem},
                       {DNNL_ARG_DST, fc1out_mem}});

  fc2a_.execute(stream, {{DNNL_ARG_SRC, fc1out_mem},
                         {DNNL_ARG_WEIGHTS, filter2a_mem},
                         {DNNL_ARG_BIAS, bias2a_mem},
                         {DNNL_ARG_DST, pool_mem}});

  fc2b_.execute(stream, {{DNNL_ARG_SRC, fc1out_mem},
                         {DNNL_ARG_WEIGHTS, filter2b_mem},
                         {DNNL_ARG_BIAS, bias2b_mem},
                         {DNNL_ARG_DST, pool2_mem}});

  mul_.execute(stream,
               {{DNNL_ARG_SRC_0, in_mem},
                {DNNL_ARG_SRC_1, pool_mem},
#if defined(DNNL_ARG_ATTR_MULTIPLE_POST_OP)
                {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, pool2_mem},
#endif
                {DNNL_ARG_DST, out_mem}});

#if !defined(DNNL_ARG_ATTR_MULTIPLE_POST_OP)
  bias_add_.execute(stream, {{DNNL_ARG_SRC_0, out_mem},
                             {DNNL_ARG_SRC_1, pool2_mem},
                             {DNNL_ARG_DST, out_mem}});
#endif
}

FCLayer::FCLayer(BaseLayer* ip, int C, int H, int W, bool relu, bool tanh)
    : BaseLayer(C, H, W, ip), use_relu_(relu), use_tanh_(tanh) {
  const size_t weight_size =
      sizeof(float) * C * H * W * ip->GetC() * ip->GetH() * ip->GetW();
  const size_t bias_size = sizeof(float) * C * H * W;

  weights_ = (float*)malloc(weight_size);
  biases_ = (float*)malloc(bias_size);
}

void FCLayer::LoadWeights(float* cpuWeight, float* cpuBias, void* /*scratch*/) {
  const size_t num_weights =
      C * H * W * input_->GetC() * input_->GetH() * input_->GetW();
  const size_t weight_size = sizeof(float) * num_weights;
  const size_t num_biases = C * H * W;
  const size_t bias_size = sizeof(float) * num_biases;

  memcpy(weights_, cpuWeight, weight_size);
  memcpy(biases_, cpuBias, bias_size);
}

void FCLayer::Eval(int N, float* output_tensor, const float* input_tensor,
                   void* /*scratch*/, size_t /*scratch_size*/,
                   dnnl::engine& eng, dnnl::stream& stream) {
  const int num_outputs = C * H * W;
  const int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

  auto in_md = dnnl::memory::desc({N, num_inputs}, dnnl::memory::data_type::f32,
                                  dnnl::memory::format_tag::ab);
  auto in_mem = dnnl::memory(in_md, eng, (void*)input_tensor);

  auto filter_md = dnnl::memory::desc({num_outputs, num_inputs},
                                      dnnl::memory::data_type::f32,
                                      dnnl::memory::format_tag::ab);
  auto filter_mem = dnnl::memory(filter_md, eng, weights_);

  auto out_md =
      dnnl::memory::desc({N, num_outputs}, dnnl::memory::data_type::f32,
                         dnnl::memory::format_tag::ab);
  auto out_mem = dnnl::memory(out_md, eng, output_tensor);

  auto bias_md = dnnl::memory::desc({num_outputs}, dnnl::memory::data_type::f32,
                                    dnnl::memory::format_tag::a);
  auto bias_mem = dnnl::memory(bias_md, eng, biases_);

  if (last_batch_ != N) {
    auto fc_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, in_md, filter_md, bias_md, out_md);
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
    last_batch_ = N;
  }
  fc_.execute(stream, {{DNNL_ARG_SRC, in_mem},
                       {DNNL_ARG_WEIGHTS, filter_mem},
                       {DNNL_ARG_BIAS, bias_mem},
                       {DNNL_ARG_DST, out_mem}});
}

FCLayer::~FCLayer() {
  free(weights_);
  free(biases_);
}

PolicyMapLayer::PolicyMapLayer(BaseLayer* ip, int C, int H, int W, int usedSize)
    : BaseLayer(C, H, W, ip), used_size_(usedSize) {
  size_t weight_size = sizeof(short) * used_size_ * 64;

  weights_ = (short*)malloc(weight_size);
}

void PolicyMapLayer::LoadWeights(const short* cpuWeight, void* /*scratch*/) {
  size_t weight_size = sizeof(short) * used_size_;

  memcpy(weights_, cpuWeight, weight_size);
}

void PolicyMapLayer::Eval(int N, float* output_tensor,
                          const float* input_tensor, void* /*scratch*/,
                          size_t /*scratch_size*/, dnnl::engine& /*eng*/,
                          dnnl::stream& /*stream*/) {
  int inputSize =
      this->input_->GetC() * this->input_->GetH() * this->input_->GetW();
  int outputSize = this->C * this->H * this->W;

  for (int batch = 0; batch < N; batch++) {
    for (int i = 0; i < used_size_; i++) {
      auto j = weights_[i];
      if (j >= 0) {
        output_tensor[batch * outputSize + j] =
            input_tensor[batch * inputSize + i];
      }
    }
  }
}

PolicyMapLayer::~PolicyMapLayer() {
  free(weights_);
}

}  // namespace dnnl_backend
}  // namespace lczero
