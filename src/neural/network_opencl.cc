/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2018 The LCZero Authors
 
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

#include "neural/factory.h"
#include "neural/network.h"
#include "neural/transforms.h"

#include <condition_variable>
#include <cassert>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <thread>

#include "utils/exception.h"
#include "utils/bititer.h"

#include "CL/OpenCLUtils.h"

#include "CL/OpenCLParams.h"
#include "CL/OpenCL.h"


#include "utils/blas.h"


namespace lczero {
  
  namespace {
    
    
    static constexpr int NUM_VALUE_INPUT_PLANES = 32;
    static constexpr int NUM_POLICY_INPUT_PLANES = 32;
    
    static constexpr int NUM_OUTPUT_POLICY = 1858;
    static constexpr int NUM_VALUE_CHANNELS = 128;
    
    
    class OpenCLNetwork;
    
    // Copy the vectors we need after weights is deallocated
    struct OpenCLWeights {
      
      const std::vector<float> ip2_val_w;
      const std::vector<float> ip2_val_b;
      
      OpenCLWeights(const Weights& weights):
      ip2_val_w(weights.ip2_val_w),
      ip2_val_b(weights.ip2_val_b)
      {
        
      }

    };
    
    
    
    class OpenCLComputation : public NetworkComputation {
      
    public:
      
      OpenCLComputation(const OpenCL_Network& opencl_net, const OpenCLWeights& weights):
      opencl_net_(opencl_net),
      weights_(weights),
      input_data_(kInputPlanes*64),
      value_data_(NUM_VALUE_CHANNELS),
      policy_data_(),
      q_value_(0) {
        
      }
      
      virtual ~OpenCLComputation() {
        
      }
      
      // Adds a sample to the batch.
      void AddInput(InputPlanes&& input) override {
        planes_.emplace_back(input);
      }

      
    public:
      
     // Do the computation.
      void ComputeBlocking() override {
        
        for (auto& sample : planes_)
          ComputeBlocking(sample);

      }
      
      
      void ComputeBlocking(const InputPlanes &sample) {
        
        
        int index=0;
        for (const InputPlane& plane : sample) {
          float value=plane.value;
          const uint64_t one=1;
          for (int i=0; i<64; i++)
            input_data_[index++]=((plane.mask&(one<<i))==0 ) ? 0 : value;
        }
        
        std::vector<float> policy_data(NUM_OUTPUT_POLICY);
        opencl_net_.forward(input_data_, policy_data, value_data_);
        
        // Get the moves
        softmax(policy_data, policy_data);
        policy_data_.emplace_back(move(policy_data));
        
        // Now get the score
        const std::vector<float>& ip2_val_w=weights_.ip2_val_w;
        const std::vector<float>& ip2_val_b=weights_.ip2_val_b;
        
        double winrate=innerproduct(ip2_val_w, value_data_)+ip2_val_b[0];
        q_value_.emplace_back(std::tanh(winrate));
        
      }
      
      // Returns how many times AddInput() was called.
      int GetBatchSize() const override {
         return planes_.size();
      }
      
      // Returns Q value of @sample.
      float GetQVal(int sample) const override {
        return q_value_[sample];
      }
      
      // Returns P value @move_id of @sample.
      float GetPVal(int sample, int move_id) const override {
        return policy_data_[sample][move_id];
      }
      
      
    private:
  
      
      static void softmax(const std::vector<float>& input,
                          std::vector<float>& output) {
        
        auto alpha = *std::max_element(begin(input),
                                       begin(input) + input.size());
        
        auto denom = 0.0f;
        for (auto i = size_t{0}; i < output.size(); i++) {
          auto val   = std::exp(input[i] - alpha);
          output[i]  = val;
          denom     += val;
        }
        for (auto i = size_t{0}; i < output.size(); i++) {
          output[i] = output[i] / denom;
        }
      }
      
      
      static float innerproduct(const std::vector<float>& x, const std::vector<float>& y) {
        // float cblas_sdot(const int __N, const float *__X, const int __incX, const float *__Y, const int __incY);
        return cblas_sdot(x.size(), &x[0], 1, &y[0], 1);
      }
      

      const OpenCL_Network& opencl_net_;
      const OpenCLWeights& weights_;
      
      std::vector<InputPlanes> planes_;
      std::vector<float> input_data_;
      std::vector<float> value_data_;
      
      std::vector<std::vector<float>> policy_data_;
      std::vector<float> q_value_;
      
    };
    
    static constexpr auto WINOGRAD_ALPHA = 4;
    
    class OpenCLNetwork : public Network {
    public:
      
      virtual ~OpenCLNetwork(){};

      OpenCLNetwork(const Weights& weights, const OptionsDict& options):
      weights_(weights),
      params_(),
      opencl_(),
      opencl_net_(opencl_)
      {
        
        params_.gpuId=options.GetOrDefault<int>("gpu", -1);
        params_.verbose=options.GetOrDefault<bool>("verbose", false);
        params_.force_tune=options.GetOrDefault<int>("force_tune", false);
        params_.tune_only=options.GetOrDefault<int>("tune_only", false);
        params_.tune_exhaustive=options.GetOrDefault<int>("tune_exhaustive", false);
        
        constexpr float EPSILON=1e-5;
        
        const int inputChannels = kInputPlanes;
        const int channels = weights.input.biases.size();
        const size_t residual_blocks = weights.residual.size();
        
        
        opencl_.initialize(channels, params_);
        
        auto tuners = opencl_.get_sgemm_tuners();
        
        auto mwg = tuners[0];
        auto kwg = tuners[2];
        auto vwm = tuners[3];
        
        
        size_t m_ceil = ceilMultiple(ceilMultiple(channels, mwg), vwm);
        size_t k_ceil = ceilMultiple(ceilMultiple(inputChannels, kwg), vwm);
        
        std::vector<float> input_conv_weights=Transforms::winograd_transform_f(weights.input.weights, channels, inputChannels);
        
        auto Upad = Transforms::zeropad_U(input_conv_weights,
                              channels, inputChannels,
                              m_ceil, k_ceil);
        
        // Biases are not calculated and are typically zero but some networks might
        // still have non-zero biases.
        // Move biases to batchnorm means to make the output match without having
        // to separately add the biases.
        std::vector<float> input_batchnorm_means=weights.input.bn_means; // copy ctor
        for (int i=0; i<input_batchnorm_means.size(); i++)
          input_batchnorm_means[i]-=weights.input.biases[i];
        
        std::vector<float> input_batchnorm_stddivs=weights.input.bn_stddivs;
        for(auto&& w : input_batchnorm_stddivs)
          w = 1.0f / std::sqrt(w + EPSILON);
        
        // Winograd filter transformation changes filter size to 4x4
        opencl_net_.push_input_convolution(WINOGRAD_ALPHA, inputChannels, channels,
                                           Upad, input_batchnorm_means, input_batchnorm_stddivs);
        
        // residual blocks
        for (auto i = size_t{0}; i < residual_blocks; i++) {
          
          const Weights::Residual& residual=weights.residual[i];
          const Weights::ConvBlock& conv1=residual.conv1;
          const Weights::ConvBlock& conv2=residual.conv2;
          
          std::vector<float> conv_weights_1=Transforms::winograd_transform_f(conv1.weights, channels, channels);
          std::vector<float> conv_weights_2=Transforms::winograd_transform_f(conv2.weights, channels, channels);
          
          auto Upad1 = Transforms::zeropad_U(conv_weights_1,
                                 channels, channels,
                                 m_ceil, m_ceil);
          auto Upad2 = Transforms::zeropad_U(conv_weights_2,
                                 channels, channels,
                                 m_ceil, m_ceil);
          
          
          // Biases are not calculated and are typically zero but some networks might
          // still have non-zero biases.
          // Move biases to batchnorm means to make the output match without having
          // to separately add the biases.
          std::vector<float> batchnorm_means_1=conv1.bn_means; // copy ctor
          for (int i=0; i<batchnorm_means_1.size(); i++)
            batchnorm_means_1[i]-=conv1.biases[i];
          
          std::vector<float> batchnorm_means_2=conv2.bn_means;
          for (int i=0; i<batchnorm_means_2.size(); i++)
            batchnorm_means_2[i]-=conv2.biases[i];
          
          std::vector<float> batchnorm_stddivs_1=conv1.bn_stddivs;
          for(auto&& w : batchnorm_stddivs_1)
            w = 1.0f / std::sqrt(w + EPSILON);
          
          std::vector<float> batchnorm_stddivs_2=conv2.bn_stddivs;
          for(auto&& w : batchnorm_stddivs_2)
            w = 1.0f / std::sqrt(w + EPSILON);
          
          
          opencl_net_.push_residual(WINOGRAD_ALPHA, channels, channels,
                                    Upad1,
                                    batchnorm_means_1,
                                    batchnorm_stddivs_1,
                                    Upad2,
                                    batchnorm_means_2,
                                    batchnorm_stddivs_2);
        }
        
        constexpr unsigned int width = 8;
        constexpr unsigned int height = 8;
        
        
        const std::vector<float>& conv_pol_w=weights.policy.weights;
        std::vector<float> bn_pol_means=weights.policy.bn_means;
        for (int i=0; i<bn_pol_means.size(); i++)
          bn_pol_means[i]-=weights.policy.biases[i];
        
        std::vector<float> bn_pol_stddivs=weights.policy.bn_stddivs;
        for(auto&& w : bn_pol_stddivs)
          w = 1.0f / std::sqrt(w + EPSILON);
        
        
        const std::vector<float>& ip_pol_w_vec=weights.ip_pol_w;
        const std::vector<float>& ip_pol_b_vec=weights.ip_pol_b;
        
        opencl_net_.push_policy(channels, NUM_POLICY_INPUT_PLANES,
                                NUM_POLICY_INPUT_PLANES*width*height, NUM_OUTPUT_POLICY,
                                conv_pol_w,
                                bn_pol_means, bn_pol_stddivs,
                                ip_pol_w_vec, ip_pol_b_vec);
        
        const std::vector<float>& conv_val_w=weights.value.weights;
        std::vector<float> bn_val_means=weights.value.bn_means;
        for (int i=0; i<bn_val_means.size(); i++)
          bn_val_means[i]-=weights.value.biases[i];
        
        std::vector<float> bn_val_stddivs=weights.value.bn_stddivs;
        for(auto&& w : bn_val_stddivs)
          w = 1.0f / std::sqrt(w + EPSILON);
        
        const std::vector<float>& ip_val_w_vec=weights.ip1_val_w;
        const std::vector<float>& ip_val_b_vec=weights.ip1_val_b;
        
        opencl_net_.push_value(channels, NUM_VALUE_INPUT_PLANES,
                               NUM_VALUE_INPUT_PLANES*width*height, NUM_VALUE_CHANNELS,
                               conv_val_w,
                               bn_val_means, bn_val_stddivs,
                               ip_val_w_vec, ip_val_b_vec);
        
        
        
      }
      
      std::unique_ptr<NetworkComputation> NewComputation() override {
        return std::make_unique<OpenCLComputation>(opencl_net_, weights_);
      }
      
      
    private:
      
      OpenCLWeights weights_;
      OpenCLParams params_;
      OpenCL opencl_;
      OpenCL_Network opencl_net_;
      
    };
      
    
    
  } // namespace
  
  REGISTER_NETWORK("opencl", OpenCLNetwork, 100)
  
  
} // namespace lc0

