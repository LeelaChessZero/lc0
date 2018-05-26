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
#include "utils/blas.h"

namespace lczero {
  
  namespace {
    
    
    static constexpr int NUM_VALUE_INPUT_PLANES = 32;
    static constexpr int NUM_POLICY_INPUT_PLANES = 32;
    
    static constexpr int NUM_OUTPUT_POLICY = 1858;
    static constexpr int NUM_VALUE_CHANNELS = 128;
    
    static constexpr auto WINOGRAD_ALPHA = 4;
    static constexpr auto WINOGRAD_TILE = WINOGRAD_ALPHA * WINOGRAD_ALPHA;

    class BlasNetwork;
    
    
    
    class BlasComputation : public NetworkComputation {
      
    public:
      
      BlasComputation(const Weights& weights):
      weights_(weights),
      input_data_(kInputPlanes*64),
      value_data_(NUM_VALUE_CHANNELS),
      policy_data_(),
      q_value_(0) {
        
      }
      
      virtual ~BlasComputation() {
        
      }
      
      // Adds a sample to the batch.
      void AddInput(InputPlanes&& input) override {
        planes_.emplace_back(input);
      }

            
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
        forward(input_data_, policy_data, value_data_);

        
/*        for (int i=0; i<value_data_.size(); i++) {
          std::cerr<<value_data_[i]<<"  ";
        }
        std::cerr<<std::endl;
 */

        // Get the moves
        Transforms::softmax(policy_data, policy_data);
        
  
        policy_data_.emplace_back(move(policy_data));
        
        // Now get the score
        const std::vector<float>& ip2_val_w=weights_.ip2_val_w;
        const std::vector<float>& ip2_val_b=weights_.ip2_val_b;
        
        double winrate=Transforms::innerproduct(ip2_val_w, value_data_)+ip2_val_b[0];
//        std::cerr<<"win rate"<<winrate <<std::endl;

        q_value_.emplace_back(std::tanh(winrate));

      }
      
      
      void forward(std::vector<float>& input,
                           std::vector<float>& output_pol,
                           std::vector<float>& output_val) {
        
        // Input convolution
        constexpr int width = 8;
        constexpr int height = 8;
        constexpr int tiles = width * height / 4;
        
        const std::vector<float>& input_conv_biases=weights_.input.biases;
        const std::vector<float>& input_conv_weights=weights_.input.weights;
        
        // Calculate output channels
        const auto output_channels = input_conv_biases.size();
        //input_channels is the maximum number of input channels of any convolution.
        //Residual blocks are identical, but the first convolution might be bigger
        //when the network has very few filters
        const auto input_channels = std::max(
                                             static_cast<size_t>(output_channels),
                                             static_cast<size_t>(kInputPlanes));
        auto conv_out = std::vector<float>(output_channels * width * height);
        
        auto V = std::vector<float>(WINOGRAD_TILE * input_channels * tiles);
        auto M = std::vector<float>(WINOGRAD_TILE * output_channels * tiles);
        
        std::vector<float> policy_data(NUM_POLICY_INPUT_PLANES * width * height);
        std::vector<float> value_data(NUM_VALUE_INPUT_PLANES * width * height);
        
        Transforms::winograd_convolve3(output_channels, input, input_conv_weights, V, M, conv_out);
        Transforms::batchnorm<64>(output_channels, conv_out,
                                  weights_.input.bn_means.data(),
                                  weights_.input.bn_stddivs.data());
        
        // Residual tower
        auto conv_in = std::vector<float>(output_channels * width * height);
        auto res = std::vector<float>(output_channels * width * height);
        
        for (auto &residual : weights_.residual) {
          
          auto& conv1=residual.conv1;
          auto output_channels = conv1.biases.size();
          std::swap(conv_out, conv_in);
          std::copy(begin(conv_in), end(conv_in), begin(res));
          
          Transforms::winograd_convolve3(output_channels, conv_in,
                                         conv1.weights, V, M, conv_out);
          Transforms::batchnorm<64>(output_channels, conv_out,
                                    conv1.bn_means.data(),
                                    conv1.bn_stddivs.data());
          
          auto& conv2=residual.conv2;
          output_channels = conv2.biases.size();
          std::swap(conv_out, conv_in);
          Transforms::winograd_convolve3(output_channels, conv_in,
                                         conv2.weights, V, M, conv_out);
          Transforms::batchnorm<64>(output_channels, conv_out,
                                    conv2.bn_means.data(),
                                    conv2.bn_stddivs.data(),
                                    res.data());
        }

        auto conv_pol_w=weights_.policy.weights;
        auto conv_pol_b=weights_.policy.biases;
        Transforms::convolve<1>(NUM_POLICY_INPUT_PLANES, conv_out, conv_pol_w, conv_pol_b, policy_data);
        
        auto conv_val_w=weights_.value.weights;
        auto conv_val_b=weights_.value.biases;
        Transforms::convolve<1>(NUM_VALUE_INPUT_PLANES, conv_out, conv_val_w, conv_val_b, value_data);
        
        auto bn_pol_w1=weights_.policy.bn_means;
        auto bn_pol_w2=weights_.policy.bn_stddivs;
        Transforms::batchnorm<width*height>(NUM_POLICY_INPUT_PLANES, policy_data, bn_pol_w1.data(), bn_pol_w2.data());
        
        auto bn_val_w1=weights_.value.bn_means;
        auto bn_val_w2=weights_.value.bn_stddivs;
        Transforms::batchnorm<width*height>(NUM_VALUE_INPUT_PLANES, value_data, bn_val_w1.data(), bn_val_w2.data());
        
        auto v2_ip_pol_w=weights_.ip_pol_w;
        auto v2_ip_pol_b=weights_.ip_pol_b;
        Transforms::innerproduct(NUM_POLICY_INPUT_PLANES*width*height, NUM_OUTPUT_POLICY, policy_data, v2_ip_pol_w, v2_ip_pol_b, output_pol);
        
        auto ip1_val_w=weights_.ip1_val_w;
        auto ip1_val_b=weights_.ip1_val_b;
        Transforms::innerproduct(NUM_VALUE_INPUT_PLANES*width*height, NUM_VALUE_CHANNELS, value_data, ip1_val_w, ip1_val_b, output_val);
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
  
      const Weights& weights_;
      
      std::vector<InputPlanes> planes_;
      std::vector<float> input_data_;
      std::vector<float> value_data_;
      
      std::vector<std::vector<float>> policy_data_;
      std::vector<float> q_value_;
      
    };
    
    class BlasNetwork : public Network {
    public:
      
      virtual ~BlasNetwork(){};

      
      BlasNetwork(const Weights& weights, const OptionsDict& /* options */):
      weights_(weights)
      {
        constexpr float EPSILON=1e-5;

        const int inputChannels = kInputPlanes;
        const int channels = weights.input.biases.size();
        const size_t residual_blocks = weights.residual.size();
        
        weights_.input.weights=Transforms::winograd_transform_f(weights_.input.weights, channels, inputChannels);
        
        std::vector<float>& input_batchnorm_means=weights_.input.bn_means;
        // Biases are not calculated and are typically zero but some networks might
        // still have non-zero biases.
        // Move biases to batchnorm means to make the output match without having
        // to separately add the biases.
        for (int i=0; i<input_batchnorm_means.size(); i++)
          input_batchnorm_means[i]-=weights_.input.biases[i];
        
        std::vector<float>& input_batchnorm_stddivs=weights_.input.bn_stddivs;
        for(auto&& w : input_batchnorm_stddivs)
          w = 1.0f / std::sqrt(w + EPSILON);
        
        // residual blocks
        for (auto i = 0; i < residual_blocks; i++) {
          
          Weights::Residual& residual=weights_.residual[i];
          Weights::ConvBlock& conv1=residual.conv1;
          Weights::ConvBlock& conv2=residual.conv2;
          
          conv1.weights=Transforms::winograd_transform_f(conv1.weights, channels, channels);
          conv2.weights=Transforms::winograd_transform_f(conv2.weights, channels, channels);
          
          std::vector<float>& batchnorm_means_1=conv1.bn_means;
          for (int i=0; i<batchnorm_means_1.size(); i++)
            batchnorm_means_1[i]-=conv1.biases[i];
          
          std::vector<float>& batchnorm_means_2=conv2.bn_means;
          for (int i=0; i<batchnorm_means_2.size(); i++)
            batchnorm_means_2[i]-=conv2.biases[i];
          
          std::vector<float>& batchnorm_stddivs_1=conv1.bn_stddivs;
          for(auto&& w : batchnorm_stddivs_1)
            w = 1.0f / std::sqrt(w + EPSILON);
          
          std::vector<float>& batchnorm_stddivs_2=conv2.bn_stddivs;
          for(auto&& w : batchnorm_stddivs_2)
            w = 1.0f / std::sqrt(w + EPSILON);
          
        }
        
        std::vector<float>& bn_pol_means=weights_.policy.bn_means;
        for (int i=0; i<bn_pol_means.size(); i++)
          bn_pol_means[i]-=weights_.policy.biases[i];
        
        std::vector<float>& bn_pol_stddivs=weights_.policy.bn_stddivs;
        for(auto&& w : bn_pol_stddivs)
          w = 1.0f / std::sqrt(w + EPSILON);
        
        std::vector<float>& bn_val_means=weights_.value.bn_means;
        for (int i=0; i<bn_val_means.size(); i++)
          bn_val_means[i]-=weights_.value.biases[i];
        
        std::vector<float>& bn_val_stddivs=weights_.value.bn_stddivs;
        for(auto&& w : bn_val_stddivs)
          w = 1.0f / std::sqrt(w + EPSILON);

#ifdef USE_OPENBLAS
        //openblas_set_num_threads(1);
        //printf("BLAS Core: %s\n", openblas_get_corename());
#endif
        
#ifdef USE_MKL
        //mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
        mkl_set_num_threads(1);
        MKLVersion Version;
        mkl_get_version(&Version);
        printf("BLAS core: MKL %s\n", Version.Processor);
#endif

      }
      
      std::unique_ptr<NetworkComputation> NewComputation() override {
        return std::make_unique<BlasComputation>(weights_);
      }
      
      
    private:
      
      Weights weights_;
      
    };
      
    
    
  } // namespace
  
  REGISTER_NETWORK("blas", BlasNetwork, 50)
  
  
} // namespace lc0

