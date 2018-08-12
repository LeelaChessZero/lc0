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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include <cassert>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include "neural/factory.h"
#include "utils/bititer.h"
#include "utils/exception.h"

#include <algorithm>
#include <cstdio>
#include "neural/network_cuda.h"
#include "NvInfer.h"

namespace lczero {
  
  class Logger : public nvinfer1::ILogger
  {
  public:
      Logger(): Logger(Severity::kWARNING) {}
  
      Logger(Severity severity): reportableSeverity(severity) {}
  
      void log(Severity severity, const char* msg) override
      {
          // suppress messages with severity enum value greater than the reportable
          if (severity > reportableSeverity)
          {
             // lot of messages, no need to spam end-user
             //printf("\nTRT message: %s\n", msg);
          }
          else
          {
            char message[128];
            sprintf(message, "TRT error, severity: %d, message: %s", msg);
            throw Exception(message);
          }
      }
  
      Severity reportableSeverity{Severity::kWARNING};
  };

  class TRTNetwork;

  class TRTNetworkComputation : public NetworkComputation 
  {
  public:
    TRTNetworkComputation(TRTNetwork *network);
    ~TRTNetworkComputation();

    void AddInput(InputPlanes &&input) override 
    {
        auto iter_mask = &inputs_outputs_->input_masks_mem_[batch_size_ * kInputPlanes];
        auto iter_val = &inputs_outputs_->input_val_mem_[batch_size_ * kInputPlanes];

        int i = 0;
        for (const auto &plane : input) 
        {
            iter_mask[i] = plane.mask;
            iter_val[i] = plane.value;
            i++;
        }

        batch_size_++;
    }

    void ComputeBlocking() override;

    int GetBatchSize() const override { return batch_size_; }

    float GetQVal(int sample) const override 
    {
        return inputs_outputs_->op_value_mem_[sample];
    }

    float GetPVal(int sample, int move_id) const override 
    {
        return inputs_outputs_->op_policy_mem_[sample * kNumOutputPolicy + move_id];
    }

    private:
    // Memory holding inputs, outputs.
    std::unique_ptr<InputsOutputs> inputs_outputs_;
    int batch_size_;

    TRTNetwork *network_;
  }; // TRTNetworkComputation

  class TRTNetwork : public Network 
  {
  public:
    TRTNetwork(Weights weights, const OptionsDict &options) 
    {
        gpu_id_ = options.GetOrDefault<int>("gpu", 0);

        int total_gpus;
        ReportCUDAErrors(cudaGetDeviceCount(&total_gpus));

        if (gpu_id_ >= total_gpus)
            throw Exception("Invalid GPU Id: " + std::to_string(gpu_id_));

        // Select GPU to run on (for *the current* thread).
        ReportCUDAErrors(cudaSetDevice(gpu_id_));

        const int kNumFilters = weights.input.biases.size();
        numBlocks_ = weights.residual.size();

        // 0. Process weights.
        processConvBlock(weights.input, true);
        for (auto i = size_t{0}; i < numBlocks_; i++) 
        {
            processConvBlock(weights.residual[i].conv1, true);
            processConvBlock(weights.residual[i].conv2, true);
        }
        processConvBlock(weights.policy, true);
        processConvBlock(weights.value, true);

        // 1. allocate scratch space used to expand the input planes
        ReportCUDAErrors(cudaMalloc(&scratch_mem_, 128*1024*1024));

        // 2. Build the network
        static Logger gLogger;
        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
        nvinfer1::INetworkDefinition* network = builder->createNetwork();

        // Create input tensor of shape { 112, 8, 8 } with name "board"
        nvinfer1::ITensor* inputLayer = network->addInput("board", nvinfer1::DataType::kFLOAT, nvinfer1::Dims3{kInputPlanes, 8, 8});
        assert(inputLayer);

        // input conv
        const nvinfer1::Weights inputConvWeights{nvinfer1::DataType::kFLOAT, &weights.input.weights[0], kNumFilters*kInputPlanes*3*3};
        const nvinfer1::Weights inputConvBiases{nvinfer1::DataType::kFLOAT, &weights.input.biases[0], kNumFilters};
        auto inputConv = network->addConvolution(*inputLayer, kNumFilters, nvinfer1::DimsHW{3, 3}, inputConvWeights, inputConvBiases);
        inputConv->setPadding(nvinfer1::DimsHW{1, 1});
        auto inputConvRelu = network->addActivation(*inputConv->getOutput(0), nvinfer1::ActivationType::kRELU);

        nvinfer1::ITensor* lastLayer = inputConvRelu->getOutput(0);

        // residual block
        for (int block = 0; block < numBlocks_; block++) 
        {
            const nvinfer1::Weights conv1Weights{nvinfer1::DataType::kFLOAT, &weights.residual[block].conv1.weights[0], kNumFilters*kNumFilters*3*3};
            const nvinfer1::Weights conv1Biases{nvinfer1::DataType::kFLOAT, &weights.residual[block].conv1.biases[0], kNumFilters};
            auto conv1 = network->addConvolution(*lastLayer, kNumFilters, nvinfer1::DimsHW{3, 3}, conv1Weights, conv1Biases);
            conv1->setPadding(nvinfer1::DimsHW{1, 1});
            auto relu1 = network->addActivation(*conv1->getOutput(0), nvinfer1::ActivationType::kRELU);

            const nvinfer1::Weights conv2Weights{nvinfer1::DataType::kFLOAT, &weights.residual[block].conv2.weights[0], kNumFilters*kNumFilters*3*3};
            const nvinfer1::Weights conv2Biases{nvinfer1::DataType::kFLOAT, &weights.residual[block].conv2.biases[0], kNumFilters};
            auto conv2 = network->addConvolution(*relu1->getOutput(0), kNumFilters, nvinfer1::DimsHW{3, 3}, conv2Weights, conv2Biases);
            conv2->setPadding(nvinfer1::DimsHW{1, 1});
           
            auto residualAdd = network->addElementWise(*conv2->getOutput(0), *lastLayer, nvinfer1::ElementWiseOperation::kSUM);
            auto relu2 = network->addActivation(*residualAdd->getOutput(0), nvinfer1::ActivationType::kRELU);            
            
            lastLayer = relu2->getOutput(0);
        }

        nvinfer1::ITensor *resiOutTensor = lastLayer;
        resiOutTensor->setName("resiOut");

        // policy head
        const int kpolicyConvFilters =  weights.policy.bn_means.size();
        const nvinfer1::Weights convPolWeights{nvinfer1::DataType::kFLOAT, &weights.policy.weights[0], kpolicyConvFilters*kNumFilters*1*1};
        const nvinfer1::Weights convPolBiases{nvinfer1::DataType::kFLOAT, &weights.policy.biases[0], kpolicyConvFilters};
        auto convPol = network->addConvolution(*resiOutTensor, kpolicyConvFilters, nvinfer1::DimsHW{1, 1}, convPolWeights, convPolBiases);
        auto reluPol = network->addActivation(*convPol->getOutput(0), nvinfer1::ActivationType::kRELU);

        const int knumPolicyOutputs = weights.ip_pol_b.size();
        const nvinfer1::Weights fcPolWeights{nvinfer1::DataType::kFLOAT, &weights.ip_pol_w[0], knumPolicyOutputs*kpolicyConvFilters*8*8};
        const nvinfer1::Weights fcPolBiases{nvinfer1::DataType::kFLOAT, &weights.ip_pol_b[0], knumPolicyOutputs};
        auto fcPol = network->addFullyConnected(*reluPol->getOutput(0), knumPolicyOutputs, fcPolWeights, fcPolBiases);

        auto policyOut = network->addSoftMax(*fcPol->getOutput(0));
        nvinfer1::ITensor *policyOutTensor = policyOut->getOutput(0);
        policyOutTensor->setName("policyOut");
        

        // value head
        const int kvalueConvFilters =  weights.value.bn_means.size();
        const nvinfer1::Weights convValWeights{nvinfer1::DataType::kFLOAT, &weights.value.weights[0], kvalueConvFilters*kNumFilters*1*1};
        const nvinfer1::Weights convValBiases{nvinfer1::DataType::kFLOAT, &weights.value.biases[0], kvalueConvFilters};
        auto convVal = network->addConvolution(*resiOutTensor, kvalueConvFilters, nvinfer1::DimsHW{1, 1}, convValWeights, convValBiases);
        auto reluVal = network->addActivation(*convVal->getOutput(0), nvinfer1::ActivationType::kRELU);


        const int kValueFcNodes = weights.ip1_val_b.size();
        const nvinfer1::Weights fcVal1Weights{nvinfer1::DataType::kFLOAT, &weights.ip1_val_w[0], kValueFcNodes*kvalueConvFilters*8*8};
        const nvinfer1::Weights fcVal1Baises{nvinfer1::DataType::kFLOAT, &weights.ip1_val_b[0], kValueFcNodes};
        auto fcVal1 = network->addFullyConnected(*reluVal->getOutput(0), kValueFcNodes, fcVal1Weights, fcVal1Baises);
        auto reluValFc1 = network->addActivation(*fcVal1->getOutput(0), nvinfer1::ActivationType::kRELU);

        const nvinfer1::Weights fcVal2Weights{nvinfer1::DataType::kFLOAT, &weights.ip2_val_w[0], 1*kValueFcNodes};
        const nvinfer1::Weights fcVal2Biases{nvinfer1::DataType::kFLOAT, &weights.ip2_val_b[0], 1};
        auto fcVal2 = network->addFullyConnected(*reluValFc1->getOutput(0), 1, fcVal2Weights, fcVal2Biases);

        auto valueOut = network->addActivation(*fcVal2->getOutput(0), nvinfer1::ActivationType::kTANH);
        nvinfer1::ITensor *valueOutTensor = valueOut->getOutput(0);
        valueOutTensor->setName("valueOut");

        network->markOutput(*valueOutTensor);
        network->markOutput(*policyOutTensor);

        // Ankan - TODO: fix these hardcoded stuff
        const int max_batch = 256;
        const size_t maxScratchSize = 2*1024*1024*1024ull;

        builder->setMaxBatchSize(max_batch);
        builder->setMaxWorkspaceSize(maxScratchSize);

        // TODO: buildCudaEngine take a very long time (as TRT does all its optimizations inside this function)
        // Need to serialize engnie object and save it to file (for each weight file), and load it directly from file if present
        engine_ = builder->buildCudaEngine(*network);
        context_ = engine_->createExecutionContext();

        network->destroy();
        builder->destroy();
    }

    void forwardEval(InputsOutputs *io, int batchSize) 
    {
        std::lock_guard<std::mutex> lock(lock_);

        #ifdef DEBUG_RAW_NPS
        auto t_start = std::chrono::high_resolution_clock::now();
        #endif

        // expand packed planes to full planes
        uint64_t *ipDataMasks = io->input_masks_mem_gpu_;
        float *ipDataValues = io->input_val_mem_gpu_;
        expandPlanes_Fp32_NCHW((float *)scratch_mem_, ipDataMasks, ipDataValues, batchSize * kInputPlanes);

        float *opPol = io->op_policy_mem_gpu_;
        float *opVal = io->op_value_mem_gpu_;


        // run the network using TRT
        assert(engine_->getNbBindings() == 3);
        void* buffers[3];

        const int inputIndex = engine_->getBindingIndex("board");
        const int valueOutIndex = engine_->getBindingIndex("valueOut");
        const int policyOutIndex = engine_->getBindingIndex("policyOut");

        buffers[inputIndex] = scratch_mem_;
        buffers[valueOutIndex] = opVal;
        buffers[policyOutIndex] = opPol;

        context_->enqueue(batchSize, buffers, (cudaStream_t) 0, nullptr);
        cudaDeviceSynchronize();

        #ifdef DEBUG_RAW_NPS
        const int reportingCalls = 100;
        static int numCalls = 0;
        static int sumBatchSize = 0;
        static double totalTime = 0;

        sumBatchSize += batchSize;
        numCalls++;

        auto t_end = std::chrono::high_resolution_clock::now();

        double dt = std::chrono::duration<double>(t_end - t_start).count();
        totalTime += dt;
        if (numCalls == reportingCalls) 
        {
            double avgBatchSize = ((double)sumBatchSize) / numCalls;
            printf("\nAvg batch size: %lf, NN eval time: %lf seconds per %d evals\n",
                    avgBatchSize, totalTime, sumBatchSize);
            sumBatchSize = 0;
            totalTime = 0;
            numCalls = 0;
        }
        #endif
    }

    ~TRTNetwork() 
    {
        if (scratch_mem_) ReportCUDAErrors(cudaFree(scratch_mem_));
        context_->destroy();
        engine_->destroy();
    }

    std::unique_ptr<NetworkComputation> NewComputation() override 
    {
        // set correct gpu id for this computation (as it might have been called
        // from a different thread)
        ReportCUDAErrors(cudaSetDevice(gpu_id_));
        return std::make_unique<TRTNetworkComputation>(this);
    }

    std::unique_ptr<InputsOutputs> GetInputsOutputs() 
    {
        std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
        if (free_inputs_outputs_.empty()) {
            return std::make_unique<InputsOutputs>();
        } else {
            std::unique_ptr<InputsOutputs> resource =
                std::move(free_inputs_outputs_.front());
            free_inputs_outputs_.pop_front();
            return resource;
        }
    }

    void ReleaseInputsOutputs(std::unique_ptr<InputsOutputs> resource) 
    {
        std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
        free_inputs_outputs_.push_back(std::move(resource));
    }

  private:
    int gpu_id_;
    nvinfer1::IExecutionContext* context_;
    nvinfer1::ICudaEngine* engine_;

    // currently only one NN Eval can happen a time (we can fix this if needed by
    // allocating more memory)
    mutable std::mutex lock_;

    int numBlocks_;
    void *scratch_mem_;
    size_t scratch_size_;   // unused?

    mutable std::mutex inputs_outputs_lock_;
    std::list<std::unique_ptr<InputsOutputs>> free_inputs_outputs_;
};

    TRTNetworkComputation::TRTNetworkComputation(TRTNetwork *network)
    : network_(network) 
    {
        batch_size_ = 0;
        inputs_outputs_ = network_->GetInputsOutputs();
    }

    TRTNetworkComputation::~TRTNetworkComputation() 
    {
        network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
    }

    void TRTNetworkComputation::ComputeBlocking() 
    {
        network_->forwardEval(inputs_outputs_.get(), GetBatchSize());
    }

    REGISTER_NETWORK("trt", TRTNetwork, 120)
} // namespace lczero