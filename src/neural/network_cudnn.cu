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
#include "network_cudnn.h"
#include "utils/bititer.h"
#include "utils/exception.h"
#include <functional>
#include <cassert>
#include <mutex>

#include <cublas_v2.h>
#include <cudnn.h>

namespace lczero {

void cudnnError(cudnnStatus_t status, const char* file, const int& line)
{
    if (status != CUDNN_STATUS_SUCCESS)
    {
        char message[128];
        sprintf(message, "CUDNN error: %s (%s:%d) ", cudnnGetErrorString(status), file, line);
        throw(new Exception(message));
    }
}

#define reportCUDNNErrors(status)  cudnnError(status, __FILE__, __LINE__)


// 256 MB fixed scratch memory size (hardcoded for now)
static constexpr int kCudaScratchSize = 256 * 1024 * 1024;

// hard-coded for now, no point in going above this anyway (can possibly save memory by reducing this)
static constexpr int kMaxBatchSize = 1024;

// the Layer objects only hold memory for weights, biases, etc
// memory for input and output tensors is provided by caller of forwardEval

class BaseLayer
{
protected:
    static bool fp16;
    static size_t bpe;  // size of each element
    int C, H, W;        // output tensor dimensions
    BaseLayer *input;

public:
    int getC() { return C; }
    int getH() { return H; }
    int getW() { return W; }

    BaseLayer(int c, int h, int w, BaseLayer *ip);

    // input2 is optional (skip connection)
    size_t getOutputSize(int N) { return bpe * N * C * H * W; }
    virtual void eval(int N, void *output, void *input, void *input2, void *scratch, cudnnHandle_t cudnn, cublasHandle_t cublas) = 0;
};

class ConvLayer : public BaseLayer
{
private:
    int Cinput;
    int filterSize;

    void *biases;
    void *weights;

    bool useRelu;
    bool useBias;

    cudnnFilterDescriptor_t         filterDesc;
    cudnnConvolutionDescriptor_t    convDesc;
    cudnnConvolutionFwdAlgo_t       convAlgo;

    cudnnTensorDescriptor_t         biasDesc;
    cudnnTensorDescriptor_t         inTensorDesc;
    cudnnTensorDescriptor_t         outTensorDesc;
    cudnnActivationDescriptor_t     activation;


public:
    ConvLayer(BaseLayer *ip, int C, int H, int W, int size, int Cin, bool relu = false, bool bias = false);
    ~ConvLayer();
    void loadWeights(void *pfilter, void *pBias = NULL);
    void eval(int N, void *output, void *input, void *input2, void *scratch, cudnnHandle_t cudnn, cublasHandle_t cublas) override;
};

class SoftMaxLayer : public BaseLayer
{
private:
    cudnnTensorDescriptor_t         outTensorDesc;

public:
    SoftMaxLayer(BaseLayer *ip);
    void eval(int N, void *output, void *input, void *input2, void *scratch, cudnnHandle_t cudnn, cublasHandle_t cublas) override;
};

class BNLayer : public BaseLayer
{
private:
    void *means;
    void *variances;
    bool useRelu;

public:
    BNLayer(BaseLayer *ip, bool relu);
    ~BNLayer();

    void loadWeights(void *cpuMeans, void *cpuVar);
    void eval(int N, void *output, void *input, void *input2, void *scratch, cudnnHandle_t cudnn, cublasHandle_t cublas) override;

};

class FCLayer : public BaseLayer
{
private:
    void *weights;
    void *biases;
    bool useBias;
    bool useRelu;
    bool useTanh;

public:
    FCLayer(BaseLayer *ip, int C, int H, int W, bool relu, bool bias, bool tanh = false);
    ~FCLayer();

    void loadWeights(void *cpuWeight, void *cpuBias);
    void eval(int N, void *output, void *input, void *input2, void *scratch, cudnnHandle_t cudnn, cublasHandle_t cublas) override;
};


// Each residual block has (4 kernels per block)
// A convolution of 128 filters of kernel size 3 × 3 with stride 1

// Batch normalisation
// A rectifier non - linearity

// A convolution of 128 filters of kernel size 3 × 3 with stride 1

// Batch normalisation
// A skip connection that adds the input to the block
// A rectifier non - linearity

// need implementations of 
//  1. convolution layer (no bias/activation needed)
//  2. Fully connected layer (with optional bias, and optional relu),
//  3. batch normilization with optional sum (skip connection) and RELU

// Need memory for 3 data buffers
//  1. input for the layer
//  2. output of the layer
//  3. data from old layer for skip connection


/////////////////////////////////////////////////////////////////////////////
//                      Static variable Definations                        //
/////////////////////////////////////////////////////////////////////////////

// TODO: fp16 support
bool BaseLayer::fp16 = false;
size_t BaseLayer::bpe = sizeof(float);


int divUp(int a, int b)
{
    return (a + b - 1) / b;
}


/////////////////////////////////////////////////////////////////////////////
//          Simple CUDA kernels used by certain layers                     //
/////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void addVectors_kernel(T *c, T *a, T *b, int size, int asize, int bsize, bool relu, bool useTanh)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size)
    {
        T aVal = 0;
        T bVal = 0;
        if (a) aVal = a[i % asize];
        if (b) bVal = b[i % bsize];

        T cVal = aVal + bVal;

        if (relu && (cVal < 0))
            cVal = 0;

        if (useTanh)
        {
            // Ankan: actually it's sigmoid in leela-zero main branch??
            // see code in Network.cpp    
            //    auto winrate_sig = (1.0f + std::tanh(winrate_out[0])) / 2.0f;
            // Different from lc0 branch? WHY ???
            // cVal = (1.0f + tanh(cVal)) / 2.0f;
            cVal = tanh(cVal);
        }

        c[i] = cVal;
    }

}

// adds two vectors (possibly of different sizes), also do optional relu activation
template <typename T>
void addVectors(T *c, T *a, T *b, int size, int asize, int bsize, bool relu, bool useTanh)
{
    const int blockSize = 256;
    int blocks = divUp(size, blockSize);

    addVectors_kernel << <blocks, blockSize >> > (c, a, b, size, asize, bsize, relu, useTanh);
}



__global__ void batchNormForward_kernel(float *output, const float *input, const float *skipInput, int N, int C, int H, int W, const float *means, const float *varMultipliers, bool relu)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int wIndex = (index / (H * W)) % C;

    float el = input[index];
    float mean = means[wIndex];
    float varMulti = varMultipliers[wIndex];

    el -= mean;
    el *= varMulti;

    // TODO: figure out order of relu and skip connection
    if (skipInput)
        el += skipInput[index];

    if (relu && (el < 0))
        el = 0;

    output[index] = el;
}


// works only on NCHW tensors
// each thread processes single element
void batchNormForward(float *output, float *input, float *skipInput, int N, int C, int H, int W, float *means, float *varMultipliers, bool relu)
{
    int totalElements = N * C * H * W;
    const int blockSize = 256;
    int blocks = divUp(totalElements, blockSize);

    batchNormForward_kernel << <blocks, blockSize >> > (output, input, skipInput, N, C, H, W, means, varMultipliers, relu);
}


BaseLayer::BaseLayer(int c, int h, int w, BaseLayer *ip) :
    C(c),
    H(h),
    W(w),
    input(ip)
{

}

SoftMaxLayer::SoftMaxLayer(BaseLayer *ip) :
    BaseLayer(ip->getC(), ip->getH(), ip->getW(), ip)
{
    cudnnCreateTensorDescriptor(&outTensorDesc);
}

void SoftMaxLayer::eval(int N, void *output, void *input, void *input2, void *scratch, cudnnHandle_t cudnn, cublasHandle_t cublas)
{
    float alpha = 1.0f, beta = 0.0f;

    // need to call this at eval as 'N' changes :-/
    cudnnSetTensor4dDescriptor(outTensorDesc,
        fp16 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
        fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT,
        N, C, H, W);

    cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, outTensorDesc, input, &beta, outTensorDesc, output);
}

ConvLayer::ConvLayer(BaseLayer *ip, int C, int H, int W, int filter, int Cin, bool relu, bool bias) :
    BaseLayer(C, H, W, ip),
    filterSize(filter),
    Cinput(Cin),
    useRelu(relu),
    useBias(bias)
{
    // allocate memory for weights (filter tensor) and biases
    size_t weightSize = bpe * Cin * C * filterSize * filterSize;
    cudaMalloc(&weights, weightSize);

    size_t biasSize = bpe * C;
    cudaMalloc(&biases, biasSize);

    // create cudnn objects for various tensors, algorithms, etc
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnCreateTensorDescriptor(&outTensorDesc);
    cudnnCreateTensorDescriptor(&inTensorDesc);
    cudnnCreateTensorDescriptor(&biasDesc);
    cudnnCreateActivationDescriptor(&activation);

    cudnnSetFilter4dDescriptor(filterDesc,
        fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT,
        fp16 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,   // TODO: support fp16 evaluation
        C,
        Cin,
        filterSize,
        filterSize);

    reportCUDNNErrors(cudnnSetTensor4dDescriptor(biasDesc,
        fp16 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
        fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT,
        1, C, 1, 1));

    int padding = filterSize / 2;
    const bool crossCorr = 1;

    cudnnSetConvolution2dDescriptor(convDesc,
        padding, padding,
        1, 1,
        1, 1,
        crossCorr ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION,
        fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT);

    // TODO: dynamic selection of algorithm!
    if (C > 32)
    {
        convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
    }
    else
    {
        convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    }

    if (useRelu)
    {
        cudnnSetActivationDescriptor(activation, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);
    }
    else
    {
        cudnnSetActivationDescriptor(activation, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0);
    }
}

void ConvLayer::loadWeights(void *pfilter, void *pBias)
{
    size_t weightSize = bpe * Cinput * C * filterSize * filterSize;
    cudaMemcpyAsync(weights, pfilter, weightSize, cudaMemcpyHostToDevice);

    size_t biasSize = bpe * C;
    if (pBias)
    {
        cudaMemcpyAsync(biases, pBias, biasSize, cudaMemcpyHostToDevice);
    }
    else
    {
        cudaMemset(biases, biasSize, 0);
    }
}

void ConvLayer::eval(int N, void *output, void *input, void *input2, void *scratch, cudnnHandle_t cudnn, cublasHandle_t cublas)
{
    reportCUDNNErrors(cudnnSetTensor4dDescriptor(outTensorDesc,
        fp16 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
        fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT,
        N, C, H, W));

    reportCUDNNErrors(cudnnSetTensor4dDescriptor(inTensorDesc,
        fp16 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
        fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT,
        N, Cinput, H, W));


    float alpha = 1.0f, beta = 0.0f;


    if (!(useRelu || useBias))
    {
        reportCUDNNErrors(cudnnConvolutionForward(cudnn, &alpha, inTensorDesc,
            input, filterDesc, weights, convDesc,
            convAlgo, scratch, kCudaScratchSize, &beta,
            outTensorDesc, output));
    }
    else if (input2)
    {
        // fused bias + sum + relu!
        reportCUDNNErrors(cudnnConvolutionBiasActivationForward(cudnn, &alpha, inTensorDesc, input, filterDesc, weights, convDesc,
            convAlgo, scratch, kCudaScratchSize, &alpha, outTensorDesc, input2, biasDesc, biases,
            activation, outTensorDesc, output));
    }
    else
    {
        reportCUDNNErrors(cudnnConvolutionBiasActivationForward(cudnn, &alpha, inTensorDesc, input, filterDesc, weights, convDesc,
            convAlgo, scratch, kCudaScratchSize, &beta, outTensorDesc, output, biasDesc, biases,
            activation, outTensorDesc, output));
    }
}

ConvLayer::~ConvLayer()
{
    cudaFree(weights);
    cudaFree(biases);
}

BNLayer::BNLayer(BaseLayer *ip, bool relu) :
    BaseLayer(ip->getC(), ip->getH(), ip->getW(), ip),
    useRelu(relu)
{
    size_t weightSize = bpe * C;

    cudaMalloc(&means, weightSize);
    cudaMalloc(&variances, weightSize);
}

void BNLayer::loadWeights(void *cpuMeans, void *cpuVar)
{
    size_t weightSize = bpe * C;
    cudaMemcpyAsync(means, cpuMeans, weightSize, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(variances, cpuVar, weightSize, cudaMemcpyHostToDevice);
}

void BNLayer::eval(int N, void *output, void *input, void *input2, void *scratch, cudnnHandle_t cudnn, cublasHandle_t cublas)
{
    batchNormForward((float *)output, (float *)input, (float *)input2, N, C, H, W, (float *)means, (float *)variances, useRelu);
}

BNLayer::~BNLayer()
{
    cudaFree(means);
    cudaFree(variances);
}

FCLayer::FCLayer(BaseLayer *ip, int C, int H, int W, bool relu, bool bias, bool tanh) :
    BaseLayer(C, H, W, ip),
    useRelu(relu),
    useBias(bias),
    useTanh(tanh)
{
    size_t weightSize = bpe * C * H * W * ip->getC() * ip->getH() * ip->getW();
    size_t biasSize = bpe * C * H * W;
    cudaMalloc(&weights, weightSize);
    if (useBias)
    {
        cudaMalloc(&biases, biasSize);
    }
    else
    {
        biases = NULL;
    }
}

void FCLayer::loadWeights(void *cpuWeight, void *cpuBias)
{
    size_t weightSize = bpe * C * H * W * input->getC() * input->getH() * input->getW();
    cudaMemcpyAsync(weights, cpuWeight, weightSize, cudaMemcpyHostToDevice);
    if (useBias)
    {
        size_t biasSize = bpe * C * H * W;
        cudaMemcpyAsync(biases, cpuBias, biasSize, cudaMemcpyHostToDevice);
    }
}

void FCLayer::eval(int N, void *outputTensor, void *inputTensor, void *input2, void *scratch, cudnnHandle_t cudnn, cublasHandle_t cublas)
{
    float alpha = 1.0f, beta = 0.0f;
    int numOutputs = C * H * W;
    int numInputs = input->getC() * input->getH() * input->getW();

    if (fp16)
    {
        // TODO: implement this!
        assert(0);
    }
    else
    {
        cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            numOutputs, N, numInputs,
            &alpha,
            (float*)weights, numInputs,
            (float*)inputTensor, numInputs,
            &beta,
            (float*)outputTensor, numOutputs);

        if (useBias || useRelu || useTanh)
        {
            addVectors((float*)outputTensor, (float*)biases, (float*)outputTensor, numOutputs * N, numOutputs, numOutputs * N, useRelu, useTanh);
        }
    }

}

FCLayer::~FCLayer()
{
    cudaFree(weights);
    cudaFree(biases);
}


class CudnnNetwork;

class CudnnNetworkComputation : public NetworkComputation {
 public:
  CudnnNetworkComputation(const CudnnNetwork* network) : network_(network) {}

  void AddInput(InputPlanes&& input) override 
  {
      raw_input_.emplace_back(input);
  }

  void ComputeBlocking() override;

  int GetBatchSize() const override { return raw_input_.size(); }

  float GetQVal(int sample) const override 
  {
      return out_val_[sample];
  }
  float GetPVal(int sample, int move_id) const override 
  {
      return out_pol_[sample][move_id];
  }

 private:
     // input
     std::vector<InputPlanes> raw_input_;

     static constexpr int kNumOutputPolicy = 1858;

     // output (TODO: try using cudaHostAlloc to avoid the copy?)
     float out_pol_[kMaxBatchSize][kNumOutputPolicy];
     float out_val_[kMaxBatchSize];
     float input_planes_[kMaxBatchSize][kInputPlanes*8*8];

     const CudnnNetwork* network_;
};

class CudnnNetwork : public Network 
{
private:
    cudnnHandle_t cudnn_;
    cublasHandle_t cublas_;

    // currently only one NN eval can happen a time (we can fix this if needed by allocating more memory)
    std::mutex *lock_;

    int numBlocks_;
    std::vector<BaseLayer *> network_;
    BaseLayer *getLastLayer()
    {
        return network_[network_.size() - 1];
    }

    BaseLayer *resi_last_;
    BaseLayer *policy_out_;
    BaseLayer *value_out_;

    void *tensor_mem_[3];
    void *scratch_mem_;



    void processConvBlock(Weights::ConvBlock &block, bool foldBNLayer = false)
    {
        const float epsilon = 1e-5f;

        // compute reciprocal of std-dev from the variances (so that it can be just multiplied)
        std::vector<float> &stddev = block.bn_stddivs;
        for (auto&& w : stddev) {
            w = 1.0f / std::sqrt(w + epsilon);
        }

        // Biases are not calculated and are typically zero but some networks might
        // still have non-zero biases.
        // Move biases to batchnorm means to make the output match without having
        // to separately add the biases.
        for (auto j = size_t{ 0 }; j < block.bn_means.size(); j++) 
        {
            block.bn_means[j] -= block.biases[j];
            block.biases[j] = 0.0f;
        }

        // get rid of the BN layer by adjusting weights and biases of the convolution
        // idea proposed by Henrik Forstén and first implemented in leela go zero
        if (foldBNLayer)
        {
            const int outputs = block.biases.size();
            const int channels = block.weights.size() / (outputs * 3 * 3);

            for (auto o = 0; o < outputs; o++)
            {
                for (auto c = 0; c < channels; c++)
                {
                    for (auto i = 0; i < 9; i++)
                    {
                        block.weights[o*channels * 9 + c * 9 + i] *= block.bn_stddivs[o];
                    }
                }

                block.bn_means[o] *= block.bn_stddivs[o];
                block.bn_stddivs[o] = 1.0f;
               
                // Move means to convolution biases
                block.biases[o] = -block.bn_means[o];
                block.bn_means[o] = 0.0f;
            }
        }
    }
public:
    CudnnNetwork(Weights& weights)
    {
        // 0. initialize stuff
        lock_ = new std::mutex();

        // TODO: error checking!
        cudnnCreate(&cudnn_);
        cublasCreate(&cublas_);

        const int numInputPlanes = kInputPlanes;
        const int numFilters = weights.input.biases.size();
        assert(numFilters == 128);  // need to make sure nothing breaks after changing the no. of filters!

        numBlocks_ = weights.residual.size();

        // 0. process weights
        processConvBlock(weights.input, true);
        for (auto i = size_t{ 0 }; i < numBlocks_; i++)
        {
            processConvBlock(weights.residual[i].conv1, true);
            processConvBlock(weights.residual[i].conv2, true);

        }
        processConvBlock(weights.policy);
        processConvBlock(weights.value);


        // 1. build the network, and copy the weights to GPU memory
        // input 
        {
            ConvLayer *inputConv = new ConvLayer(NULL, numFilters, 8, 8, 3, numInputPlanes, true, true);
            inputConv->loadWeights(&weights.input.weights[0], &weights.input.biases[0]);
            network_.push_back(inputConv);
        }

        // residual block
        for (int block = 0; block < weights.residual.size(); block++)
        {
            ConvLayer *conv1 = new ConvLayer(getLastLayer(), numFilters, 8, 8, 3, numFilters, true, true);
            conv1->loadWeights(&weights.residual[block].conv1.weights[0], &weights.residual[block].conv1.biases[0]);
            network_.push_back(conv1);

            ConvLayer *conv2 = new ConvLayer(getLastLayer(), numFilters, 8, 8, 3, numFilters, true, true);
            conv2->loadWeights(&weights.residual[block].conv2.weights[0], &weights.residual[block].conv2.biases[0]);
            network_.push_back(conv2);
        }

        resi_last_ = getLastLayer();

        // policy head
        {
            ConvLayer *convPol = new ConvLayer(resi_last_, weights.policy.bn_means.size(), 8, 8, 1, numFilters);
            convPol->loadWeights(&weights.policy.weights[0]);
            network_.push_back(convPol);

            BNLayer *BNPol = new BNLayer(getLastLayer(), true);
            BNPol->loadWeights(&weights.policy.bn_means[0], &weights.policy.bn_stddivs[0]);
            network_.push_back(BNPol);

            FCLayer *FCPol = new FCLayer(getLastLayer(), weights.ip_pol_b.size(), 1, 1, false, true);
            FCPol->loadWeights(&weights.ip_pol_w[0], &weights.ip_pol_b[0]);
            network_.push_back(FCPol);

            SoftMaxLayer *softmaxPol = new SoftMaxLayer(getLastLayer());
            network_.push_back(softmaxPol);
        }
        policy_out_ = getLastLayer();

        // Value head
        {
            ConvLayer *convVal = new ConvLayer(resi_last_, weights.value.bn_means.size(), 8, 8, 1, numFilters);
            convVal->loadWeights(&weights.value.weights[0]);
            network_.push_back(convVal);

            BNLayer *BNVal = new BNLayer(getLastLayer(), true);
            BNVal->loadWeights(&weights.value.bn_means[0], &weights.value.bn_stddivs[0]);
            network_.push_back(BNVal);

            FCLayer *FCVal1 = new FCLayer(getLastLayer(), weights.ip1_val_b.size(), 1, 1, true, true);
            FCVal1->loadWeights(&weights.ip1_val_w[0], &weights.ip1_val_b[0]);
            network_.push_back(FCVal1);

            FCLayer *FCVal2 = new FCLayer(getLastLayer(), 1, 1, 1, false, true, true);
            FCVal2->loadWeights(&weights.ip2_val_w[0], &weights.ip2_val_b[0]);
            network_.push_back(FCVal2);
        }
        value_out_ = getLastLayer();

        // 2. allocate GPU memory for running the network
        //    - three buffers of max size are enough (one to hold input, second to hold output and third to hold skip connection's input)
        size_t maxSize = resi_last_->getOutputSize(kMaxBatchSize);
        for (int i = 0; i < 3; i++)
        {
            cudaMalloc(&tensor_mem_[i], maxSize);
            cudaMemset(tensor_mem_[i], 0, maxSize);
        }

        //printf("Allocated %d bytes of GPU memory to run the network\n", 3 * maxSize);

        // 3. allocate scratch space (used internally by cudnn to run convolutions)
        cudaMalloc(&scratch_mem_, kCudaScratchSize);

    }

    void forwardEval(float *input, float *op_pol, float *op_val, int batchSize) const
    {
        //printf(" ..%d.. ", batchSize);

        std::lock_guard<std::mutex> lock(*lock_);

        // copy data from CPU memory to GPU memory
        cudaMemcpyAsync(tensor_mem_[0], &input[0],
                        batchSize * kInputPlanes * network_[0]->getH() * network_[0]->getW() * sizeof(float),
                        cudaMemcpyHostToDevice);

        int l = 0;
        // input
        network_[l++]->eval(batchSize, tensor_mem_[2], tensor_mem_[0], NULL, scratch_mem_, cudnn_, cublas_);  // input conv

        // residual block
        for (int block = 0; block < numBlocks_; block++)
        {
            network_[l++]->eval(batchSize, tensor_mem_[0], tensor_mem_[2], NULL, scratch_mem_, cudnn_, cublas_);           // conv1
            network_[l++]->eval(batchSize, tensor_mem_[2], tensor_mem_[0], tensor_mem_[2], scratch_mem_, cudnn_, cublas_);   // conv2
        }

        // policy head
        network_[l++]->eval(batchSize, tensor_mem_[0], tensor_mem_[2], NULL, scratch_mem_, cudnn_, cublas_);    // pol conv
        network_[l++]->eval(batchSize, tensor_mem_[1], tensor_mem_[0], NULL, scratch_mem_, cudnn_, cublas_);    // pol BN
        network_[l++]->eval(batchSize, tensor_mem_[0], tensor_mem_[1], NULL, scratch_mem_, cudnn_, cublas_);    // pol FC       
        network_[l++]->eval(batchSize, tensor_mem_[1], tensor_mem_[0], NULL, scratch_mem_, cudnn_, cublas_);    // pol softmax  // POLICY

        // value head
        network_[l++]->eval(batchSize, tensor_mem_[0], tensor_mem_[2], NULL, scratch_mem_, cudnn_, cublas_);    // value conv
        network_[l++]->eval(batchSize, tensor_mem_[2], tensor_mem_[0], NULL, scratch_mem_, cudnn_, cublas_);    // value BN
        network_[l++]->eval(batchSize, tensor_mem_[0], tensor_mem_[2], NULL, scratch_mem_, cudnn_, cublas_);    // value FC1
        network_[l++]->eval(batchSize, tensor_mem_[2], tensor_mem_[0], NULL, scratch_mem_, cudnn_, cublas_);    // value FC2    // VALUE

        // copy results back to CPU memory
        cudaMemcpyAsync(&op_pol[0], tensor_mem_[1], policy_out_->getOutputSize(batchSize), cudaMemcpyDeviceToHost);
        cudaError_t status = cudaMemcpy(&op_val[0], tensor_mem_[2], value_out_->getOutputSize(batchSize), cudaMemcpyDeviceToHost);

        if (status != cudaSuccess)
        {
            throw(new Exception("Some error running cuda based eval!"));
        }

    }

    ~CudnnNetwork()
    {
        delete lock_;
        cudnnDestroy(cudnn_);
        cublasDestroy(cublas_);

        for (int i = 0; i < 3; i++)
        {
            if(tensor_mem_[i])
                cudaFree(tensor_mem_[i]);
        }

        if (scratch_mem_)
            cudaFree(scratch_mem_);
    }

    std::unique_ptr<NetworkComputation> NewComputation() override 
    {
        return std::make_unique<CudnnNetworkComputation>(this);
    }
};

std::unique_ptr<Network> MakeCudnnNetwork(Weights& weights) {
  return std::make_unique<CudnnNetwork>(weights);
}

void CudnnNetworkComputation::ComputeBlocking()
{
    // Convert raw_input to "expanded planes" - format the first convolutional layer expects
    // TODO: can probably do this on the GPU if this becomes a bottleneck
    float *data = &(input_planes_[0][0]);
    memset(data, 0, sizeof(float) * GetBatchSize()  * kInputPlanes * 8 * 8);
    auto iter = data;
    for (const auto& sample : raw_input_)
    {
        //CHECK_EQ(sample.size(), kInputPlanes);
        for (const auto& plane : sample)
        {
            for (auto bit : IterateBits(plane.mask))
            {
                *(iter + bit) = plane.value;
            }
            iter += 64;
        }
    }

    network_->forwardEval(data, &(out_pol_[0][0]), &(out_val_[0]), GetBatchSize());
    return;
}

}  // namespace lczero