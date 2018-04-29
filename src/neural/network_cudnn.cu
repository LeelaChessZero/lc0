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
#include <functional>
#include <cassert>
#include <mutex>

#include <cublas_v2.h>
#include <cudnn.h>


void cudnnError(cudnnStatus_t status, const char* file, const int& line)
{
    if (status != CUDNN_STATUS_SUCCESS)
    {
        printf("CUDNN error: %s (%s:%d)\n", cudnnGetErrorString(status), file, line);
        getchar();
        exit(1);
    }
}

#define reportCUDNNErrors(status)  cudnnError(status, __FILE__, __LINE__)


// 256 MB fixed scratch memory size (hardcoded for now)
#define CUDA_SCRATCH_SIZE (256*1024*1024)

// hard-coded for now, no point in going above this anyway (can possibly save memory by reducing this)
static constexpr int MAX_BATCH_SIZE = 1024;

// the Layer objects only hold memory for weights, biases, etc
// memory for input and output tensors is provided by caller of forwardEval

class BaseLayer
{
protected:
    static bool fp16;
    static size_t bpe;  // size of each element
    static int N;       // batch size is fixed for the network
    int C, H, W;        // output tensor dimensions
    BaseLayer *input;

public:
    int getC() { return C; }
    int getH() { return H; }
    int getW() { return W; }
    static int getN() { return N; }

    static void setBatchSize(int n) { N = n; };

    BaseLayer(int c, int h, int w, BaseLayer *ip);

    // input2 is optional (skip connection)
    size_t getOutputSize() { return bpe * N * C * H * W; }
    virtual void eval(void *output, void *input, void *input2, void *scratch) = 0;
};

class ConvLayer : public BaseLayer
{
private:
    int Cinput;
    int filterSize;
    void *weights;

    cudnnFilterDescriptor_t			filterDesc;
    cudnnConvolutionDescriptor_t	convDesc;
    cudnnConvolutionFwdAlgo_t       convAlgo;

    cudnnTensorDescriptor_t         inTensorDesc;
    cudnnTensorDescriptor_t         outTensorDesc;


public:
    ConvLayer(BaseLayer *ip, int C, int H, int W, int size, int Cin);
    ~ConvLayer();
    void loadWeights(void *pfilter);
    void eval(void *output, void *input, void *input2, void *scratch) override;
};

class SoftMaxLayer : public BaseLayer
{
private:
    cudnnTensorDescriptor_t         outTensorDesc;

public:
    SoftMaxLayer(BaseLayer *ip);
    void eval(void *output, void *input, void *input2, void *scratch) override;
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
    void eval(void *output, void *input, void *input2, void *scratch) override;

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
    void eval(void *output, void *input, void *input2, void *scratch) override;
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
//                      Global/Static variables                            //
/////////////////////////////////////////////////////////////////////////////

int BaseLayer::N = 1;
bool BaseLayer::fp16 = false;
size_t BaseLayer::bpe = sizeof(float);

cudnnHandle_t g_cudnn;
cublasHandle_t g_cublas;


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

void SoftMaxLayer::eval(void *output, void *input, void *input2, void *scratch)
{
    float alpha = 1.0f, beta = 0.0f;

    // need to call this at eval as 'N' changes :-/
    cudnnSetTensor4dDescriptor(outTensorDesc,
        fp16 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
        fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT,
        N, C, H, W);

    cudnnSoftmaxForward(g_cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, outTensorDesc, input, &beta, outTensorDesc, output);
}

ConvLayer::ConvLayer(BaseLayer *ip, int C, int H, int W, int filter, int Cin) :
    BaseLayer(C, H, W, ip),
    filterSize(filter),
    Cinput(Cin)
{
    // allocate memory for weights (filter tensor)
    size_t weightSize = bpe * Cin * C * filterSize * filterSize;
    cudaMalloc(&weights, weightSize);

    // create cudnn objects for various tensors, algorithms, etc
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnCreateTensorDescriptor(&outTensorDesc);
    cudnnCreateTensorDescriptor(&inTensorDesc);

    cudnnSetFilter4dDescriptor(filterDesc,
        fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT,
        fp16 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,   // TODO: support fp16 evaluation
        C,
        Cin,
        filterSize,
        filterSize);

    int padding = filterSize / 2;
    const bool crossCorr = 1;           // TODO: find out if it's right?

    cudnnSetConvolution2dDescriptor(convDesc,
        padding, padding,
        1, 1,
        1, 1,
        crossCorr ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION,
        fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT);

    /*
    cudnnGetConvolutionForwardAlgorithm(g_cudnn,
    inTensorDesc,
    filterDesc,
    convDesc,
    outTensorDesc,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    0,
    &convAlgo);

    */

    // TODO: dynamic selection of algorithm!
    if (C > 32)
    {
        convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
    }
    else
    {
        convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    }

}

void ConvLayer::loadWeights(void *pfilter)
{
    size_t weightSize = bpe * Cinput * C * filterSize * filterSize;
    cudaMemcpyAsync(weights, pfilter, weightSize, cudaMemcpyHostToDevice);
}

void ConvLayer::eval(void *output, void *input, void *input2, void *scratch)
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

    reportCUDNNErrors(cudnnConvolutionForward(g_cudnn, &alpha, inTensorDesc,
        input, filterDesc, weights, convDesc,
        convAlgo, scratch, CUDA_SCRATCH_SIZE, &beta,
        outTensorDesc, output));
}

ConvLayer::~ConvLayer()
{
    cudaFree(weights);
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

void BNLayer::eval(void *output, void *input, void *input2, void *scratch)
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

void FCLayer::eval(void *outputTensor, void *inputTensor, void *input2, void *scratch)
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
        cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
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


namespace lczero {

class CudnnNetwork;

class CudnnNetworkComputation : public NetworkComputation {
 public:
  CudnnNetworkComputation(const CudnnNetwork* network) : m_network(network) {}

  void AddInput(InputPlanes&& input) override 
  {
      raw_input.emplace_back(input);
  }

  void ComputeBlocking() override;

  int GetBatchSize() const override { return raw_input.size(); }

  float GetQVal(int sample) const override 
  {
      return out_val[sample];
  }
  float GetPVal(int sample, int move_id) const override 
  {
      return out_pol[sample][move_id];
  }

 private:
     // input
     std::vector<InputPlanes> raw_input;

     static constexpr int V2_NUM_OUTPUT_POLICY = 1858;

     // output (TODO: try using cudaHostAlloc to avoid the copy?)
     float out_pol[MAX_BATCH_SIZE][V2_NUM_OUTPUT_POLICY];
     float out_val[MAX_BATCH_SIZE];
     float input_planes[MAX_BATCH_SIZE][kInputPlanes*8*8];

     const CudnnNetwork* m_network;
};

// currently only one NN eval can happen a time (we can fix this if needed by allocating more memory)
static std::mutex g_lock;

class CudnnNetwork : public Network 
{
private:
    int numBlocks;
    std::vector<BaseLayer *> network;
    BaseLayer *getLastLayer()
    {
        return network[network.size() - 1];
    }

    BaseLayer *resiLast;
    BaseLayer *policyOut;
    BaseLayer *valueOut;

    void *tensorMem[3];
    void *scratchMem;

    void fixStdDevs(std::vector<float> &arr, const float epsilon = 1e-5f)
    {
        for (auto&& w : arr) {
            w = 1.0f / std::sqrt(w + epsilon);
        }
    }

    void fixBiases(Weights::ConvBlock &block)
    {
        for (auto j = size_t{ 0 }; j < block.bn_means.size(); j++) 
        {
            block.bn_means[j] -= block.biases[j];
            block.biases[j] = 0.0f;
        }
    }
public:
    CudnnNetwork(Weights& weights)
    {
        // 0. initialize stuff
        // TODO: error checking!
        cudnnCreate(&g_cudnn);
        cublasCreate(&g_cublas);

        const int numInputPlanes = kInputPlanes;
        const int numFilters = weights.input.biases.size();
        assert(numFilters == 128);  // need to make sure nothing breaks after changing the no. of filters!

        numBlocks = weights.residual.size();

        // 0. process weights
        
        // Biases are not calculated and are typically zero but some networks might
        // still have non-zero biases.
        // Move biases to batchnorm means to make the output match without having
        // to separately add the biases.

        // Also compute reciprocal of std-dev from the variances (so that it can be just multiplied)

        fixBiases(weights.input);
        fixStdDevs(weights.input.bn_stddivs);
        for (auto i = size_t{ 0 }; i < numBlocks; i++)
        {
            fixBiases(weights.residual[i].conv1);
            fixBiases(weights.residual[i].conv2);

            fixStdDevs(weights.residual[i].conv1.bn_stddivs);
            fixStdDevs(weights.residual[i].conv2.bn_stddivs);
        }

        fixBiases(weights.policy);
        fixBiases(weights.value);
        fixStdDevs(weights.policy.bn_stddivs);
        fixStdDevs(weights.value.bn_stddivs);



        // 1. build the network, and copy the weights to GPU memory
        // input 
        {
            ConvLayer *inputConv = new ConvLayer(NULL, numFilters, 8, 8, 3, numInputPlanes);
            inputConv->loadWeights(&weights.input.weights[0]);
            network.push_back(inputConv);

            BNLayer *inputBN = new BNLayer(getLastLayer(), true);
            inputBN->loadWeights(&weights.input.bn_means[0], &weights.input.bn_stddivs[0]);
            network.push_back(inputBN);
        }

        // residual block
        for (int block = 0; block < weights.residual.size(); block++)
        {
            ConvLayer *conv1 = new ConvLayer(getLastLayer(), numFilters, 8, 8, 3, numFilters);
            conv1->loadWeights(&weights.residual[block].conv1.weights[0]);
            network.push_back(conv1);

            BNLayer *BN1 = new BNLayer(getLastLayer(), true);
            BN1->loadWeights(&weights.residual[block].conv1.bn_means[0], &weights.residual[block].conv1.bn_stddivs[0]);
            network.push_back(BN1);

            ConvLayer *conv2 = new ConvLayer(getLastLayer(), numFilters, 8, 8, 3, numFilters);
            conv2->loadWeights(&weights.residual[block].conv2.weights[0]);
            network.push_back(conv2);

            BNLayer *BN2 = new BNLayer(getLastLayer(), true);
            BN2->loadWeights(&weights.residual[block].conv2.bn_means[0], &weights.residual[block].conv2.bn_stddivs[0]);
            network.push_back(BN2);
        }

        resiLast = getLastLayer();

        // policy head
        {
            ConvLayer *convPol = new ConvLayer(resiLast, weights.policy.bn_means.size(), 8, 8, 1, numFilters);
            convPol->loadWeights(&weights.policy.weights[0]);
            network.push_back(convPol);

            BNLayer *BNPol = new BNLayer(getLastLayer(), true);
            BNPol->loadWeights(&weights.policy.bn_means[0], &weights.policy.bn_stddivs[0]);
            network.push_back(BNPol);

            FCLayer *FCPol = new FCLayer(getLastLayer(), weights.ip_pol_b.size(), 1, 1, false, true);
            FCPol->loadWeights(&weights.ip_pol_w[0], &weights.ip_pol_b[0]);
            network.push_back(FCPol);

            SoftMaxLayer *softmaxPol = new SoftMaxLayer(getLastLayer());
            network.push_back(softmaxPol);
        }
        policyOut = getLastLayer();

        // Value head
        {
            ConvLayer *convVal = new ConvLayer(resiLast, weights.value.bn_means.size(), 8, 8, 1, numFilters);
            convVal->loadWeights(&weights.value.weights[0]);
            network.push_back(convVal);

            BNLayer *BNVal = new BNLayer(getLastLayer(), true);
            BNVal->loadWeights(&weights.value.bn_means[0], &weights.value.bn_stddivs[0]);
            network.push_back(BNVal);

            FCLayer *FCVal1 = new FCLayer(getLastLayer(), weights.ip1_val_b.size(), 1, 1, true, true);
            FCVal1->loadWeights(&weights.ip1_val_w[0], &weights.ip1_val_b[0]);
            network.push_back(FCVal1);

            FCLayer *FCVal2 = new FCLayer(getLastLayer(), 1, 1, 1, false, true, true);
            FCVal2->loadWeights(&weights.ip2_val_w[0], &weights.ip2_val_b[0]);
            network.push_back(FCVal2);
        }
        valueOut = getLastLayer();

        // 2. allocate GPU memory for running the network
        //    - three buffers of max size are enough (one to hold input, second to hold output and third to hold skip connection's input)
        resiLast->setBatchSize(MAX_BATCH_SIZE);
        size_t maxSize = resiLast->getOutputSize();
        for (int i = 0; i < 3; i++)
        {
            cudaMalloc(&tensorMem[i], maxSize);
            cudaMemset(tensorMem[i], 0, maxSize);
        }

        //printf("Allocated %d bytes of GPU memory to run the network\n", 3 * maxSize);

        // 3. allocate scratch space (used internally by cudnn to run convolutions)
        cudaMalloc(&scratchMem, CUDA_SCRATCH_SIZE);

    }

    void forwardEval(float *input, float *op_pol, float *op_val, int batchSize) const
    {
        // printf(" ..%d.. ", batchSize);

        g_lock.lock();

        BaseLayer::setBatchSize(batchSize);

        // copy data from CPU memory to GPU memory
        cudaMemcpyAsync(tensorMem[0], &input[0], 
                        BaseLayer::getN() * kInputPlanes * network[0]->getH() * network[0]->getW() * sizeof(float), 
                        cudaMemcpyHostToDevice);

        int l = 0;
        // input
        network[l++]->eval(tensorMem[1], tensorMem[0], NULL, scratchMem);  // input conv
        network[l++]->eval(tensorMem[2], tensorMem[1], NULL, scratchMem);  // input BN

        // residual block
        for (int block = 0; block < numBlocks; block++)
        {
            network[l++]->eval(tensorMem[0], tensorMem[2], NULL, scratchMem);  // conv1
            network[l++]->eval(tensorMem[1], tensorMem[0], NULL, scratchMem);  // bn1

            network[l++]->eval(tensorMem[0], tensorMem[1], NULL, scratchMem);          // conv2
            network[l++]->eval(tensorMem[2], tensorMem[0], tensorMem[2], scratchMem);  // bn2 (with skip connection)
        }

        // policy head
        network[l++]->eval(tensorMem[0], tensorMem[2], NULL, scratchMem);    // pol conv
        network[l++]->eval(tensorMem[1], tensorMem[0], NULL, scratchMem);    // pol BN
        network[l++]->eval(tensorMem[0], tensorMem[1], NULL, scratchMem);    // pol FC       
        network[l++]->eval(tensorMem[1], tensorMem[0], NULL, scratchMem);    // pol softmax  // POLICY

        // value head
        network[l++]->eval(tensorMem[0], tensorMem[2], NULL, scratchMem);    // value conv
        network[l++]->eval(tensorMem[2], tensorMem[0], NULL, scratchMem);    // value BN
        network[l++]->eval(tensorMem[0], tensorMem[2], NULL, scratchMem);    // value FC1
        network[l++]->eval(tensorMem[2], tensorMem[0], NULL, scratchMem);    // value FC2    // VALUE

        // copy results back to CPU memory
        cudaMemcpyAsync(&op_pol[0], tensorMem[1], policyOut->getOutputSize(), cudaMemcpyDeviceToHost);
        cudaError_t status = cudaMemcpy(&op_val[0], tensorMem[2], valueOut->getOutputSize(), cudaMemcpyDeviceToHost);

        if (status != cudaSuccess)
        {
            printf("\nSome error running cuda based eval!\n");
            exit(1);
        }

        g_lock.unlock();

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
    // 1. convert raw_input to "expanded planes" - format the first convolutional layer expects
    float *data = &(input_planes[0][0]);
    memset(data, 0, GetBatchSize()  * kInputPlanes * 8 * 8);
    auto iter = data;
    for (const auto& sample : raw_input)
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

    m_network->forwardEval(data, &(out_pol[0][0]), &(out_val[0]), GetBatchSize());
    return;
}

}  // namespace lczero