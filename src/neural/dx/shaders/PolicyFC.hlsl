// see for HLSL tricks:
// https://github.com/Microsoft/DirectXShaderCompiler/wiki/16-Bit-Scalar-Types

// Shader for the Fully connected layer for policy out.
// Also includes softmax.
// Output is fp32.
// Input is fp16.
// Weights are in fp16, biases in fp32.
// The shader itself is pretty naive, we should switch
// to a Metacommand when its supported by HW vendors.

// Each thread block writes one complete policy vector (1858 elements)
// 'N' blocks are launched, where N is the batch size
// each thread processes two consecutive elements of output
#define OutW 1858

#define NumThreads (OutW / 2)

// Size of output from last convolutional layer.
#define InW (8 * 8 * 32)

// Shared memory used to compute softmax for each policy vector.
// Ceil(1858 / 2 / 32).
#define NumSharedMemEntries ((NumThreads - 1) / 32 + 1)
groupshared float shDnorm[NumSharedMemEntries];

RWByteAddressBuffer output : register(u0);
RWByteAddressBuffer input : register(u1);
RWByteAddressBuffer weight : register(u2);
RWByteAddressBuffer bias : register(u3);

float2 extractElements(uint packedVal) {
  return float2(f16tof32(packedVal & 0xFFFF),
                f16tof32((packedVal >> 16) & 0xFFFF));
}

[numthreads(NumThreads, 1, 1)] void PolicyFC_With_Softmax_kernel(
    uint3 threadIdx
    : SV_GroupThreadID, uint3 blockIdx
    : SV_GroupID) {
  uint x = threadIdx.x * 2;  // First of the two consecutive output elements.
  uint y = blockIdx.x;

  float2 S = float2(0, 0);

  //[unroll] 
  for (int i = 0; i < InW; i += 2) {
    uint aIndex = y * InW + i;
    uint aVal = input.Load(aIndex * 2);  // Byte offset.
    float2 a = extractElements(aVal);

    uint bIndex1 = i * OutW + x;
    uint bVal1 = weight.Load(bIndex1 * 2);
    float2 b1 = extractElements(bVal1);

    uint bIndex2 = (i+1)*OutW + x;
    uint bVal2 = weight.Load(bIndex2 * 2);
    float2 b2 = extractElements(bVal2);

    S.x += a.x * b1.x + a.y * b2.x;
    S.y += a.x * b1.y + a.y * b2.y;
  }


  // Add bias (bias is in fp32).
  float2 b;
  b.x = asfloat(bias.Load(x * 4));
  b.y = asfloat(bias.Load((x + 1) * 4));
  S += b;

  // S now contains result after fully connected layer.
  // Apply Softmax.

  // TODO: Maybe get a max first, and substract it from each element
  //       to avoid overflows?

  S = exp(S);

  // Compute denorm (i.e, sum of all elements in block).
  float dnorm = S.x + S.y;

  // First get sum of all elements in warp / wave.
  dnorm = WaveActiveSum(dnorm);

  // Next use shared memory to get total sum.
  uint warpInBlock = threadIdx.x >> 5;
  if (WaveIsFirstLane()) {
    // First thread of warp writes to shared memory.
    shDnorm[warpInBlock] = dnorm;
  }
  GroupMemoryBarrierWithGroupSync();
  if (warpInBlock == 0 && WaveGetLaneIndex() < NumSharedMemEntries) {
    float val = shDnorm[WaveGetLaneIndex()];
    dnorm = WaveActiveSum(val);

    if (WaveIsFirstLane()) shDnorm[0] = dnorm;
  }

  GroupMemoryBarrierWithGroupSync();

  dnorm = shDnorm[0];

  // Dnorm now contains the sum.
  S.x /= dnorm;
  S.y /= dnorm;

  // Write output (note fp32).
  uint cIndex = y * OutW + x;
  output.Store(cIndex * 4, asuint(S.x));
  output.Store((cIndex + 1) * 4, asuint(S.y));
}

cbuffer consts : register(b0) {
  uint N;  // Batch size.
};

#define blockWidth 16
#define blockHeight 2

#define elementsPerThreadX 4
#define elementsPerThreadY 4

#define elementsPerBlockX (blockWidth * elementsPerThreadX)
#define elementsPerBlockY (blockHeight * elementsPerThreadY)

// bias add done by softmax layer
[numthreads(blockWidth, blockHeight, 1)] 
void PolicyFC(uint3 threadIdx : SV_DispatchThreadID) {
  uint xBase = threadIdx.x * elementsPerThreadX;
  uint yBase = threadIdx.y * elementsPerThreadY;

  float S[elementsPerThreadY][elementsPerThreadX];

  uint i, j, k;

  for (j = 0; j < elementsPerThreadY; j++)
    for (i = 0; i < elementsPerThreadX; i++) S[j][i] = 1.25f;

  for (k = 0; k < InW; k+=2) {
    for (j = 0; j < elementsPerThreadY; j++) {
      for (i = 0; i < elementsPerThreadX; i+=2) {
        uint x = xBase + i;
        uint y = yBase + j;
        if (x < OutW && y < N) {

           uint aIndex = y * InW + k;
           uint aVal = input.Load(aIndex * 2);  // Byte offset.
           float2 a = extractElements(aVal);

           uint bIndex1 = k * OutW + x;
           uint bVal1 = weight.Load(bIndex1 * 2);
           float2 b1 = extractElements(bVal1);

           uint bIndex2 = (k + 1) * OutW + x;
           uint bVal2 = weight.Load(bIndex2 * 2);
           float2 b2 = extractElements(bVal2);

           S[j][i] += a.x * b1.x + a.y * b2.x;
           S[j][i+1] += a.x * b1.y + a.y * b2.y;
         }
      }
    }
  }

  for (j = 0; j < elementsPerThreadY; j++)
    for (i = 0; i < elementsPerThreadX; i += 2) {
      uint val = f32tof16(S[j][i]) | (f32tof16(S[j][i+1]) << 16);
      uint x = xBase + i;
      uint y = yBase + j;
      if (x < OutW && y < N) {
        uint index = y * OutW + x;
        output.Store(index * 2, val);
      }
    }
}


[numthreads(NumThreads, 1, 1)] 
void PolicySoftmax(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID) {
  uint x = threadIdx.x * 2;  // First of the two consecutive output elements.
  uint y = blockIdx.x;
  
  uint index = y * OutW + x;

  // Load result of policy FC.
  uint valu = output.Load(index * 2);
  float2 S = extractElements(valu);

  // Add bias.
  float2 b;
  b.x = bias.Load<float>(x);
  b.y = bias.Load<float>(x+1);
  S += b;

  // Apply Softmax.

  // TODO: Maybe get a max first, and substract it from each element
  //       to avoid overflows?

  S = exp(S);

  // Compute denorm (i.e, sum of all elements in block).
  float dnorm = S.x + S.y;

  // First get sum of all elements in warp / wave.
  dnorm = WaveActiveSum(dnorm);

  // Next use shared memory to get total sum.
  uint warpInBlock = threadIdx.x >> 5;
  if (WaveIsFirstLane()) {
    // First thread of warp writes to shared memory.
    shDnorm[warpInBlock] = dnorm;
  }
  GroupMemoryBarrierWithGroupSync();
  if (warpInBlock == 0 && WaveGetLaneIndex() < NumSharedMemEntries) {
    float val = shDnorm[WaveGetLaneIndex()];
    dnorm = WaveActiveSum(val);

    if (WaveIsFirstLane()) shDnorm[0] = dnorm;
  }

  GroupMemoryBarrierWithGroupSync();

  dnorm = shDnorm[0];

  // Dnorm now contains the sum.
  S.x /= dnorm;
  S.y /= dnorm;

  // Write output (note fp32).
  output.Store(index * 4, asuint(S.x));
  output.Store((index + 1) * 4, asuint(S.y));
}