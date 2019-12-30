#include "shader_shared.h"

// ------------------- Simple 1x1 convolution shader -----------------------------//

#if FP16_IO == 1
RWStructuredBuffer<float16_t>   output_fp32 : register(u0);
RWStructuredBuffer<float16_t>   input_fp32  : register(u1);
RWStructuredBuffer<float16_t>   filter_fp32 : register(u2);
RWStructuredBuffer<float16_t>   bias_fp32   : register(u3);
#else
RWStructuredBuffer<float>       output_fp32 : register(u0);
RWStructuredBuffer<float>       input_fp32  : register(u1);
RWStructuredBuffer<float>       filter_fp32 : register(u2);
RWStructuredBuffer<float>       bias_fp32   : register(u3);
#endif

cbuffer ConvConsts : register(b0) {
  uint N, K, C;       
  uint useBias;
  uint relu;
};


#define MAX_FILTERS 1024
groupshared float sh_filter[MAX_FILTERS];
groupshared float sh_bias;


// N*K thread blocks launched (groupIdx.y, and groupIdx.x resp.)
// Each block has 64 (8x8) thread.
// Each thread writes single output element.
[numthreads(kConv1x1BlockSize, 1, 1)] 
#if FP16_IO == 1
void conv_1x1_shader_fp16
#else
void conv_1x1_shader_fp32
#endif
(
    uint3 gtid  : SV_DispatchThreadID, 
    uint3 tid   : SV_GroupThreadID,
    uint3 gid   : SV_GroupID
) 
{
  int k = gid.x;
  int n = gid.y;

  // load bias into shared memory
  if (tid.x == 0) 
      sh_bias = useBias ? bias_fp32[k] : 0;

  // load filter into shared memory
  const int iterations = (C - 1) / kConv1x1BlockSize + 1;
  for (int i = 0; i < iterations; i++)
  {
    int c = i * kConv1x1BlockSize + tid.x;
    if (c < C) 
        sh_filter[c] = filter_fp32[k * C + c];
  }

  GroupMemoryBarrierWithGroupSync();

  float op = sh_bias;
  for (int c = 0; c < C; c++)
  {
    float ip = input_fp32[n * C * 64 + c * 64 + tid.x];
    float filter = sh_filter[c];
    op += ip * filter;
  }

  if (relu && op < 0) op = 0;

  output_fp32[n * K * 64 + k * 64 + tid.x] = op;
}

