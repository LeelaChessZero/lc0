// Shaders for the Fully connected layers for value head.
// For FC1, input and output are fp16, weight is always in fp16 
// and bias in fp32.

// For FC1
// Each thread block writes one complete vector (128 elements)
// 'N' blocks are launched, where N is the batch size
// each thread processes two consecutive elements of output
#define OutW 128


#define NumThreads (OutW / 2)

// Size of output from last convolutional layer.
#define InW (8 * 8 * 32)

RWByteAddressBuffer output : register(u0);
RWByteAddressBuffer input : register(u1);
RWByteAddressBuffer weight : register(u2);
RWByteAddressBuffer bias : register(u3);

float2 extractElements(uint packedVal) {
  return float2(f16tof32(packedVal & 0xFFFF),
                f16tof32((packedVal >> 16) & 0xFFFF));
}

[numthreads(NumThreads, 1, 1)] 
void ValueFC1(
    uint3 threadIdx : SV_GroupThreadID, 
    uint3 blockIdx  : SV_GroupID) {
  uint x = threadIdx.x * 2;  // First of the two consecutive output elements.
  uint y = blockIdx.x;

  float2 S = float2(0, 0);

  //[unroll] 
  for (int i = 0; i < InW; i += 2) {
    uint aIndex = y * InW + i;
    uint aVal = input.Load(aIndex * 2);  // Byte offset.
    float2 a = extractElements(aVal);

    // Note: Weight matrix is transposed.
    //uint bIndex1 = x * InW + i;   // not any more!
    uint bIndex1 = i * OutW + x;
    uint bVal1 = weight.Load(bIndex1 * 2);
    float2 b1 = extractElements(bVal1);

    // uint bIndex2 = (x + 1) * InW + i;
    uint bIndex2 = (i + 1) * OutW + x;
    uint bVal2 = weight.Load(bIndex2 * 2);
    float2 b2 = extractElements(bVal2);

    //S.x += a.x * b1.x + a.y * b1.y;
    //S.y += a.x * b2.x + a.y * b2.y;
    S.x += a.x * b1.x + a.y * b2.x;
    S.y += a.x * b1.y + a.y * b2.y;

  }

  // Add bias (bias is in fp32).
  float2 b;
  b.x = asfloat(bias.Load(x * 4));
  b.y = asfloat(bias.Load((x + 1) * 4));
  S += b;

  // Apply Relu activation.
  if (S.x < 0) S.x = 0;
  if (S.y < 0) S.y = 0;

  uint outVal = f32tof16(S.x) | (f32tof16(S.y) << 16);

  // Write output (fp16x2).
  uint cIndex = y * OutW + x;
  output.Store(cIndex * 2, outVal);
}
