// Shader for the Fully connected layer 2 of the value head

#define InW 128

cbuffer consts : register(b0) {
  uint N;             // Batch size.
};

RWByteAddressBuffer output : register(u0);
RWByteAddressBuffer input : register(u1);
RWByteAddressBuffer weight : register(u2);
RWByteAddressBuffer bias : register(u3);

// N threads are launched, each thread writes one output

float2 extractElements(uint packedVal) {
  return float2(f16tof32(packedVal & 0xFFFF),
                f16tof32((packedVal >> 16) & 0xFFFF));
}

[numthreads(128, 1, 1)] 
void ValueFC2(uint3 threadIdx : SV_DispatchThreadID) {
  uint y = threadIdx.x;

  if (y >= N) return;

  float S = 0;

  [unroll] 
  for (int i = 0; i < InW; i += 2) {
    uint aIndex = y * InW + i;
    uint aVal = input.Load(aIndex * 2);  // Byte offset.
    float2 a = extractElements(aVal);

    // Note: Weight matrix is transposed.
    uint bIndex1 = i;
    uint bVal1 = weight.Load(bIndex1 * 2);
    float2 b1 = extractElements(bVal1);

    S += a.x * b1.x + a.y * b1.y;
  }

  // Add bias (bias is in fp32).
  float b = asfloat(bias.Load(0));
  S += b;

  // Tanh activation function.
  S = tanh(S);


  // Write output (fp32).
  output.Store(y * 4, asuint(S));
}
