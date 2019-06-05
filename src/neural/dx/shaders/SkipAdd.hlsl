// Shader for the Skip connection add + Relu operation

cbuffer consts : register(b0) {
  uint numEl;
  uint relu;
};

RWByteAddressBuffer output : register(u0);
RWByteAddressBuffer input1 : register(u1);
RWByteAddressBuffer input2 : register(u2);

float2 extractElements(uint packedVal) {
  return float2(f16tof32(packedVal & 0xFFFF),
                f16tof32((packedVal >> 16) & 0xFFFF));
}

// Each thread processes 2 elements
[numthreads(512, 1, 1)] void SkipAdd(uint3 threadIdx
                                      : SV_DispatchThreadID) {
  uint index = threadIdx.x * 4;

  if (threadIdx.x * 2 >= numEl) return;

  float2 ip1 = extractElements(input1.Load(index));
  float2 ip2 = extractElements(input2.Load(index));
  float2 op = ip1 + ip2;

  if (relu) {
    if (op.x < 0)
      op.x = 0;
    if (op.y < 0)
      op.y = 0;
  }

  uint2 opu = f32tof16(op);
  uint opVal = opu.x | (opu.y << 16);
  output.Store(index, opVal);
}
