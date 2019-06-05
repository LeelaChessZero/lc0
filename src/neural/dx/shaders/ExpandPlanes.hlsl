// ------------------- Expand Planes Shader -----------------------------//

RWByteAddressBuffer output : register(u0);
RWByteAddressBuffer masks : register(u1);
RWByteAddressBuffer values : register(u2);

cbuffer ExpandPlanesConsts : register(b0) {
  uint N;             // total no of planes to process
  uint kInputPlanes;  // no of planes per position
};

// TODO: Can optimize using shared memory if this becomes a bottleneck.
// 256 threads per block
// each thrad writes 2 output elements
[numthreads(256, 1, 1)] void ExpandPlanes_kernel_Fp16_NHWC(
    uint3 threadID
    : SV_DispatchThreadID) {
  const int index = threadID.x * 2;
  if (index >= N * 8 * 8) return;

  const int planeIndex0 = index % kInputPlanes;
  const int planeIndex1 = planeIndex0 + 1;

  const int boardIndex = index / (kInputPlanes * 8 * 8);
  const int sqIndex = (index / kInputPlanes) & 0x3F;

  uint inpIndex0 = boardIndex * kInputPlanes + planeIndex0;
  uint inpIndex1 = boardIndex * kInputPlanes + planeIndex1;

  uint64_t mask0 = 0, mask1 = 0;

  mask0 = masks.Load(inpIndex0 * 8) |
          (((uint64_t)masks.Load(inpIndex0 * 8 + 4)) << 32);

  mask1 = masks.Load(inpIndex1 * 8) |
          (((uint64_t)masks.Load(inpIndex1 * 8 + 4)) << 32);

  float2 opf;
  bool set = !!(mask0 & (1ull << sqIndex));
  if (set) {
    opf.x = asfloat(values.Load(inpIndex0 * 4));  // byte offset
  }

  set = !!(mask1 & (1ull << sqIndex));
  if (set) {
    opf.y = asfloat(values.Load(inpIndex1 * 4));  // byte offset
  }

  uint2 opu = f32tof16(opf);
  uint opVal = opu.x | (opu.y << 16);
  output.Store(index * 2, opVal);  // byte offset
}
