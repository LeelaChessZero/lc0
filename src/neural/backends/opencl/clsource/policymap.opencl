// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.

R"(
__kernel void policymap(
              __global const net_t * restrict input,
              __global net_t * restrict output,
              __global short* restrict indices,
              const int N,
              const int inputSize,
              const int usedSize,
              const int outputSize) {

  int tid = get_global_id(0);

  int n = tid / usedSize;
  int i = tid % usedSize;

  if (n >= N) return;

  int j = indices[i];

  if (j >= 0) {
    output[n * outputSize + j] = input[n * inputSize + i];
  }
}
// End of the C++11 raw string literal
)"
