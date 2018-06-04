// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

void __in_transform_eq(float x[4][4], __global float * restrict V, int offset, int CPpad) {
  float T1[4][4];
  
  T1[0][0] = x[0][0] - x[2][0];
  T1[0][1] = x[0][1] - x[2][1];
  T1[0][2] = x[0][2] - x[2][2];
  T1[0][3] = x[0][3] - x[2][3];
  T1[1][0] = x[1][0] + x[2][0];
  T1[1][1] = x[1][1] + x[2][1];
  T1[1][2] = x[1][2] + x[2][2];
  T1[1][3] = x[1][3] + x[2][3];
  T1[2][0] = x[2][0] - x[1][0];
  T1[2][1] = x[2][1] - x[1][1];
  T1[2][2] = x[2][2] - x[1][2];
  T1[2][3] = x[2][3] - x[1][3];
  T1[3][0] = x[1][0] - x[3][0];
  T1[3][1] = x[1][1] - x[3][1];
  T1[3][2] = x[1][2] - x[3][2];
  T1[3][3] = x[1][3] - x[3][3];
  
  V[(0*4 + 0)*CPpad + offset] = T1[0][0] - T1[0][2];
  V[(0*4 + 1)*CPpad + offset] = T1[0][1] + T1[0][2];
  V[(0*4 + 2)*CPpad + offset] = T1[0][2] - T1[0][1];
  V[(0*4 + 3)*CPpad + offset] = T1[0][1] - T1[0][3];
  V[(1*4 + 0)*CPpad + offset] = T1[1][0] - T1[1][2];
  V[(1*4 + 1)*CPpad + offset] = T1[1][1] + T1[1][2];
  V[(1*4 + 2)*CPpad + offset] = T1[1][2] - T1[1][1];
  V[(1*4 + 3)*CPpad + offset] = T1[1][1] - T1[1][3];
  V[(2*4 + 0)*CPpad + offset] = T1[2][0] - T1[2][2];
  V[(2*4 + 1)*CPpad + offset] = T1[2][1] + T1[2][2];
  V[(2*4 + 2)*CPpad + offset] = T1[2][2] - T1[2][1];
  V[(2*4 + 3)*CPpad + offset] = T1[2][1] - T1[2][3];
  V[(3*4 + 0)*CPpad + offset] = T1[3][0] - T1[3][2];
  V[(3*4 + 1)*CPpad + offset] = T1[3][1] + T1[3][2];
  V[(3*4 + 2)*CPpad + offset] = T1[3][2] - T1[3][1];
  V[(3*4 + 3)*CPpad + offset] = T1[3][1] - T1[3][3];
}

__kernel void in_transform(__global net_t * restrict in, __global float * restrict V,
                           const int C, const int Cpad,
                           const int Ppad) {
  const int W = 8;
  const int H = 8;
  const int T = W*H;
  const int WTILES = (W + 1) / 2;
  const int P = WTILES*WTILES;
  const int CPpad = Ppad * Cpad;
  
  const int block = get_global_id(0);
  const int ch = get_global_id(1);
  const int chT = ch*(T);
  
  const int block_x = block % WTILES;
  const int block_y = block / WTILES;
  
  // Tiles overlap by 2
  const int yin = 2 * block_y - 1;
  const int xin = 2 * block_x - 1;
  
  if (block < P && ch < C) {
    // Cache input tile and handle zero padding
    float x[4][4];
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        int a = xin + j;
        int b = yin + i;
        if (b >= 0 && a >= 0 && b < H && a < W) {
          x[i][j] = vload_net_t(chT + b*W + a, in);
        } else {
          x[i][j] = 0.0f;
        }
      }
    }
    
    const int offset = ch*Ppad + block;
    __in_transform_eq(x, V, offset, CPpad);
  }
}

void __out_transform_eq(__global const float * restrict M, float o[4],
                        int Kpad, int Ppad, int block_x, int block_y)
{
  const int W = 8;
  const int H = 8;
  const int WTILES = (W + 1) / 2;
  const int b = block_y * WTILES + block_x;
  const int KPpad = Kpad * Ppad;
  const int k = get_global_id(0);
  float temp_m[16];
  for (int xn = 0, xnKPpad = b*Kpad + k; xn < 16; xn++, xnKPpad += KPpad) {
    temp_m[xn] = M[xnKPpad];
  }
  
  o[0] = temp_m[0*4 + 0] + temp_m[0*4 + 1] + temp_m[0*4 + 2] +
  temp_m[1*4 + 0] + temp_m[1*4 + 1] + temp_m[1*4 + 2] +
  temp_m[2*4 + 0] + temp_m[2*4 + 1] + temp_m[2*4 + 2];
  
  o[1] = temp_m[0*4 + 1] - temp_m[0*4 + 2] - temp_m[0*4 + 3] +
  temp_m[1*4 + 1] - temp_m[1*4 + 2] - temp_m[1*4 + 3] +
  temp_m[2*4 + 1] - temp_m[2*4 + 2] - temp_m[2*4 + 3];
  
  o[2] = temp_m[1*4 + 0] + temp_m[1*4 + 1] + temp_m[1*4 + 2] -
  temp_m[2*4 + 0] - temp_m[2*4 + 1] - temp_m[2*4 + 2] -
  temp_m[3*4 + 0] - temp_m[3*4 + 1] - temp_m[3*4 + 2];
  
  o[3] = temp_m[1*4 + 1] - temp_m[1*4 + 2] - temp_m[1*4 + 3] -
  temp_m[2*4 + 1] + temp_m[2*4 + 2] + temp_m[2*4 + 3] -
  temp_m[3*4 + 1] + temp_m[3*4 + 2] + temp_m[3*4 + 3];
}

__kernel void out_transform_fused_bn(__global const float * restrict M,
                                     __global net_t * restrict Y,
                                     const int K,
                                     const int Kpad, const int Ppad,
                                     __global const net_t * restrict residual,
                                     __constant const net_t * restrict means,
                                     __constant const net_t * restrict stddivs) {
  const int W = 8;
  const int H = 8;
  const int WTILES = (W + 1) / 2;
  const int P = WTILES * WTILES;
  
  int k = get_global_id(0);
  int block = get_global_id(1);
  
  const int block_x = block % WTILES;
  const int block_y = block / WTILES;
  
  int x = 2*block_x;
  int y = 2*block_y;
  int a_ind = (y)*W + (x);
  if (k < K && block < P) {
    const int kHW = k * W * H;
    float o[4];
    __out_transform_eq(M, o, Kpad, Ppad, block_x, block_y);
    
    const float mean = vload_net_t(k, means);
    const float scale_stddiv = vload_net_t(k, stddivs);
    
    const bool pred[4] = { 1, x+1 < W, y+1 < H, x+1 < W & y+1 < H};
    
    const int a[4] = {a_ind, a_ind+1, a_ind+W, a_ind+W+1};
    
    for (int i = 0; i < 4; i++) {
      if (pred[i]) {
        o[i] = scale_stddiv * (o[i] - mean);
        if (residual) {
          o[i] += vload_net_t(kHW + a[i], residual);
        }
        o[i] = o[i] > 0 ? o[i] : 0.0f;
        vstore_net_t(o[i], kHW + a[i], Y);
      }
    }
  }
}

__kernel void out_transform_fused_bn_in(
                                        __global const float * restrict M,
                                        __global net_t * restrict Y,
                                        __global net_t * restrict V,
                                        const int K,
                                        const int Kpad, const int Ppad, const int Cpad,
                                        __global const net_t * restrict residual,
                                        __constant const net_t * restrict means,
                                        __constant const net_t * restrict stddivs,
                                        __local float * ybuf) {
  const int W = 8;
  const int H = 8;
  const int T = W*H;
  const int WTILES = (W + 1) / 2;
  const int P = WTILES * WTILES;
  const int KPpad = Kpad * Ppad;
  
  const int k = get_global_id(0);
  const int kg = get_local_id(0);
  const int block = get_global_id(1);
  
  const int block_x = block % WTILES;
  const int block_y = block / WTILES;
  
  const int yin = 2 * block_y - 1;
  const int xin = 2 * block_x - 1;
  
  
  const int x = 2*block_x;
  const int y = 2*block_y;
  int a_ind = (y)*W + (x);
  
  
  if (k < K && block < P) {
    const int a[4] = {a_ind, a_ind+1, a_ind+W, a_ind+W+1};
    const bool pred[4] = { 1, x+1 < W, y+1 < H, x+1 < W & y+1 < H};
    const int kHW = k * W * H;
    
    float o[4];
    __out_transform_eq(M, o, Kpad, Ppad, block_x, block_y);
    
    const float mean = vload_net_t(k, means);
    const float scale_stddiv = vload_net_t(k, stddivs);
    
    for (int i = 0; i < 4; i++) {
      if (pred[i]) {
        o[i] = scale_stddiv * (o[i] - mean);
        if (residual) {
          o[i] += vload_net_t(kHW + a[i], residual);
        }
        o[i] = o[i] > 0 ? o[i] : 0.0f;
        ybuf[kg * T + a[i]] = o[i];
        if (Y) {
          vstore_net_t(o[i], kHW + a[i], Y);
        }
      }
    }
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  if (block < P && k < K) {
    const int CPpad = Ppad * Cpad;
    // Cache input tile and handle zero padding
    float xx[4][4];
    for (int i = 0; i < 4; i++) {
      int b = yin + i;
      for (int j = 0; j < 4; j++) {
        int a = xin + j;
        if (b >= 0 && a >= 0 && b < H && a < W) {
          xx[i][j] = ybuf[kg * T + b*W + a];
        } else {
          xx[i][j] = 0.0f;
        }
      }
    }
    
    const int offset = k*Ppad + block;
    __in_transform_eq(xx, V, offset, CPpad);
  }
}

// End of the C++11 raw string literal
)"
