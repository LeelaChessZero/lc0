// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

__kernel
__attribute__((work_group_size_hint(8, 16, 1)))
void convolve1(
               __global const net_t * restrict in,
               __global net_t * restrict merge,
               __global const net_t * restrict weights,
               __local float * channel_buff,
               __local float * row_buff) {
  // cl::NDRange global(channels, outputs, row);
  const int c   = get_global_id(0);  // channel
  const int o   = get_global_id(1);  // output
  const int row_batch = get_global_id(2);  // row * batch_size

  const int width = 8;
  const int height = 8;
  const int boardsize = width * height;

  const int row = row_batch % 8;
  const int batch = row_batch / 8;

  const int channels = get_global_size(0);
  const int outputs  = get_global_size(1);

  const int input_offset = batch * boardsize * channels;
  const int merge_offset = batch * boardsize * (channels >> 3) * outputs;

  // cl::NDRange local(2, (1->32), 1);
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int chan_buff_size = 8;
  const int out_buff_size  = get_local_size(1);
  const int row_buff_size  = 7;
  const int chan_shift     = 3;
  // input = channels * height * width
  // output = outputs * height * width
  // weights = output * channels * filter
  // merge = channels * outputs * height * width
  const int strip_size = width;
  // Copy the input channels (strips) locally
  if (out_buff_size < 8 && ly == 0) {
    // strip-row
    for (int w = 0; w < width; w++) {
      channel_buff[lx * width + w] =
      vload_net_t((c * height + row) * width + w + input_offset, in);
    }
  } else if (out_buff_size >= 8 && ly < 8) {
    // Every thread copies a column
    channel_buff[lx * width + ly] = vload_net_t((c * height + row) * width +
	    ly + input_offset, in);
  }
  // Copy the filter we are applying locally
  __private float filter_buff = vload_net_t((o * channels + c), weights);
  barrier(CLK_LOCAL_MEM_FENCE);
  int out_lane = 0;
  int out_cw   = 0;
#pragma unroll
  for (int cw = 0; cw < width; cw++) {
    int fid = lx * strip_size;
    float out  = channel_buff[fid + cw] * filter_buff;
    row_buff[(ly * chan_buff_size + lx) * row_buff_size + out_lane] = out;
    out_lane++;
    // Row buffer full or last lane?
    if (out_lane == row_buff_size || (cw == width - 1)) {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (lx < out_lane) {
        float val;
        val  = row_buff[(ly * chan_buff_size + 0) * row_buff_size + lx];
        val += row_buff[(ly * chan_buff_size + 1) * row_buff_size + lx];
        val += row_buff[(ly * chan_buff_size + 2) * row_buff_size + lx];
        val += row_buff[(ly * chan_buff_size + 3) * row_buff_size + lx];
        val += row_buff[(ly * chan_buff_size + 4) * row_buff_size + lx];
        val += row_buff[(ly * chan_buff_size + 5) * row_buff_size + lx];
        val += row_buff[(ly * chan_buff_size + 6) * row_buff_size + lx];
        val += row_buff[(ly * chan_buff_size + 7) * row_buff_size + lx];
        vstore_net_t(val, (((c >> chan_shift) * height + row) * width +
		    out_cw + lx) * outputs + o + merge_offset, merge);
      }
      out_cw  += row_buff_size;
      out_lane = 0;
    }
  }
}

__kernel void merge_bn(
                       __global const net_t * restrict in,
                       __global net_t * restrict out,
                       __private const int channels,
                       __constant const net_t * restrict biases) {
  // cl::NDRange global(outputs, 8*8);
  const int gx = get_global_id(0);
  const int gy = get_global_id(1);
  const int batch = get_global_id(2);
  const int output = gx;
  const int b = gy;
  const int outputs = get_global_size(0);
  const int width = 8;
  const int height = 8;
  const int boardsize = width * height;
  const int o = output;
  float sum = 0;
  for (int c = 0; c < channels; c++) {
    sum += vload_net_t(batch * channels * boardsize * outputs +
	    (c * boardsize + b) * outputs + o, in);
  }
  if (biases) {
    const float bias = vload_net_t(o, biases);

    sum = sum + bias;
    sum = sum > 0 ? sum : 0.0f;
  }
  vstore_net_t(sum, batch * outputs * boardsize + o * boardsize + b, out);
}

// End of the C++11 raw string literal
)"
