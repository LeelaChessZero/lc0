// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.

R"(
    __kernel void global_avg_pooling(
                   const int channels,
                   __global const net_t * restrict in,
                   __global net_t * restrict out) {

        const int col = get_global_id(0);  // column
        const int c = get_global_id(1);  // channel

        const int lid = get_local_id(0);

        __local net_t row_acc[BOARD_SIZE];

        if (c < channels && col < BOARD_SIZE) {

            net_t acc = 0.0f;

            for ( int i = 0; i < BOARD_SIZE; i++) {
                acc += vload_net_t(c * BOARD_SQUARES + i * BOARD_SIZE + col, in);
            }
            row_acc[lid] = acc;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0) {
            net_t acc = 0.0f;
            for ( int i = 0; i < BOARD_SIZE; i++) {
                acc += row_acc[i];
            }
            acc = acc/BOARD_SQUARES;
            vstore_net_t(acc, c, out);
        }
    }

    __kernel void apply_se(
                  const int channels,
                  const int batch_size,
                  __global const net_t * restrict input,
                  __global net_t * restrict residual,
                  __global const net_t * restrict fc_out) {

        const int col = get_global_id(0);  // column
        const int c = get_global_id(1);  // channel

        const int batch = c / channels;

        if (c < batch_size * channels && col < BOARD_SIZE) {
            net_t gamma = vload_net_t(c + batch * channels, fc_out);
            gamma = 1.0f/(1.0f + exp(-gamma)); // Sigmoid
            net_t beta = vload_net_t(c + batch * channels + channels, fc_out);

            for ( int i = 0; i < BOARD_SIZE; i++) {
                const int idx = c * BOARD_SQUARES + i * BOARD_SIZE + col;
                const net_t in = vload_net_t(idx, input);
                const net_t res = vload_net_t(idx, residual);

                net_t val = gamma * in + res + beta;

                val = val > 0.0f ? val : 0.0f;

                vstore_net_t(val, idx, residual);
            }
        }
    }
// End of the C++11 raw string literal
)"
