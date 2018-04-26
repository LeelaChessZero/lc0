# Building and running this version of lc0.

Building is not very streightforward, but here's roughly the process:

## Linux

1. Install tensorflow_cc by following steps described [here](https://github.com/FloopCZ/tensorflow_cc).
2. Install ninja and meson
3. Go to lc0/ and run ./build.sh

## Windows

Building for windows is currently complicated, you can try executeing steps listed [here](https://github.com/glinscott/leela-chess/issues/334#issuecomment-382848569).

Alternatively, you can use pre-built binary:

1. Install CUDA v9.0 (not 9.1)
2. Inscall cuDNN for CUDA v9.0
3. Take latest binary from [here](https://crem.xyz/lc0/)

That's it.