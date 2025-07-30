# Lc0

Lc0 is a UCI-compliant chess engine designed to play chess via
neural network, specifically those of the LeelaChessZero project
(https://lczero.org).

# Installation

Summary: run `instrall.cmd` and follow the instructions.

To run this version you will also need several dll files from NVIDA's
CUDA, cuDNN and TensorRT. Those dlls can either be on the system path
from a separate installation of these libraries, or can be placed
directly in the Lc0 folder. Either way, you will get an error message
for any that isn't found.

The dlls needed are the following:

1. CUDA
* cublas64_12.dll
* cublasLt64_12.dll
* cudart64_12.dll
* cufft64_11.dll

2. cuDNN
* cudnn64_9.dll
* cudnn_graph64_9.dll

3. TensorRT:
* nvinfer_10.dll
* nvinfer_builder_resource_10.dll
* nvinfer_plugin_10.dll
* nvonnxparser_10.dll

The install.cmd script included in this package will download the
CUDA and cuDNN files needed and will open the TensorRT download page
using your browser. If it fails, you can download the files manually
using the following addresses, the dlls are in the `bin` directory
in the CUDA/cuDNN zips and the `lib` directory in the TensorRT zip.

* https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.9.79-archive.zip
* https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.9.1.4-archive.zip
* https://developer.download.nvidia.com/compute/cuda/redist/libcufft/windows-x86_64/libcufft-windows-x86_64-11.4.1.4-archive.zip
* https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.11.0.98_cuda12-archive.zip
* https://developer.nvidia.com/tensorrt/download/10x#trt1012

The TensorRT link will take you to the download page, after
registering go to the "TensorRT 10.12 GA for x86_64 Architecture"
section and get the "TensorRT 10.12 GA for Windows 10, 11,
Server 2022 and CUDA 12.0 to 12.9 ZIP Package".

Finally, if Lc0 still won't run, get the latest Visual C++
redistributable from: https://aka.ms/vs/17/release/vc_redist.x64.exe

# Running

When running Lc0 with a new network file, it will take some time to
create the optimized model to use. This is normal. The model will be
cached for future runs in the `trt_cache` folder, so next time it will
be faster. If you want to experiment you can rename the `trt_cache`
folder and rerun, sometimes TensorRT will generate a different model
that may be faster. Moreover, if you are having issues, you can
delete/rename the cache and rerun.

# License

Leela Chess is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Leela Chess is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or
combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
modified version of those libraries), containing parts covered by the
terms of the respective license agreement, the licensors of this
Program grant you additional permission to convey the resulting work.

