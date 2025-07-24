Lc0

Lc0 is a UCI-compliant chess engine designed to play chess via
neural network, specifically those of the LeelaChessZero project
(https://lczero.org).

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

License

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

