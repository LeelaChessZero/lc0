@echo off
where /q tar
if errorlevel 1 goto error

cd /d %~dp0

cls

echo This script will download and install the CUDA/cuDNN/tensorRT dlls required by the Lc0 onnx-trt backend.
echo(
echo If you are using a metered internet connection, be aware the download will be arounbd 3 Gb.
echo(
pause

echo Installing the CUDA dlls required by the Lc0 onnx-trt backend.

echo 1/6. Downloading cudart.
curl -# --ssl-no-revoke -o tmp_cudart.zip https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.9.79-archive.zip"
if errorlevel 1 goto error

echo 2/6. Extracting files.
tar -xzOf tmp_cudart.zip cuda_cudart-windows-x86_64-12.9.79-archive/bin/cudart64_12.dll >cudart64_12.dll
if errorlevel 1 goto error

tar -xzOf tmp_cudart.zip cuda_cudart-windows-x86_64-12.9.79-archive/LICENSE >CUDA.txt

del /q tmp_cudart.zip

echo 3/6. Downloading cublas.
curl -# --ssl-no-revoke -o tmp_cublas.zip https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.9.1.4-archive.zip"
if errorlevel 1 goto error

echo 4/6. Extracting files.
tar -xzOf tmp_cublas.zip libcublas-windows-x86_64-12.9.1.4-archive/bin/cublas64_12.dll >cublas64_12.dll
if errorlevel 1 goto error

tar -xzOf tmp_cublas.zip libcublas-windows-x86_64-12.9.1.4-archive/bin/cublasLt64_12.dll >cublasLt64_12.dll
if errorlevel 1 goto error

del /q tmp_cublas.zip

echo 5/6. Downloading cufft.
curl -# --ssl-no-revoke -o tmp_cufft.zip https://developer.download.nvidia.com/compute/cuda/redist/libcufft/windows-x86_64/libcufft-windows-x86_64-11.4.1.4-archive.zip"
if errorlevel 1 goto error

echo 6/6. Extracting files.
tar -xzOf tmp_cufft.zip libcufft-windows-x86_64-11.4.1.4-archive/bin/cufft64_11.dll >cufft64_11.dll
if errorlevel 1 goto error

del /q tmp_cufft.zip

echo Installing the cuDNN dlls required by the Lc0 onnx-trt backend.

echo 1/2. Downloading cudnn.
curl -# --ssl-no-revoke -o tmp_cudnn.zip https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.11.0.98_cuda12-archive.zip"
if errorlevel 1 goto error

echo 2/2. Extracting files.
tar -xzOf tmp_cudnn.zip cudnn-windows-x86_64-9.11.0.98_cuda12-archive/bin/cudnn64_9.dll >cudnn64_9.dll
if errorlevel 1 goto error

tar -xzOf tmp_cudnn.zip cudnn-windows-x86_64-9.11.0.98_cuda12-archive/bin/cudnn_graph64_9.dll >cudnn_graph64_9.dll
if errorlevel 1 goto error

tar -xzOf tmp_cudnn.zip cudnn-windows-x86_64-9.11.0.98_cuda12-archive/LICENSE >CUDNN.txt

del /q tmp_cudnn.zip

echo Installing the tensorRT dlls required by the Lc0 onnx-trt backend.

echo 1/2. Downloading tensorRT.
curl -# --ssl-no-revoke -o tmp_tensorrt.zip https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.12.0/zip/TensorRT-10.12.0.36.Windows.win10.cuda-12.9.zip"
if errorlevel 1 goto error

echo 2/2. Extracting files.
tar -xzOf tmp_tensorrt.zip TensorRT-10.12.0.36/lib/nvinfer_10.dll >nvinfer_10.dll
if errorlevel 1 goto error

tar -xzOf tmp_tensorrt.zip TensorRT-10.12.0.36/lib/nvinfer_builder_resource_10.dll >nvinfer_builder_resource_10.dll
if errorlevel 1 goto error

tar -xzOf tmp_tensorrt.zip TensorRT-10.12.0.36/lib/nvinfer_plugin_10.dll >nvinfer_plugin_10.dll
if errorlevel 1 goto error

tar -xzOf tmp_tensorrt.zip TensorRT-10.12.0.36/lib/nvonnxparser_10.dll >nvonnxparser_10.dll
if errorlevel 1 goto error

tar -xzOf tmp_tensorrt.zip TensorRT-10.12.0.36/doc/Readme.txt >TENSORRT.txt

del /q tmp_tensorrt.zip

pause
exit /b

:error
cls
echo Installation failed - see the README for alternative download instructions.
pause

