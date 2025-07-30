@echo off
where /q tar
if errorlevel 1 goto error

cd /d %~dp0

cls
echo Installing the CUDA dlls required by the Lc0 cuda backend.

echo 1/4. Downloading cudart.
curl -# --ssl-no-revoke -o tmp_cudart.zip https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.9.37-archive.zip"
if errorlevel 1 goto error

echo 2/4. Extracting files.
tar -xzOf tmp_cudart.zip cuda_cudart-windows-x86_64-12.9.37-archive/bin/cudart64_12.dll >cudart64_12.dll
if errorlevel 1 goto error

tar -xzOf tmp_cudart.zip cuda_cudart-windows-x86_64-12.9.37-archive/LICENSE >CUDA.txt

del /q tmp_cudart.zip

echo 3/4. Downloading cublas.
curl -# --ssl-no-revoke -o tmp_cublas.zip https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.9.0.13-archive.zip"
if errorlevel 1 goto error

echo 4/4. Extracting files.
tar -xzOf tmp_cublas.zip libcublas-windows-x86_64-12.9.0.13-archive/bin/cublas64_12.dll >cublas64_12.dll
if errorlevel 1 goto error

tar -xzOf tmp_cublas.zip libcublas-windows-x86_64-12.9.0.13-archive/bin/cublasLt64_12.dll >cublasLt64_12.dll
if errorlevel 1 goto error

del /q tmp_cublas.zip

echo Installation successful.
pause
exit /b

:error
cls
echo Installation failed - you will have to download cuda 12.9 yourself.
pause

