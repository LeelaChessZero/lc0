@echo off
where /q tar
if errorlevel 1 goto error

cd /d %~dp0

cls
echo Installing the CUDA dlls required by the Lc0 cuda backend.

echo 1/4. Downloading cudart.
curl -# --ssl-no-revoke -o tmp_cudart.zip https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-11.8.89-archive.zip"
if errorlevel 1 goto error

echo 2/4. Extracting files.
tar -xzOf tmp_cudart.zip cuda_cudart-windows-x86_64-11.8.89-archive/bin/cudart64_110.dll >cudart64_110.dll
if errorlevel 1 goto error

del /q tmp_cudart.zip

echo 3/4. Downloading cublas.
curl -# --ssl-no-revoke -o tmp_cublas.zip https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-11.11.3.6-archive.zip"
if errorlevel 1 goto error

echo 4/4. Extracting files.
tar -xzOf tmp_cublas.zip libcublas-windows-x86_64-11.11.3.6-archive/bin/cublas64_11.dll >cublas64_11.dll
if errorlevel 1 goto error

tar -xzOf tmp_cublas.zip libcublas-windows-x86_64-11.11.3.6-archive/bin/cublasLt64_11.dll >cublasLt64_11.dll
if errorlevel 1 goto error

del /q tmp_cublas.zip

echo Installation successful.
pause
exit /b

:error
cls
echo Installation failed - see the README for an alternative approach.
pause

