@echo off
setlocal

rem 1. Set the following for the options you want to build.
rem SYCL can be off, l0, amd or nvidia.
set SYCL=l0
set CUDNN=true
set CUDA=true
set DX12=false
set OPENCL=false
set MKL=false
set DNNL=true
set OPENBLAS=false
set EIGEN=false
set TEST=false

rem 2. Edit the paths for the build dependencies.
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0
set CUDNN_PATH=%CUDA_PATH%
set OPENBLAS_PATH=C:\OpenBLAS
set MKL_PATH=C:\Program Files (x86)\Intel\oneAPI\mkl\latest\
set DNNL_PATH=C:\Program Files (x86)\Intel\oneAPI\dnnl\latest\cpu_iomp
set OPENCL_LIB_PATH=%CUDA_PATH%\lib\x64
set OPENCL_INCLUDE_PATH=%CUDA_PATH%\include

rem 3. In most cases you won't need to change anything further down.
echo Deleting build directory:
rd /s build

rem Use cl for C files to get a resource compiler as needed for zlib.
set CC=cl
set CXX=icx

set BLAS=true
if %MKL%==false if %DNNL%==false if %OPENBLAS%==false if %EIGEN%==false set BLAS=false

if "%CUDA_PATH%"=="%CUDNN_PATH%" (
  set CUDNN_LIB_PATH=%CUDNN_PATH%\lib\x64
  set CUDNN_INCLUDE_PATH=%CUDNN_PATH%\include
) else (
  set CUDNN_LIB_PATH=%CUDA_PATH%\lib\x64,%CUDNN_PATH%\lib\x64
  set CUDNN_INCLUDE_PATH=%CUDA_PATH%\include,%CUDNN_PATH%\include
)

if %CUDNN%==true set PATH=%CUDA_PATH%\bin;%PATH%

meson setup build --buildtype release -Ddx=%DX12% -Dcudnn=%CUDNN% -Dplain_cuda=%CUDA% ^
-Dopencl=%OPENCL% -Dblas=%BLAS% -Dmkl=%MKL% -Dopenblas=%OPENBLAS% -Ddnnl=%DNNL% -Dgtest=%TEST% ^
-Dcudnn_include="%CUDNN_INCLUDE_PATH%" -Dcudnn_libdirs="%CUDNN_LIB_PATH%" ^
-Dmkl_include="%MKL_PATH%\include" -Dmkl_libdirs="%MKL_PATH%\lib\intel64" -Ddnnl_dir="%DNNL_PATH%" ^
-Dopencl_libdirs="%OPENCL_LIB_PATH%" -Dopencl_include="%OPENCL_INCLUDE_PATH%" ^
-Dopenblas_include="%OPENBLAS_PATH%\include" -Dopenblas_libdirs="%OPENBLAS_PATH%\lib" ^
-Ddefault_library=static -Dsycl=%SYCL% -Db_vscrt=md

if errorlevel 1 exit /b

pause

cd build

ninja