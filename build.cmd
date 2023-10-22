@echo off
setlocal

rem 1. Set the following for the options you want to build.
set CUDNN=false
set CUDA=false
set DX12=false
set OPENCL=true
set MKL=false
set DNNL=true
set OPENBLAS=true
set EIGEN=true
set TEST=false
set ONNX=true


rem 2. Edit the paths for the build dependencies.
set CUDA_PATH=%CUDNN_PATH%
set CUDNN_PATH=C:\cudnn-windows-x86_64-8.5.0.96_cuda10-archive\cudnn-windows-x86_64-8.5.0.96_cuda10-archive
set OPENBLAS_PATH=C:\Users\cenam\Documents\Builds\OpenBLAS-0.3.3-win-oldthread\dist64
set MKL_PATH=C:\Program Files (x86)\Intel\oneAPI\mkl\2023.1.0
set DNNL_PATH=C:\Users\cenam\Documents\Builds\dnnl_win_2.6.0_cpu_vcomp_gpu_vcomp\dnnl_win_2.6.0_cpu_vcomp_gpu_vcomp
setx OPENCL_LIBS=C:\Program Files (x86)\IntelSWTools\system_studio_2020\OpenCL\sdk\lib\x64
setx OPENCL_INCS=C:\Program Files (x86)\IntelSWTools\system_studio_2020\OpenCL\sdk\include
set ONNX_PATH=C:\Users\cenam\Documents\onnxruntime1.16custom
set ISPC_PATH=C:\Users\cenam\Downloads\ispc-trunk-windows\ispc-trunk-windows\bin

rem 3. In most cases you won't need to change anything further down.
echo Deleting build directory:
rd /s build

set CC=cl
set CXX=cl
set CC_LD=link
set CXX_LD=link

if exist "C:\Program Files\Microsoft Visual Studio\2022" (
  where /q cl
  if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
  set backend=vs2022
) else (
  where /q cl
  if errorlevel 1 call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
  set backend=vs2017
)


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

meson build --backend %backend% --buildtype release -Ddx=%DX12% -Dcudnn=%CUDNN% -Dplain_cuda=%CUDA% -Dblas=%BLAS% -Dmkl=%MKL% ^
-Dgtest=%TEST% -Donnx_libdir="%ONNX_PATH%\runtimes\win-x64\native" -Donnx_include="%ONNX_PATH%\build\native\include"  ^
-Dopencl=%OPENCL% -Dcudnn_include="%CUDNN_INCLUDE_PATH%" -Dcudnn_libdirs="%CUDNN_LIB_PATH%" ^
-Donednn=false -Dispc=true -Dispc_native_only=false -Ddnnl=true -Dmkl_include="%MKL_PATH%\include" ^
-Dmkl_libdirs="%MKL_PATH%\lib\intel64" -Ddnnl_dir="%DNNL_PATH%" ^
-Dopencl_libdirs="%OPENCL_LIBS%" -Dopencl_include="%OPENCL_INCS%" ^
-Dopenblas_include="%OPENBLAS_PATH%\include" -Dopenblas_libdirs="%OPENBLAS_PATH%\lib" ^
-Ddefault_library=static

if errorlevel 1 exit /b

pause

cd build

msbuild /m /p:Configuration=Release /p:Platform=x64 /p:WholeProgramOptimization=true  ^
/p:PreferredToolArchitecture=x64 -p:Instruction_Set=AdvancedVectorExtensions2 lc0.sln /filelogger
