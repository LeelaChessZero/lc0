@echo off
setlocal

rem 1. Set the following for the options you want to build.
set CUDNN=true
set DX12=false
set OPENCL=false
set MKL=false
set DNNL=false
set OPENBLAS=false
set EIGEN=false
set TEST=false

rem 2. Edit the paths for the build dependencies.
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0
set CUDNN_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0
set OPENBLAS_PATH=C:\OpenBLAS
set MKL_PATH=C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl
set DNNL_PATH=C:\dnnl_win_1.1.1_cpu_vcomp
set OPENCL_LIB_PATH=%CUDA_PATH%\lib\x64
set OPENCL_INCLUDE_PATH=%CUDA_PATH%\include

rem 3. In most cases you won't need to change anything further down.
echo Deleting build directory:
rd /s build

if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019" (
  where /q cl
  if errorlevel 1 call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
  set backend=vs2019
) else (
  where /q cl
  if errorlevel 1 call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
  set backend=vs2017
)

if "%CUDA_PATH%"=="%CUDNN_PATH%" (
  set CUDNN_LIB_PATH=%CUDNN_PATH%\lib\x64
  set CUDNN_INCLUDE_PATH=%CUDNN_PATH%\include
) else (
  set CUDNN_LIB_PATH=%CUDA_PATH%\lib\x64,%CUDNN_PATH%\lib\x64
  set CUDNN_INCLUDE_PATH=%CUDA_PATH%\include,%CUDNN_PATH%\include
)

if %CUDNN%==true set PATH=%CUDA_PATH%\bin;%PATH%

meson build --backend %backend% --buildtype release -Ddx=%DX12% -Dcudnn=%CUDNN% -Dopencl=%OPENCL% ^
-Dblas=true -Dmkl=%MKL% -Dopenblas=%OPENBLAS% -Deigen=%EIGEN% -Ddnnl=%DNNL% -Dgtest=%TEST% ^
-Dcudnn_include="%CUDNN_INCLUDE_PATH%" -Dcudnn_libdirs="%CUDNN_LIB_PATH%" ^
-Dmkl_include="%MKL_PATH%\include" -Dmkl_libdirs="%MKL_PATH%\lib\intel64" -Ddnnl_dir="%DNNL_PATH%" ^
-Dopencl_libdirs="%OPENCL_LIB_PATH%" -Dopencl_include="%OPENCL_INCLUDE_PATH%" ^
-Dopenblas_include="%OPENBLAS_PATH%\include" -Dopenblas_libdirs="%OPENBLAS_PATH%\lib" ^
-Ddefault_library=static

if errorlevel 1 exit /b

pause

cd build

msbuild /m /p:Configuration=Release /p:Platform=x64 /p:WholeProgramOptimization=true ^
/p:PreferredToolArchitecture=x64 lc0.sln /filelogger
