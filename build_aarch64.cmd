@echo off
setlocal

rem 1. Set the following for the options you want to build.
set DX12=false
set OPENCL=false
set MKL=false
set DNNL=false
set OPENBLAS=false
set EIGEN=true
set TEST=false

rem 2. Edit the paths for the build dependencies.
set OPENBLAS_PATH=C:\OpenBLAS
set MKL_PATH=C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl
set DNNL_PATH=C:\dnnl_win_1.1.1_cpu_vcomp
set OPENCL_LIB_PATH=%CUDA_PATH%\lib\x64
set OPENCL_INCLUDE_PATH=%CUDA_PATH%\include

rem 3. In most cases you won't need to change anything further down.
echo Deleting build directory:
rd /s build

set CC=cl
set CXX=cl
set CC_LD=link
set CXX_LD=link

if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019" (
  where /q cl
  if errorlevel 1 call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x86_arm64
  set backend=vs2019
) else (
  where /q cl
  if errorlevel 1 call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x86_arm64
  set backend=vs2017
)

set BLAS=true
if %MKL%==false if %DNNL%==false if %OPENBLAS%==false if %EIGEN%==false set BLAS=false

meson build --backend %backend% --buildtype release -Ddx=%DX12% -Dispc=false ^
-Dopencl=%OPENCL% -Dblas=%BLAS% -Dmkl=%MKL% -Dopenblas=%OPENBLAS% -Ddnnl=%DNNL% -Dgtest=%TEST% ^
-Dmkl_include="%MKL_PATH%\include" -Dmkl_libdirs="%MKL_PATH%\lib\intel64" -Ddnnl_dir="%DNNL_PATH%" ^
-Dopencl_libdirs="%OPENCL_LIB_PATH%" -Dopencl_include="%OPENCL_INCLUDE_PATH%" ^
-Dopenblas_include="%OPENBLAS_PATH%\include" -Dopenblas_libdirs="%OPENBLAS_PATH%\lib" ^
-Ddefault_library=static --cross-file cross-files/aarch64-windows

if errorlevel 1 exit /b

pause

cd build

msbuild /m /p:Configuration=Release /p:Platform=arm64 /p:WholeProgramOptimization=true ^
/p:PreferredToolArchitecture=arm64 lc0.sln /filelogger
