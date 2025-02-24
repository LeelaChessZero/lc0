@echo off
setlocal

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
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019" (
  where /q cl
  if errorlevel 1 call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
  set backend=vs2019
) else (
  where /q cl
  if errorlevel 1 call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
  set backend=vs2017
)

meson setup build --backend %backend% --buildtype release -Drescorer=true -Dlc0=false -Dgtest=false -Ddefault_library=static

if errorlevel 1 exit /b

pause

cd build

msbuild /m /p:Configuration=Release /p:Platform=x64 /p:WholeProgramOptimization=true ^
/p:PreferredToolArchitecture=x64 lc0.sln /filelogger
