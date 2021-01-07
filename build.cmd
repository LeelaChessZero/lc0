@echo off
setlocal

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

meson build --backend %backend% --buildtype release -Dbuild_backends=false -Dgtest=false -Ddefault_library=static

if errorlevel 1 exit /b

pause

cd build

msbuild /m /p:Configuration=Release /p:Platform=x64 /p:WholeProgramOptimization=true ^
/p:PreferredToolArchitecture=x64 rescorer.sln /filelogger
