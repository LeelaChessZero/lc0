SET PGO=false
IF %APPVEYOR_REPO_TAG%==true IF %DX%==false IF %ONNX_DML%==false SET PGO=true
IF %PGO%==false msbuild "C:\projects\lc0\build\lc0.sln" /m /p:WholeProgramOptimization=true /logger:"C:\Program Files\AppVeyor\BuildAgent\Appveyor.MSBuildLogger.dll"
IF EXIST build\lc0.pdb del build\lc0.pdb
IF %PGO%==true msbuild "C:\projects\lc0\build\lc0.sln" /m /p:WholeProgramOptimization=PGInstrument /logger:"C:\Program Files\AppVeyor\BuildAgent\Appveyor.MSBuildLogger.dll"
IF ERRORLEVEL 1 EXIT
cd build
IF %NAME%==cpu-openblas copy C:\cache\OpenBLAS\dist64\bin\libopenblas.dll
IF %NAME%==cpu-dnnl copy C:\cache\%DNNL_NAME%\bin\dnnl.dll
IF %NAME%==onednn copy C:\cache\%DNNL_NAME%\bin\dnnl.dll
IF %NAME%==onednn copy dnnl.dll ..
copy "%MIMALLOC_PATH%"\out\msvc-x64\Release\mimalloc-override.dll
copy "%MIMALLOC_PATH%"\out\msvc-x64\Release\mimalloc-redirect.dll
IF %PGO%==true (
  IF %OPENCL%==true copy C:\cache\opencl-nug.0.777.77\build\native\bin\OpenCL.dll
  IF %CUDA%==true copy "%CUDA_PATH%"\bin\*.dll
  IF %CUDNN%==true copy "%CUDA_PATH%"\cuda\bin\cudnn64_7.dll
  lc0 benchmark --num-positions=1 --backend=trivial --movetime=10000
)
cd ..
IF %PGO%==true msbuild "C:\projects\lc0\build\lc0.sln" /m /p:WholeProgramOptimization=PGOptimize /p:DebugInformationFormat=ProgramDatabase /logger:"C:\Program Files\AppVeyor\BuildAgent\Appveyor.MSBuildLogger.dll"
