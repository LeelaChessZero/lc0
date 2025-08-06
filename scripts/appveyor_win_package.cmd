7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip %APPVEYOR_BUILD_FOLDER%\build\lc0.exe
IF %NAME%==gpu-nvidia-cuda12 appveyor DownloadFile "https://github.com/LeelaChessZero/lczero-client/releases/latest/download/lc0-training-client.exe"
IF %NAME%==gpu-nvidia-cuda12 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip lc0-training-client.exe
type COPYING |more /P > dist\COPYING
7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\COPYING
7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip c:\cache\%NET%.pb.gz
type "%MIMALLOC_PATH%"\readme.md |more /P > dist\mimalloc-readme.md
type "%MIMALLOC_PATH%"\LICENSE |more /P > dist\mimalloc-LICENSE
7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip "%MIMALLOC_PATH%"\out\msvc-x64\Release\mimalloc-override.dll
7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip "%MIMALLOC_PATH%"\out\msvc-x64\Release\mimalloc-redirect.dll
7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\mimalloc-readme.md
7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\mimalloc-LICENSE
IF %CUDA%==true copy lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%-nodll.zip
IF %NAME%==cpu-openblas 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip C:\cache\OpenBLAS\dist64\bin\libopenblas.dll
IF %NAME%==cpu-dnnl 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip C:\cache\%DNNL_NAME%\bin\dnnl.dll
IF %NAME%==onednn 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip C:\cache\%DNNL_NAME%\bin\dnnl.dll
IF %OPENCL%==true 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip C:\cache\opencl-nug.0.777.77\build\native\bin\OpenCL.dll
IF %CUDNN%==true 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip "%CUDA_PATH%\bin\cudart64_101.dll" "%CUDA_PATH%\bin\cublas64_10.dll" "%CUDA_PATH%\bin\cublasLt64_10.dll"
IF %CUDNN%==true 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip "%CUDA_PATH%\cuda\bin\cudnn64_7.dll"
IF %NAME%==gpu-nvidia-cuda11 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip "%CUDA_PATH%\bin\cudart64_110.dll" "%CUDA_PATH%\bin\cublas64_11.dll" "%CUDA_PATH%\bin\cublasLt64_11.dll"
IF %NAME%==gpu-nvidia-cuda12 (
  7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip "%CUDA_PATH%\bin\cudart64_12.dll" "%CUDA_PATH%\bin\cublas64_12.dll" "%CUDA_PATH%\bin\cublasLt64_12.dll"
  type dist\install-cuda_12_9.cmd |more /P > dist\install.cmd
  7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%-nodll.zip .\dist\install.cmd
)
IF %NAME%==cpu-dnnl (
  copy "%PKG_FOLDER%\%DNNL_NAME%\LICENSE" dist\DNNL-LICENSE
  copy "%PKG_FOLDER%\%DNNL_NAME%\THIRD-PARTY-PROGRAMS" dist\DNNL-THIRD-PARTY-PROGRAMS
  7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\DNNL-LICENSE
  7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\DNNL-THIRD-PARTY-PROGRAMS
)
IF %NAME%==onednn (
  copy "%PKG_FOLDER%\%DNNL_NAME%\LICENSE" dist\DNNL-LICENSE
  copy "%PKG_FOLDER%\%DNNL_NAME%\THIRD-PARTY-PROGRAMS" dist\DNNL-THIRD-PARTY-PROGRAMS
  7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\DNNL-LICENSE
  7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\DNNL-THIRD-PARTY-PROGRAMS
)
IF %ONNX%==true (
  copy "%PKG_FOLDER%\%ONNX_NAME%\LICENSE" dist\ONNX-LICENSE
  copy "%PKG_FOLDER%\%ONNX_NAME%\ThirdPartyNotices.txt" dist\ONNX-ThirdPartyNotices.txt
  7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip "%PKG_FOLDER%\%ONNX_NAME%\runtimes\win-x64\native\onnxruntime.dll"
  7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\ONNX-LICENSE
  7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\ONNX-ThirdPartyNotices.txt
  copy lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%-trt.zip
  ren lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%-dml.zip
  7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%-dml.zip %APPVEYOR_BUILD_FOLDER%\build\lc0-dml.exe
  7z rn lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%-dml.zip lc0-dml.exe lc0.exe
  type dist\README-onnx-dml.txt |more /P > dist\README.txt
  type dist\install-dml.cmd |more /P > dist\install.cmd
  7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%-dml.zip .\dist\README.txt
  7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%-dml.zip .\dist\install.cmd
  7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%-trt.zip %APPVEYOR_BUILD_FOLDER%\build\lc0-trt.exe
  7z rn lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%-trt.zip lc0-trt.exe lc0.exe
  7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%-trt.zip "%PKG_FOLDER%\%ONNX_NAME%\runtimes\win-x64\native\onnxruntime_providers_shared.dll"
  7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%-trt.zip "%PKG_FOLDER%\%ONNX_NAME_TWO%\lib\onnxruntime_providers_cuda.dll"
  7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%-trt.zip "%PKG_FOLDER%\%ONNX_NAME_TWO%\lib\onnxruntime_providers_tensorrt.dll"
  type dist\README-onnx-trt.txt |more /P > dist\README.txt
  type dist\install-trt.cmd |more /P > dist\install.cmd
  7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%-trt.zip .\dist\README.txt
  7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%-trt.zip .\dist\install.cmd
)
IF %OPENCL%==true type scripts\check_opencl.bat |more /P > dist\check_opencl.bat
IF %OPENCL%==true 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\check_opencl.bat
IF %DX%==true type scripts\check_dx.bat |more /P > dist\check_dx.bat
IF %DX%==true 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\check_dx.bat
IF %CUDA%==true copy "%CUDA_PATH%\EULA.txt" dist\CUDA.txt
IF %CUDA%==true type dist\README-cuda.txt |more /P > dist\README.txt
IF %CUDA%==true 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\README.txt .\dist\CUDA.txt
IF %CUDNN%==true copy "%CUDA_PATH%\cuda\NVIDIA_SLA_cuDNN_Support.txt" dist\CUDNN.txt
IF %CUDNN%==true 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\CUDNN.txt
IF EXIST lc0-%APPVEYOR_REPO_TAG_NAME%-windows-gpu-dx12.zip ren lc0-%APPVEYOR_REPO_TAG_NAME%-windows-gpu-dx12.zip lc0-%APPVEYOR_REPO_TAG_NAME%-windows10-gpu-dx12.zip
