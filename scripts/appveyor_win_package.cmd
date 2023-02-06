7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip %APPVEYOR_BUILD_FOLDER%\build\lc0.exe
appveyor DownloadFile "https://ci.appveyor.com/api/projects/LeelaChessZero/lczero-client/artifacts/lc0-training-client.exe?branch=release&pr=false&job=Environment%%3A%%20NAME%%3D.exe%%2C%%20GOOS%%3Dwindows"
7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip lc0-training-client.exe
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
IF %CUDNN%==true 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip "%CUDA_PATH%\bin\cudart64_100.dll" "%CUDA_PATH%\bin\cublas64_100.dll"
IF %CUDNN%==true 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip "%CUDA_PATH%\cuda\bin\cudnn64_7.dll"
IF %CUDA%==true 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip "%CUDA_PATH%\bin\cudart64_110.dll" "%CUDA_PATH%\bin\cublas64_11.dll" "%CUDA_PATH%\bin\cublasLt64_11.dll"
IF %NAME%==cpu-dnnl copy "%PKG_FOLDER%\%DNNL_NAME%\LICENSE" dist\DNNL-LICENSE
IF %NAME%==cpu-dnnl copy "%PKG_FOLDER%\%DNNL_NAME%\THIRD-PARTY-PROGRAMS" dist\DNNL-THIRD-PARTY-PROGRAMS
IF %NAME%==cpu-dnnl 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\DNNL-LICENSE
IF %NAME%==cpu-dnnl 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\DNNL-THIRD-PARTY-PROGRAMS
IF %NAME%==onednn copy "%PKG_FOLDER%\%DNNL_NAME%\LICENSE" dist\DNNL-LICENSE
IF %NAME%==onednn copy "%PKG_FOLDER%\%DNNL_NAME%\THIRD-PARTY-PROGRAMS" dist\DNNL-THIRD-PARTY-PROGRAMS
IF %NAME%==onednn 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\DNNL-LICENSE
IF %NAME%==onednn 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\DNNL-THIRD-PARTY-PROGRAMS
IF %ONNX_DML%==true type dist\README-onnx-dml.txt |more /P > dist\README.txt
IF %ONNX_DML%==true copy "%PKG_FOLDER%\%ONNX_NAME%\LICENSE" dist\ONNX-DML-LICENSE
IF %ONNX_DML%==true copy "%PKG_FOLDER%\%ONNX_NAME%\ThirdPartyNotices.txt" dist\ONNX-DML-ThirdPartyNotices.txt
IF %ONNX_DML%==true 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip "%PKG_FOLDER%\%ONNX_NAME%\lib\onnxruntime.dll"
IF %ONNX_DML%==true 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\README.txt
IF %ONNX_DML%==true 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\ONNX-DML-LICENSE
IF %ONNX_DML%==true 7z a lc0-%APPVEYOR_REPO_TAG_NAME%-windows-%NAME%.zip .\dist\ONNX-DML-ThirdPartyNotices.txt
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
