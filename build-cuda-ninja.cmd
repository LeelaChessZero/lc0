rd /s build

call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
meson.py build --buildtype release ^
-Dcudnn_libdirs="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64","C:\dev\cuDNN\cuda\lib\x64" ^
-Dcudnn_include="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include","C:\dev\cuDNN\cuda\include" ^
-Ddefault_library=static

pause


cd build

ninja