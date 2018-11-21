rd /s build

rem set MSBuild="C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe"
set MSBuild="C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\MSBuild.exe"
rem call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
meson.py build --backend vs2017 --buildtype release ^
-Dcuda_include="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include" ^
-Dcudnn_libdirs="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64","C:\dev\cuDNN\cuda\lib\x64" ^
-Dcudnn_include="C:\dev\cuDNN\cuda\include" ^
-Ddefault_library=static

pause


cd build

%MSBuild%  ^
/p:Configuration=Release ^
/p:Platform=x64 ^
/p:PreferredToolArchitecture=x64 lc0.sln ^
/filelogger

