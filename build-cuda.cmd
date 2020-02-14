rd /s build

rem set MSBuild="C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe"
rem set MSBuild="C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\MSBuild.exe"
set MSBuild="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe"
rem call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
rem call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64

rem meson.py build --backend vs2017 --buildtype release ^
meson build --backend vs2019 --buildtype release ^
-Dcudnn_libdirs="C:\CUDA\lib\x64","C:\dev\cuDNN\cuda\lib\x64" ^
-Dcudnn_include="C:\CUDA\include","C:\dev\cuDNN\cuda\include" ^
-Ddefault_library=static ^
-Dopencl=false ^
-Dcudnn=true ^ 
-Ddx=false ^

rem -Ddx=false

rem meson.py build --backend vs2017 --buildtype release ^

rem -Dcudnn_libdirs="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64","C:\dev\cuDNN\cuda\lib\x64" ^
rem -Dcudnn_include="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include","C:\dev\cuDNN\cuda\include" ^

rem -Dprotobuf_libdir="C:\code\lc0\subprojects\protobuf-3.5.1" 

 
pause
rem -Ddefault_library=static ^


cd build

%MSBuild%  ^
/p:Configuration=Release ^
/p:Platform=x64 ^
/p:PreferredToolArchitecture=x64 lc0.sln ^
/filelogger

