rd /s build

rem set MSBuild="C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe"
set MSBuild="C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\MSBuild.exe"

rem call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64

rem change to '-Dblas=true' to also build the blas backend with mkl
meson build --backend vs2017 --buildtype release -Dblas=false ^
-Dmkl_include="C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl\include" ^
-Dmkl_libdirs="C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl\lib\intel64" ^
-Dopencl_libdirs="C:\Program Files (x86)\AMD APP SDK\3.0\lib\x86_64" ^
-Dopencl_include="C:\Program Files (x86)\AMD APP SDK\3.0\include" ^
-Ddefault_library=static

pause

cd build

%MSBuild% /p:Configuration=Release /p:Platform=x64 ^
/p:PreferredToolArchitecture=x64 "subprojects\zlib-1.2.11\Windows resource for file 'win32_zlib1.rc'@cus.vcxproj" ^
/filelogger

%MSBuild% /p:Configuration=Release /p:Platform=x64 ^
/p:PreferredToolArchitecture=x64 subprojects\zlib-1.2.11\subprojects@zlib-1.2.11@@z@sta.vcxproj ^
/filelogger

%MSBuild% /p:Configuration=Release /p:Platform=x64 ^
/p:PreferredToolArchitecture=x64 lc0@exe.vcxproj ^
/filelogger
