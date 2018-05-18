rd /s build

call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
meson.py build --backend vs2015 --buildtype release ^
-Dtensorflow_libdirs= ^
-Dtensorflow_include=""

pause

cd build
"C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe" ^
/p:Configuration=Release ^
/p:Platform=x64 ^
/p:PreferredToolArchitecture=x64 lc0@exe.vcxproj ^
/filelogger

