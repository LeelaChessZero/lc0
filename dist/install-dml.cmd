@echo off
where /q tar
if errorlevel 1 goto error

cd /d %~dp0

cls
echo Installing the DirectML.dll version required by the Lc0 onnx-dml backend.
curl -# --ssl-no-revoke -o tmp_directml.zip https://globalcdn.nuget.org/packages/microsoft.ai.directml.1.10.0.nupkg"
if errorlevel 1 goto error

tar -xzOf tmp_directml.zip bin/x64-win/DirectML.dll >DirectML.dll
if errorlevel 1 goto error

del /q tmp_directml.zip
echo Installation successful.
pause
exit /b

:error
cls
echo Installation failed - see the README for an alternative approach.
pause

