@echo off
setlocal

rem ============================================================================
rem Lc0 SYCL Dependency Packager for Windows
rem 
rem Description:
rem   A standalone utility to create portable Windows builds of Leela Chess Zero.
rem   It scans the lc0.exe binary for dependencies and recursively copies all
rem   required DLLs from the system/OneAPI environment into the build folder.
rem
rem Usage:
rem   1. Build Lc0.
rem   2. Run this script from the command line.
rem      scripts\package_SYCL_dependencies.cmd [optional_path_to_build_dir]
rem
rem Requirements:
rem   - Microsoft Visual Studio Tools (dumpbin.exe) must be in PATH.
rem   - The build environment (e.g., OneAPI setvars) must be active so that
rem     dependencies can be found in %PATH%.
rem ============================================================================

rem --- 1. Environment Checks ---
where dumpbin >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] 'dumpbin.exe' not found.
    echo This script requires Visual Studio build tools.
    echo Please run this from a Developer Command Prompt or initialize your environment.
    exit /b 1
)

rem --- 2. Locate Build Directory ---
set "TARGET_DIR="

rem Case A: User provided path
if not "%~1"=="" (
    if exist "%~1\lc0.exe" (
        set "TARGET_DIR=%~1"
    ) else (
        echo [ERROR] lc0.exe not found in provided directory: "%~1"
        exit /b 1
    )
)

rem Case B: Standard layout (ROOT/scripts/script.cmd -> ROOT/build/lc0.exe)
if "%TARGET_DIR%"=="" (
    if exist "%~dp0..\build\lc0.exe" (
        set "TARGET_DIR=%~dp0..\build"
    ) else if exist "%~dp0..\build\release\lc0.exe" (
        set "TARGET_DIR=%~dp0..\build\release"
    )
)

rem Case C: Run from within build directory
if "%TARGET_DIR%"=="" (
    if exist ".\lc0.exe" (
        set "TARGET_DIR=."
    )
)

if "%TARGET_DIR%"=="" (
    echo [ERROR] Could not locate 'build' directory containing lc0.exe.
    echo Please provide the path as an argument.
    echo Usage: %~nx0 [path_to_build_folder]
    exit /b 1
)

rem Normalize path
pushd "%TARGET_DIR%"
echo [INFO] Targeting build directory: %CD%

rem --- 3. Packaging Logic (Ported from build-sycl-standalone.cmd) ---

echo.
echo ==========================================
echo  Post-Build: Auto-Packaging Dependencies
echo ==========================================

rem --- 1. Clean old DLLs to ensure a fresh, minimal package ---
rem We are currently inside the 'build' folder.
del /q *.dll 2>nul

rem --- 2. Define temp script path ---
set "PS_SCRIPT=%CD%\_lc0_packager.ps1"
if exist "%PS_SCRIPT%" del "%PS_SCRIPT%"

rem --- 3. Generate PowerShell Script Line-by-Line (Safest method for Batch) ---
echo $exe = "lc0.exe" >> "%PS_SCRIPT%"
echo $dest = "." >> "%PS_SCRIPT%"
echo. >> "%PS_SCRIPT%"
echo function Get-Deps($path) { >> "%PS_SCRIPT%"
echo     $out = cmd /c "dumpbin /dependents `"$path`" 2>nul" >> "%PS_SCRIPT%"
echo     return $out ^| Select-String -Pattern "[a-zA-Z0-9_\-\.]+\.dll" ^| ForEach-Object { $_.Matches.Value.Trim() } >> "%PS_SCRIPT%"
echo } >> "%PS_SCRIPT%"
echo. >> "%PS_SCRIPT%"
echo $queue = new-object System.Collections.Generic.Queue[string] >> "%PS_SCRIPT%"
echo $processed = new-object System.Collections.Generic.HashSet[string] >> "%PS_SCRIPT%"
echo. >> "%PS_SCRIPT%"
echo Write-Host "Scanning $exe for dependencies..." -ForegroundColor Cyan >> "%PS_SCRIPT%"
echo Get-Deps $exe ^| ForEach-Object { $queue.Enqueue($_) } >> "%PS_SCRIPT%"
echo. >> "%PS_SCRIPT%"
echo while ($queue.Count -gt 0) { >> "%PS_SCRIPT%"
echo     $targetDll = $queue.Dequeue() >> "%PS_SCRIPT%"
echo     if ($processed.Contains($targetDll)) { continue } >> "%PS_SCRIPT%"
echo     $processed.Add($targetDll) ^| Out-Null >> "%PS_SCRIPT%"
echo     if ($targetDll -match "^(api-ms-win|ext-ms-|kernel32|user32|gdi32|winspool|shell32|ole32|oleaut32|uuid|comdlg32|advapi32|msvcrt|ucrtbase|ws2_32|rpcrt4|shlwapi|version|imm32|crypt32|bcrypt|wldap32|ntdll|d3d12|dxgi|opengl32|glu32)") { continue } >> "%PS_SCRIPT%"
echo. >> "%PS_SCRIPT%"
echo     $foundPath = "" >> "%PS_SCRIPT%"
echo     foreach ($p in $env:PATH.Split(';')) { >> "%PS_SCRIPT%"
echo         $candidate = Join-Path $p $targetDll >> "%PS_SCRIPT%"
echo         if (Test-Path $candidate) { $foundPath = $candidate; break } >> "%PS_SCRIPT%"
echo     } >> "%PS_SCRIPT%"
echo. >> "%PS_SCRIPT%"
echo     if ($foundPath) { >> "%PS_SCRIPT%"
echo         Write-Host "Packaging: $targetDll" -ForegroundColor Green >> "%PS_SCRIPT%"
echo         Copy-Item $foundPath $dest -Force >> "%PS_SCRIPT%"
echo. >> "%PS_SCRIPT%"
echo         Get-Deps $foundPath ^| ForEach-Object { >> "%PS_SCRIPT%"
echo             if (!$processed.Contains($_)) { $queue.Enqueue($_) } >> "%PS_SCRIPT%"
echo         } >> "%PS_SCRIPT%"
echo. >> "%PS_SCRIPT%"
echo         if ($targetDll -match "sycl[0-9]+\.dll") { >> "%PS_SCRIPT%"
echo             $syclDir = Split-Path $foundPath -Parent >> "%PS_SCRIPT%"
echo             $patterns = @("pi_*.dll", "ur_*.dll") >> "%PS_SCRIPT%"
echo             foreach ($pat in $patterns) { >> "%PS_SCRIPT%"
echo                 Get-ChildItem -Path $syclDir -Filter $pat ^| ForEach-Object { >> "%PS_SCRIPT%"
echo                     if (!$processed.Contains($_.Name)) { >> "%PS_SCRIPT%"
echo                         Write-Host "  + Found Indirect Dependency: $($_.Name)" -ForegroundColor Yellow >> "%PS_SCRIPT%"
echo                         Copy-Item $_.FullName $dest -Force >> "%PS_SCRIPT%"
echo                         $processed.Add($_.Name) ^| Out-Null >> "%PS_SCRIPT%"
echo                         Get-Deps $_.FullName ^| ForEach-Object { >> "%PS_SCRIPT%"
echo                             if (!$processed.Contains($_)) { $queue.Enqueue($_) } >> "%PS_SCRIPT%"
echo                         } >> "%PS_SCRIPT%"
echo                     } >> "%PS_SCRIPT%"
echo                 } >> "%PS_SCRIPT%"
echo             } >> "%PS_SCRIPT%"
echo         } >> "%PS_SCRIPT%"
echo     } >> "%PS_SCRIPT%"
echo } >> "%PS_SCRIPT%"

rem --- 4. Run and Cleanup ---
powershell -ExecutionPolicy Bypass -File "%PS_SCRIPT%"
if exist "%PS_SCRIPT%" del "%PS_SCRIPT%"

echo.
echo ==========================================
echo  Packaging Complete!
echo ==========================================

popd
exit /b 0