@ECHO OFF
REM Starts lc0 on the Level Zero v2 adapter of the SYCL runtime, see the SYCL
REM section of the README. Keep this file next to lc0.exe and use it in place
REM of lc0.exe; all arguments are passed on.
SETLOCAL
SET SYCL_UR_USE_LEVEL_ZERO_V2=1
"%~dp0lc0.exe" %*
