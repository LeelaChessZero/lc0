@ECHO OFF
ECHO Setting SYCL_UR_USE_LEVEL_ZERO_V2=1 for your user account, see the SYCL section of the README.
SETX SYCL_UR_USE_LEVEL_ZERO_V2 1
IF NOT ERRORLEVEL 1 ECHO Lc0 uses the Level Zero v2 adapter from its next start. Restart your chess GUI if it is open.
PAUSE
