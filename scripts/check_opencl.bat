@ECHO OFF
ECHO Sanity checking the opencl driver.
lc0 benchmark --starting-position --backend=check --backend-opts=mode=check,freq=1.0,opencl,blas %*
PAUSE

