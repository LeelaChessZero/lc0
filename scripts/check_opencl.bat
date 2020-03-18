@ECHO OFF
ECHO Sanity checking the opencl driver.
lc0 benchmark --backend=check --backend-opts=mode=check,freq=1.0,opencl,blas %*
PAUSE

