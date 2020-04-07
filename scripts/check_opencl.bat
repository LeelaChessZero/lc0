@ECHO OFF
ECHO Sanity checking the opencl driver.
lc0 benchmark --num-positions=1 --backend=check --backend-opts=mode=check,freq=1.0,opencl %*
PAUSE

