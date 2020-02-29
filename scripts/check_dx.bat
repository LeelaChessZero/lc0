@ECHO OFF
ECHO Sanity checking the dx12 driver.
lc0 benchmark --starting-position --backend=check --backend-opts=mode=check,freq=1.0,atol=5e-1,dx12,blas %*
PAUSE

