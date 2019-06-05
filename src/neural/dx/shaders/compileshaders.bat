REM Need to run this batch file after making changes in any of the shaders.
REM The batch files generates/updates "shaders.h" (DXIL asm code for compiled HLSL).
REM TODO: maybe get rid of this and do this as part of build process?

del shaders.h

dxc /Tcs_5_0 /EExpandPlanes_kernel_Fp16_NHWC /Fh temp.txt ExpandPlanes.hlsl
type temp.txt >> shaders.h
del temp.txt

dxc /Tcs_6_2 /EPolicyFC_With_Softmax_kernel /Fh temp.txt PolicyFC.hlsl  -enable-16bit-types
type temp.txt >> shaders.h
del temp.txt


dxc /Tcs_6_2 /EPolicyFC /Fh temp.txt PolicyFC.hlsl -enable-16bit-types
type temp.txt >> shaders.h
del temp.txt

dxc /Tcs_6_2 /EPolicySoftmax /Fh temp.txt PolicyFC.hlsl -enable-16bit-types
type temp.txt >> shaders.h
del temp.txt

dxc /Tcs_5_0 /EValueFC1 /Fh temp.txt ValueFC1.hlsl
type temp.txt >> shaders.h
del temp.txt

dxc /Tcs_5_0 /EValueFC2 /Fh temp.txt ValueFC2.hlsl
type temp.txt >> shaders.h
del temp.txt

dxc /Tcs_5_0 /ESkipAdd /Fh temp.txt SkipAdd.hlsl
type temp.txt >> shaders.h
del temp.txt

pause