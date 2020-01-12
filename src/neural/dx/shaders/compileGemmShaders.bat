del shaders_gemm.h

dxc /Tcs_6_2 /EMatrixMul /Vn g_MatrixMul_Fp32 /Fh temp.txt Gemm.hlsl
type temp.txt >> shaders_gemm.h
del temp.txt

dxc /Tcs_6_2 /EMatrixMul /Vn g_MatrixMul_Fp16 /DUSE_FP16_MATH=1 /Fh temp.txt Gemm.hlsl -enable-16bit-types
type temp.txt >> shaders_gemm.h
del temp.txt

pause