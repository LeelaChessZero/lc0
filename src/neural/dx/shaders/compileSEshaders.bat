del shaders_se.h

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp32_se_128 /DBLOCK_SIZE=128 /Fh temp.txt WinogradTransformSE.hlsl
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp32_se_256 /DBLOCK_SIZE=256 /Fh temp.txt WinogradTransformSE.hlsl
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp32_se_320 /DBLOCK_SIZE=320 /Fh temp.txt WinogradTransformSE.hlsl
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp32_se_384 /DBLOCK_SIZE=384 /Fh temp.txt WinogradTransformSE.hlsl
type temp.txt >> shaders_se.h
del temp.txt


dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp32_se_512 /DBLOCK_SIZE=512 /Fh temp.txt WinogradTransformSE.hlsl
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp32_se_640 /DBLOCK_SIZE=640 /Fh temp.txt WinogradTransformSE.hlsl
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp32_se_768 /DBLOCK_SIZE=768 /Fh temp.txt WinogradTransformSE.hlsl
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp32_se_1024 /DBLOCK_SIZE=1024 /Fh temp.txt WinogradTransformSE.hlsl
type temp.txt >> shaders_se.h
del temp.txt



dxc /Tcs_6_2 /ESE -Vn g_se_128 /DBLOCK_SIZE=128 /Fh temp.txt SE.hlsl
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /ESE -Vn g_se_256 /DBLOCK_SIZE=256 /Fh temp.txt SE.hlsl
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /ESE -Vn g_se_320 /DBLOCK_SIZE=320 /Fh temp.txt SE.hlsl
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /ESE -Vn g_se_384 /DBLOCK_SIZE=384 /Fh temp.txt SE.hlsl
type temp.txt >> shaders_se.h
del temp.txt


dxc /Tcs_6_2 /ESE -Vn g_se_512 /DBLOCK_SIZE=512 /Fh temp.txt SE.hlsl
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /ESE -Vn g_se_640 /DBLOCK_SIZE=640 /Fh temp.txt SE.hlsl
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /ESE -Vn g_se_768 /DBLOCK_SIZE=768 /Fh temp.txt SE.hlsl
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /ESE -Vn g_se_1024 /DBLOCK_SIZE=1024 /Fh temp.txt SE.hlsl
type temp.txt >> shaders_se.h
del temp.txt