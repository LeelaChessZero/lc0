del shaders.h

dxc /Tcs_6_2 /EExpandPlanes_shader_fp32 /Fh temp.txt ExpandPlanes.hlsl
type temp.txt >> shaders.h
del temp.txt

dxc /Tcs_6_2 /EExpandPlanes_shader_fp16 /Fh temp.txt ExpandPlanes.hlsl -enable-16bit-types
type temp.txt >> shaders.h
del temp.txt

dxc /Tcs_6_2 /Einput_transform_shader_fp32 /Fh temp.txt WinogradTransform.hlsl
type temp.txt >> shaders.h
del temp.txt

dxc /Tcs_6_2 /Eoutput_transform_shader_fp32 /Fh temp.txt WinogradTransform.hlsl
type temp.txt >> shaders.h
del temp.txt


dxc /Tcs_6_2 /Econv_1x1_shader_fp32 /Fh temp.txt Conv1x1.hlsl
type temp.txt >> shaders.h
del temp.txt

dxc /Tcs_6_2 /Eadd_vectors_shader /Fh temp.txt AddVectors.hlsl
type temp.txt >> shaders.h
del temp.txt

dxc /Tcs_6_2 /EPolicyMapShader -Vn g_policy_map_shader_fp32  /Fh temp.txt PolicyMap.hlsl
type temp.txt >> shaders.h
del temp.txt
