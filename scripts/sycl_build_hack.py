#!/usr/bin/env python3

import os

dir = os.getenv('MESON_BUILD_ROOT')

with open(dir + '/build.ninja', 'r') as file:
  lines = file.readlines()

updated = []
dep_flag = False
link_flag = False

for line in lines:
  # Replace xilink with icx -fsycl as the linker.
  if not link_flag:
    link_flag = 'xilink.exe' in line
  if link_flag:
    line = line.replace('xilink.exe', 'icx')
    line = line.replace('/MACHINE:x64', '-fsycl')
    line = line.replace('/OUT:', '-o ')
    line = line.replace('/SUBSYSTEM:CONSOLE', '')
    line = line.replace('/OPT:REF', '')
    line = line.replace('/PDB:', '/Fd')
  # Replace msvc compatible dependencies with gcc ones as icx output with /showincludes includes
  # temporary header files causing full project rebuilds.
  if line.startswith('rule') or line.startswith('build'):
    dep_flag = 'cpp_COMPILER' in line
  if dep_flag:
    line = line.replace('deps = msvc', 'deps = gcc\n depfile = $out.d')
    line = line.replace('/showIncludes', '/QMD')
    if 'icx' in line:
      line = line.replace('/Fo$out', '/Fo$out /QMF$out.d')
  updated.append(line)

with open(dir + '/build.ninja', 'w') as file:
  file.writelines(updated)
