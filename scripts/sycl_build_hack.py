#!/usr/bin/env python3

import os

dir = os.getenv('MESON_BUILD_ROOT')

with open(dir + '/build.ninja', 'r') as file:
  lines = file.readlines()

updated = []
dep_flag = False
link_flag = False

for line in lines:
  # Replace the MSVC linker with icx: plain link.exe cannot merge SYCL
  # device code across translation units, so a link that "succeeds" this
  # way silently produces a binary where any kernel is liable to go
  # missing at runtime ("No kernel named ... was found"). Older meson/
  # oneAPI combos emitted `xilink.exe` here; current meson (>=1.11) just
  # emits the plain `command = "link"` linker invocation, so match both.
  if not link_flag:
    link_flag = ('xilink.exe' in line) or ('command = "link"' in line)
  if link_flag:
    line = line.replace('xilink.exe', 'icx')
    line = line.replace('command = "link"', 'command = "icx"')
    line = line.replace('/MACHINE:x64', '')
    line = line.replace('/OUT:', '-o ')
    line = line.replace('/SUBSYSTEM:CONSOLE', '')
    line = line.replace('/OPT:REF', '')
    line = line.replace('/PDB:', '/Fd')
    # icx accepts these fine on a plain command line (as an unknown-argument
    # warning, via its clang-cl compatibility layer), but response-file
    # (@file.rsp) parsing hits a different code path where an unrecognized
    # `/xxx` token is treated as a positional input file instead of a flag,
    # turning it into a hard "no such file or directory" link failure. Since
    # meson switches to a response file once the command line gets long
    # enough, translate these to spellings icx understands unconditionally.
    line = line.replace('/LIBPATH:', '-L')
    line = line.replace('/release', '')
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
