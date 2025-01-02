project('gaviotatb', 'c')

gaviotatb_src = [
  'gtb-probe.c',
  'gtb-dec.c',
  'gtb-att.c',
  'sysport/sysport.c',
  'compression/wrap.c',
  'compression/huffman/hzip.c',
  'compression/liblzf/lzf_c.c',
  'compression/liblzf/lzf_d.c',
  'compression/lzma/LzmaEnc.c',
  'compression/lzma/LzmaDec.c',
  'compression/lzma/Alloc.c',
  'compression/lzma/LzFind.c',
  'compression/lzma/Lzma86Enc.c',
  'compression/lzma/Lzma86Dec.c',
  'compression/lzma/Bra86.c'
]

gaviotatb_includes = [
  '.',
  'sysport',
  'compression',
  'compression/huffman',
  'compression/liblzf',
  'compression/lzma'
]

gaviota_lib = static_library('gaviota',
  gaviotatb_src,
  c_args : '-Dz_uLong=uLong',
  include_directories : gaviotatb_includes)

incdir = include_directories('.')

gaviotatb_dep = declare_dependency(
  link_with : gaviota_lib,
  include_directories : gaviotatb_includes)
