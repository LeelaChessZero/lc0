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

gaviotatb_dep = declare_dependency(
  sources: gaviotatb_src,
  include_directories: gaviotatb_includes,
  compile_args:'-Dz_uLong=uLong'
)
