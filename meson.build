# This file is part of Leela Chess Zero.
# Copyright (C) 2018 The LCZero Authors
#
# Leela Chess is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Leela Chess is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

project('lc0', 'cpp',
        default_options : ['cpp_std=c++14', 'b_ndebug=if-release'],
        meson_version: '>=0.45')

cc = meson.get_compiler('cpp')
if cc.get_id() == 'clang'
  # Thread safety annotation
  add_project_arguments('-Wthread-safety', language : 'cpp')
endif
if cc.get_id() == 'clang' or cc.get_id() == 'gcc'
  add_project_arguments('-Wextra', language : 'cpp')
  add_project_arguments('-pedantic', language : 'cpp')

  if get_option('buildtype') == 'release'
    add_project_arguments('-march=native', language : 'cpp')
    add_project_arguments('-mtune=native', language : 'cpp')
  endif
endif


# Files to compile.
deps = []
files = []
includes = []
has_backends = false


#############################################################################
## Main files
#############################################################################
files += [
  'src/engine.cc',
  'src/analyzer/analyzer.cc',
  'src/analyzer/table.cc',
  'src/chess/bitboard.cc',
  'src/chess/board.cc',
  'src/chess/position.cc',
  'src/chess/uciloop.cc',
  'src/mcts/node.cc',
  'src/mcts/search.cc',
  'src/neural/cache.cc',
  'src/neural/encoder.cc',
  'src/neural/factory.cc',
  'src/neural/loader.cc',
  'src/neural/network_check.cc',
  'src/neural/network_mux.cc',
  'src/neural/network_random.cc',
  'src/neural/network_st_batch.cc',
  'src/neural/writer.cc',
  'src/selfplay/game.cc',
  'src/selfplay/loop.cc',
  'src/selfplay/tournament.cc',
  'src/utils/commandline.cc',
  'src/utils/optionsdict.cc',
  'src/utils/optionsparser.cc',
  'src/utils/random.cc',
  'src/utils/string.cc',
  'src/utils/transpose.cc',
]
includes += include_directories('src')


#############################################################################
## Platform specific files
############################################################################
if host_machine.system() == 'windows'
  files += 'src/utils/filesystem.win32.cc'
else
  files += 'src/utils/filesystem.posix.cc'
  deps += [
    cc.find_library('pthread'),
    ]
endif



#############################################################################
## BACKENDS
#############################################################################

if get_option('build_backends')
  ## ~~~~~~~~~~
  ## Tensorflow
  ## ~~~~~~~~~~
  # Installed from https://github.com/FloopCZ/tensorflow_cc
  tensorflow_include = get_option('tensorflow_include')
  tensorflow_libdir = get_option('tensorflow_libdir')
  tf_dl_lib = cc.find_library('dl', required: false)
  tf_tensorflow_lib = cc.find_library('libtensorflow_cc',
            dirs: tensorflow_libdir, required: false)
  tf_protobuf_lib = cc.find_library('libprotobuf',
            dirs: tensorflow_libdir, required: false)

  if get_option('tf') and tf_dl_lib.found() and tf_tensorflow_lib.found() and tf_protobuf_lib.found()
    includes += include_directories(
      tensorflow_include,
      tensorflow_include[0] + '/bazel-genfiles',
      tensorflow_include[0] + '/tensorflow/contrib/makefile/downloads',
      tensorflow_include[0] + '/tensorflow/contrib/makefile/downloads/eigen',
      tensorflow_include[0] + '/tensorflow/contrib/makefile/downloads/gemmlowp',
      tensorflow_include[0] + '/tensorflow/contrib/makefile/downloads/nsync/public',
      tensorflow_include[0] + '/tensorflow/contrib/makefile/gen/protobuf-host/include',
      is_system: true
    )
    deps += [tf_dl_lib, tf_tensorflow_lib, tf_protobuf_lib]
    files += 'src/neural/network_tf.cc'
    has_backends = true
  endif


  ## ~~~~~
  ## Blas
  ## ~~~~~

  has_blas = false

  accelerate_lib = dependency('Accelerate', required: false)

  mkl_libdirs = get_option('mkl_libdirs')
  mkl_lib = cc.find_library('mkl_rt', dirs: mkl_libdirs, required: false)

  openblas_lib = cc.find_library('openblas', required: false)

  if mkl_lib.found() 
    add_project_arguments('-DUSE_MKL', language : 'cpp')
    includes += include_directories(get_option('mkl_include'))
    deps += [ mkl_lib ]
    has_blas = true

  elif accelerate_lib.found()
    includes += include_directories('/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers')
    deps += [ accelerate_lib ]
    has_blas = true

  elif openblas_lib.found() 
    add_project_arguments('-DUSE_OPENBLAS', language : 'cpp')
    includes += include_directories(get_option('openblas_include'))
    deps += [ openblas_lib ]
    has_blas = true

  endif

  if get_option('blas') and has_blas

    blas_files = [
    'src/neural/BLAS/transforms.cc',
    'src/neural/network_blas.cc'
    ]

    files += blas_files
    has_backends = true

  endif


  ## ~~~~~
  ## OpenCL
  ## ~~~~~

  has_opencl = false

  opencl_libdirs = get_option('opencl_libdirs')
  opencl_lib=cc.find_library('OpenCL', dirs: opencl_libdirs, required: false)

  opencl_framework=dependency('OpenCL', required: false)
  if opencl_framework.found()
      deps += [ opencl_framework ]
      has_opencl = true

  elif opencl_lib.found()

      deps += [ opencl_lib ]
      has_opencl = true

  endif
  

  if get_option('opencl') and has_opencl and has_blas

    opencl_files = [
    'src/neural/CL/OpenCL.cc',
    'src/neural/CL/OpenCLTuner.cc',
    'src/neural/network_opencl.cc',
    ]

    if not get_option('blas')
      opencl_files += 'src/neural/transforms.cc'
    endif

    files += opencl_files
    has_backends = true

  endif


  ## ~~~~~
  ## cuDNN
  ## ~~~~~
  cudnn_libdirs = get_option('cudnn_libdirs')
  cu_blas = cc.find_library('cublas', dirs: cudnn_libdirs, required: false)
  cu_dnn = cc.find_library('cudnn', dirs: cudnn_libdirs, required: false)
  cu_dart = cc.find_library('cudart', dirs: cudnn_libdirs, required: false)
  nvcc = find_program('/usr/local/cuda-9.2/bin/nvcc',
                      '/usr/local/cuda-9.1/bin/nvcc',
                      'nvcc', required: false)

  cuda_files = [
    'src/neural/network_cudnn.cu',
  ]

  if get_option('cudnn') and cu_blas.found() and cu_dnn.found() and cu_dart.found() and nvcc.found()
    includes += include_directories(get_option('cudnn_include'))
    deps += [cu_blas, cu_dnn, cu_dart]
  cuda_arguments = ['-c', '@INPUT@', '-o', '@OUTPUT@',
                    '-I', meson.current_source_dir() + '/src']
  if host_machine.system() == 'windows'
    cuda_arguments += ['-Xcompiler', '-MD']
  else
    cuda_arguments += ['--std=c++14', '-Xcompiler', '-fPIC']
  endif
  foreach x : get_option('cudnn_include')
     cuda_arguments += ['-I', x]
  endforeach
  if host_machine.system() == 'windows'
    outputname = '@BASENAME@.obj'
  else
    outputname = '@BASENAME@.o'
  endif
  cuda_gen = generator(nvcc,
      output: outputname,
      arguments: cuda_arguments,
  )
    files += cuda_gen.process(cuda_files)
    has_backends = true
  endif

endif # if get_option('build_backends')

if not has_backends and get_option('build_backends')
  error('''

        No usable computation backends (cudnn/tensorflow/etc) are found. 
        If you want to build it with random only backend, pass
        -D build_backends=false to a meson build.''')
endif


#############################################################################
## Dependencies
#############################################################################
  ## ~~~~
  ## zlib
  ## ~~~~
  # Pick latest from https://wrapdb.mesonbuild.com/zlib and put into
  # subprojects/zlib.wrap
  deps += subproject('zlib').get_variable('zlib_dep')

  ## ~~~~~~~~
  ## Profiler
  ## ~~~~~~~~
  if get_option('buildtype') != 'release'
    deps += cc.find_library('libprofiler',
      dirs: ['/usr/local/lib'], required: false)
  endif


#############################################################################
## Main Executable
#############################################################################

executable('lc0', 'src/main.cc',
  files, include_directories: includes, dependencies: deps, install: true)


### Tests
gtest = dependency('gtest', required: false)

test_deps = deps
if gtest.found()
  test_deps += gtest

  test('ChessBoard',
    executable('chessboard_test', 'src/chess/board_test.cc',
    files, include_directories: includes, dependencies: test_deps
  ))

  test('HashCat',
    executable('hashcat_test', 'src/utils/hashcat_test.cc',
    files, include_directories: includes, dependencies: test_deps
  ))
endif
