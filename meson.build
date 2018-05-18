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

project('lc0', 'cpp', default_options : ['cpp_std=c++14', 'b_ndebug=if-release'])

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


#############################################################################
## Main files
#############################################################################
if host_machine.system() != 'windows'
  deps += [
    cc.find_library('stdc++fs'),
    cc.find_library('pthread'),
    ]
endif
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
  'src/neural/network_mux.cc',
  'src/neural/network_random.cc',
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
## Tensorflow
#############################################################################
# Installed from https://github.com/FloopCZ/tensorflow_cc
tensorflow_include = get_option('tensorflow_include')
tensorflow_libdir = get_option('tensorflow_libdir')
tf_dl_lib = cc.find_library('dl', required: false)
tf_tensorflow_lib = cc.find_library('libtensorflow_cc',
          dirs: tensorflow_libdir, required: false)
tf_protobuf_lib = cc.find_library('libprotobuf',
          dirs: tensorflow_libdir, required: false)

if tf_dl_lib.found() and tf_tensorflow_lib.found() and tf_protobuf_lib.found()
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
endif
  

#############################################################################
## cuDNN
#############################################################################
cudnn_libdirs = get_option('cudnn_libdirs')
cu_blas = cc.find_library('cublas', dirs: cudnn_libdirs, required: false)
cu_dnn = cc.find_library('cudnn', dirs: cudnn_libdirs, required: false)
cu_dart = cc.find_library('cudart', dirs: cudnn_libdirs, required: false)
nvcc = find_program('nvcc', required: false)

cuda_files = [
  'src/neural/network_cudnn.cu',
]

if cu_blas.found() and cu_dnn.found() and cu_dart.found() and nvcc.found()
  includes += include_directories(get_option('cudnn_include'))
  deps += [cu_blas, cu_dnn, cu_dart]
cuda_arguments = ['--std=c++14', '-c', '@INPUT@',
                  '-o', '@OUTPUT@', '-Xcompiler', '-fPIC',
                  '-I', meson.current_source_dir() + '/src']
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
endif


#############################################################################
## Profiler
#############################################################################
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
