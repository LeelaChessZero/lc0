# This file is part of Leela Chess Zero.
# Copyright (C) 2018-2019 The LCZero Authors
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
  endif
endif

# Files to compile.
deps = []
files = []
includes = []
has_backends = false

# Third party files.
includes += include_directories('third_party', is_system: true)

# Both protobuf and protoc must be the same version, so couple them together.
protobuf_lib = cc.find_library('libprotobuf', dirs : get_option('protobuf_libdir'), required : false)
if not protobuf_lib.found()
  protobuf_dep = dependency('protobuf', required : false)
else
  protobuf_dep = protobuf_lib
endif
protoc = find_program('protoc', required : false)
# For tensorflow skip system protobuf, chances are it will not work.
if get_option('protobuf-3-6-0')
  deps += subproject('protobuf-3.6.0').get_variable('protobuf_dep')
  protoc = subproject('protobuf-3.6.0').get_variable('protoc')
elif not protobuf_dep.found() or not protoc.found() or get_option('tensorflow')
  deps += subproject('protobuf').get_variable('protobuf_dep')
  protoc = subproject('protobuf').get_variable('protoc')
else
  deps += protobuf_dep
  if protobuf_lib.found()
    includes += include_directories(get_option('protobuf_include'))
  endif
endif

gen = generator(protoc, output: ['@BASENAME@.pb.cc', '@BASENAME@.pb.h'],
  arguments : ['--proto_path=@CURRENT_SOURCE_DIR@/libs/lczero-common', '--cpp_out=@BUILD_DIR@', '@INPUT@'])

if run_command('checkdir.py', 'libs/lczero-common/proto').returncode() != 0
  if run_command('git', 'status').returncode() == 0
    message('updating git submodule libs/lczero-common')
    run_command('git', 'submodule', 'update', '--init', '--recursive')
  else
    message('cloning lczero-common.git into libs/lczero-common')
    run_command('git', 'clone', '--depth=1',
                'https://github.com/LeelaChessZero/lczero-common.git',
                'libs/lczero-common/')
  endif
endif

files += gen.process('libs/lczero-common/proto/net.proto',
  preserve_path_from : meson.current_source_dir() + '/libs/lczero-common/')

#############################################################################
## Main files
#############################################################################
files += [
  'src/engine.cc',
  'src/version.cc',
  'src/benchmark/benchmark.cc',
  'src/chess/bitboard.cc',
  'src/chess/board.cc',
  'src/chess/position.cc',
  'src/chess/uciloop.cc',
  'src/mcts/node.cc',
  'src/mcts/params.cc',
  'src/mcts/search.cc',
  'src/neural/cache.cc',
  'src/neural/encoder.cc',
  'src/neural/factory.cc',
  'src/neural/loader.cc',
  'src/neural/network_check.cc',
  'src/neural/network_demux.cc',
  'src/neural/network_legacy.cc',
  'src/neural/network_mux.cc',
  'src/neural/network_random.cc',
  'src/neural/network_rr.cc',
  'src/neural/network_st_batch.cc',
  'src/neural/writer.cc',
  'src/selfplay/game.cc',
  'src/selfplay/loop.cc',
  'src/selfplay/tournament.cc',
  'src/syzygy/syzygy.cc',
  'src/utils/commandline.cc',
  'src/utils/configfile.cc',
  'src/utils/histogram.cc',
  'src/utils/logging.cc',
  'src/utils/optionsdict.cc',
  'src/utils/optionsparser.cc',
  'src/utils/random.cc',
  'src/utils/string.cc',
  'src/utils/transpose.cc',
  'src/utils/weights_adapter.cc',
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

  if get_option('tensorflow') and tf_dl_lib.found() and tf_tensorflow_lib.found()
    includes += include_directories(
      tensorflow_include,
      tensorflow_include[0] + '/bazel-genfiles',
      tensorflow_include[0] + '/tensorflow/contrib/makefile/downloads',
      tensorflow_include[0] + '/tensorflow/contrib/makefile/downloads/absl',
      tensorflow_include[0] + '/tensorflow/contrib/makefile/downloads/eigen',
      tensorflow_include[0] + '/tensorflow/contrib/makefile/downloads/gemmlowp',
      tensorflow_include[0] + '/tensorflow/contrib/makefile/downloads/nsync/public',
      tensorflow_include[0] + '/tensorflow/contrib/makefile/gen/protobuf-host/include',
      is_system: true
    )
    deps += [tf_dl_lib, tf_tensorflow_lib]
    files += 'src/neural/network_tf.cc'
    has_backends = true
  endif


  ## ~~~~~
  ## Blas
  ## ~~~~~

  shared_files = []

  has_blas = false

  accelerate_lib = dependency('Accelerate', required: false)

  mkl_libdirs = get_option('mkl_libdirs')
  mkl_lib = cc.find_library('mkl_rt', dirs: mkl_libdirs, required: false)
  if not mkl_lib.found()
    mkl_lib = cc.find_library('mklml', dirs: mkl_libdirs, required: false)
  endif

  openblas_libdirs = get_option('openblas_libdirs')
  openblas_lib = cc.find_library('openblas.dll', dirs: openblas_libdirs, required: false)
  if not openblas_lib.found()
    openblas_lib = cc.find_library('openblas', dirs: openblas_libdirs, required: false)
  endif

  if get_option('blas')
    if get_option('mkl') and mkl_lib.found()
      add_project_arguments('-DUSE_MKL', language : 'cpp')
      includes += include_directories(get_option('mkl_include'))
      deps += [ mkl_lib ]
      has_blas = true

    elif get_option('accelerate') and accelerate_lib.found()
      includes += include_directories('/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers')
      deps += [ accelerate_lib ]
      has_blas = true

    elif get_option('openblas') and openblas_lib.found()
      add_project_arguments('-DUSE_OPENBLAS', language : 'cpp')

      required_openblas_header = 'openblas_config.h'
      if not cc.has_header(required_openblas_header)
        openblas_headers_found = false

        # add the first valid include directory
        foreach d : get_option('openblas_include')
          if not openblas_headers_found and cc.has_header(required_openblas_header, args: '-I' + d)
            includes += include_directories(d)
            openblas_headers_found = true
          endif
        endforeach

        if not openblas_headers_found
          error('Failed to detect OpenBLAS headers. Did you install libopenblas-dev?')
        endif
      endif

      deps += [ openblas_lib ]
      has_blas = true
    endif

    ispc = find_program('ispc', required: false)
    ispc_extra_args = []
    if has_blas and get_option('ispc') and ispc.found()
      ispc_native_only = get_option('ispc_native_only')
      if host_machine.system() == 'windows'
        outputnames = [ '@BASENAME@.obj']
        if not ispc_native_only
          outputnames += ['@BASENAME@_sse2.obj', '@BASENAME@_sse4.obj',
                          '@BASENAME@_avx.obj', '@BASENAME@_avx11.obj',
                          '@BASENAME@_avx2.obj' ]
        endif
      else
        ispc_extra_args += ['--pic']
        outputnames = [ '@BASENAME@.o']
        if not ispc_native_only
          outputnames += ['@BASENAME@_sse2.o', '@BASENAME@_sse4.o', 
                           '@BASENAME@_avx.o', '@BASENAME@_avx11.o',
                           '@BASENAME@_avx2.o' ]
        endif
      endif
      if ispc_native_only
        ispc_target = 'host'
      else
        ispc_target = 'sse2-i32x8,sse4-i32x8,avx1-i32x8,avx1.1-i32x8,avx2-i32x8'
      endif
      iscp_gen = generator(ispc,
        output: [ '@BASENAME@_ispc.h', outputnames ],
        arguments: [ '-O2', '--wno-perf', '--arch=x86-64',
                     '--target=' + ispc_target,
                     '@INPUT@', '-o', '@OUTPUT1@' ,'-h', '@OUTPUT0@' ]
                     + ispc_extra_args
      )
    endif

  endif

  if get_option('blas') and has_blas

    blas_files = [
    'src/neural/blas/convolution1.cc',
    'src/neural/blas/fully_connected_layer.cc',
    'src/neural/blas/se_unit.cc',
    'src/neural/blas/network_blas.cc',
    'src/neural/blas/winograd_convolution3.cc'
    ]

    shared_files = [
    'src/neural/shared/activation.cc',
    'src/neural/shared/winograd_filter.cc',
    ]

    files += blas_files
    has_backends = true

    if get_option('ispc') and ispc.found()
      files += iscp_gen.process('src/neural/blas/winograd_transform.ispc')
      add_project_arguments('-DUSE_ISPC', language : 'cpp')
    endif

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

  if get_option('opencl') and has_opencl

    opencl_files = [
      'src/neural/opencl/network_opencl.cc',
      'src/neural/opencl/OpenCL.cc',
      'src/neural/opencl/OpenCLTuner.cc',
      'src/neural/opencl/OpenCLBuffers.cc',
    ]

    shared_files = [
    'src/neural/shared/activation.cc',
    'src/neural/shared/winograd_filter.cc',
    ]

    includes += include_directories(get_option('opencl_include'))
    files += opencl_files
    has_backends = true

  endif

  files += shared_files

  ## ~~~~~
  ## cuDNN
  ## ~~~~~
  cudnn_libdirs = get_option('cudnn_libdirs')
  cu_blas = cc.find_library('cublas', dirs: cudnn_libdirs, required: false)
  cu_dnn = cc.find_library('cudnn', dirs: cudnn_libdirs, required: false)
  cu_dart = cc.find_library('cudart', dirs: cudnn_libdirs, required: false)
  nvcc = find_program('nvcc', '/usr/local/cuda/bin/nvcc', '/opt/cuda/bin/nvcc',
                      required: false)

  #.cc files
  cuda_files = [
    'src/neural/cuda/network_cudnn.cc',
	'src/neural/cuda/layers.cc',
  ]

  # .cu files compiled without any specific sm version
  cuda_files_nvcc_common = [
    'src/neural/cuda/common_kernels.cu',
  ]

  # .cu files compiled with sm_70 (volta+). Only used by cudnn-fp16
  cuda_files_nvcc_fp16 = [
    'src/neural/cuda/fp16_kernels.cu',
  ]

  if get_option('cudnn') and cu_blas.found() and cu_dnn.found() and cu_dart.found() and nvcc.found()
    deps += [cu_blas, cu_dnn, cu_dart]
    foreach d : get_option('cudnn_include')
      if run_command('checkdir.py', d).returncode() == 0
        includes += include_directories(d)
      endif
    endforeach
    includes += include_directories('src/neural/cuda/')

    cuda_arguments = ['@EXTRA_ARGS@', '-c', '@INPUT@', '-o', '@OUTPUT@',
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
	 files += cuda_files
    files += cuda_gen.process(cuda_files_nvcc_common)
    files += cuda_gen.process(cuda_files_nvcc_fp16, extra_args: ['-arch=compute_70', '-code=sm_70'])
    has_backends = true
  endif

endif # if get_option('build_backends')

if not has_backends and get_option('build_backends')
  error('''

        No usable computation backends (cudnn/opencl/blas/etc) enabled.
        If you want to build with the random backend only, add
        -Dbuild_backends=false to the build command line.''')
endif


#############################################################################
## Dependencies
#############################################################################
  ## ~~~~
  ## zlib
  ## ~~~~
  # Pick latest from https://wrapdb.mesonbuild.com/zlib and put into
  # subprojects/zlib.wrap
  deps += dependency('zlib', fallback: ['zlib', 'zlib_dep'])

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

if not get_option('popcnt')
  add_project_arguments('-DNO_POPCNT', language : 'cpp')
endif

if not get_option('pext')
  add_project_arguments('-DNO_PEXT', language : 'cpp')
endif

executable('lc0', 'src/main.cc',
  files, include_directories: includes, dependencies: deps, install: true)


### Tests
gtest = dependency('gtest', fallback: ['gtest', 'gtest_dep'], required: false)

if get_option('gtest') and gtest.found()
  lc0_lib = library('lc0_lib', files, include_directories: includes, dependencies: deps)

  test('ChessBoard',
    executable('chessboard_test', 'src/chess/board_test.cc',
    include_directories: includes, link_with: lc0_lib, dependencies: gtest
  ), args: '--gtest_output=xml:chessboard.xml', timeout: 90)

  test('HashCat',
    executable('hashcat_test', 'src/utils/hashcat_test.cc',
    include_directories: includes, link_with: lc0_lib, dependencies: gtest
  ), args: '--gtest_output=xml:hashcat.xml', timeout: 90)

  test('PositionTest',
    executable('position_test', 'src/chess/position_test.cc',
    include_directories: includes, link_with: lc0_lib, dependencies: gtest
  ), args: '--gtest_output=xml:position.xml', timeout: 90)

  test('OptionsParserTest',
    executable('optionsparser_test', 'src/utils/optionsparser_test.cc',
    include_directories: includes, link_with: lc0_lib, dependencies: gtest
  ), timeout: 90)

  test('SyzygyTest',
    executable('syzygy_test', 'src/syzygy/syzygy_test.cc',
    include_directories: includes, link_with: lc0_lib, dependencies: gtest
  ), args: '--gtest_output=xml:syzygy.xml', timeout: 90)

  test('EncodePositionForNN', 
    executable('encoder_test', 'src/neural/encoder_test.cc',
    include_directories: includes, link_with: lc0_lib, dependencies: gtest
  ), args: '--gtest_output=xml:encoder.xml', timeout: 90)

endif
