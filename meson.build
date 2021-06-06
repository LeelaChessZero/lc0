# This file is part of Leela Chess Zero.
# Copyright (C) 2018-2020 The LCZero Authors
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
        default_options : ['cpp_std=c++17', 'b_ndebug=if-release', 'warning_level=3', 'b_lto=true', 'b_vscrt=mt'],
        meson_version: '>=0.52')

cc = meson.get_compiler('cpp')

if not cc.has_header('optional') or not cc.has_header('string_view')
    error('Lc0 requires a compiler supporting C++17, for example g++ v8.0, ' +
          'clang v5.0 or later (with C++17 stdlib) and Visual Studio 2017 or ' +
          'later.')
endif

if not cc.has_header('charconv')
    warning('Your compiler or library does not have full C++17 support. ' +
            'See the README for compilers that are known to be working. ' +
            'This will become an error in the future.')
endif

if cc.get_id() == 'clang'
  # Thread safety annotation
  add_project_arguments('-Wthread-safety', language : 'cpp')
endif
if cc.get_id() == 'clang' or cc.get_id() == 'gcc'
  if get_option('buildtype') == 'release'
    add_project_arguments(cc.get_supported_arguments(['-march=native']), language : 'cpp')
  endif
endif
if cc.get_id() == 'msvc'
  # Silence some zlib warnings.
  add_global_arguments('/wd4131', '/wd4267', '/wd4127', '/wd4244', '/wd4245', language : 'c')
endif
if host_machine.system() == 'windows'
  add_project_arguments('-DNOMINMAX', language : 'cpp')
endif
if host_machine.cpu_family() == 'arm'
  if get_option('neon')
    add_project_arguments(cc.get_supported_arguments(['-mfpu=neon']), language : 'cpp')
    add_project_link_arguments(cc.get_supported_arguments(['-mfpu=neon']), language : 'cpp')
  endif
endif

# Files to compile.
deps = []
files = []
includes = []
has_backends = false

# Third party files.
includes += include_directories('third_party', is_system: true)

# Compiling protobufs.
compile_proto = find_program('scripts/compile_proto.py')
gen = generator(compile_proto, output: ['@BASENAME@.pb.h'],
  arguments : [
    '--proto_path=@CURRENT_SOURCE_DIR@/libs/lczero-common',
    '--cpp_out=@BUILD_DIR@',
    '@INPUT@'])

# Handle submodules.
git = find_program('git', required: false)
if run_command('scripts/checkdir.py', 'libs/lczero-common/proto').returncode() != 0
  if git.found()
    if run_command(git, 'status').returncode() == 0
      message('updating git submodule libs/lczero-common')
      run_command(git, 'submodule', 'update', '--init', '--recursive')
    else
      message('cloning lczero-common.git into libs/lczero-common')
      run_command(git, 'clone', '--depth=1',
                  'https://github.com/LeelaChessZero/lczero-common.git',
                  'libs/lczero-common/')
    endif
  else
    error('Please install git to automatically fetch submodules or download the archives manually from GitHub.')
  endif
endif

pb_files = [
  'src/utils/protomessage.cc',
  gen.process('libs/lczero-common/proto/net.proto',
    preserve_path_from : meson.current_source_dir() + '/libs/lczero-common/')
]
files += pb_files

# Extract git short revision.
short_rev = 'unknown'
if git.found()
  r = run_command(git, 'rev-parse', '--short', 'HEAD')
  if r.returncode() == 0
    # Now let's check if the working directory is clean.
    if run_command(git, 'diff-index', '--quiet', 'HEAD').returncode() == 0
      short_rev = r.stdout().strip()
      if run_command(git, 'describe', '--exact-match', '--tags').returncode() == 0
        short_rev = ''
      endif
    else
      short_rev = 'dirty'
      warning('Cannot extract valid git short revision from dirty working directory.')
    endif
  else
    warning('Failed to parse short revision. Use git clone instead of downloading the archive from GitHub.')
  endif
endif

# Construct build identifier.
if short_rev != ''
  build_identifier = 'git.' + short_rev
  add_project_arguments('-DBUILD_IDENTIFIER="' + build_identifier + '"', language : 'cpp')
  message('Using build identifier "' + build_identifier + '".')
endif

# Some malloc libraries require to be linked first.
if get_option('malloc') == 'mimalloc' and cc.get_id() == 'msvc'
  if get_option('b_vscrt') != 'md' and get_option('b_vscrt') != 'mdd'
    error('You need -Db_vscrt=md (or mdd)')
  endif
  add_project_link_arguments('/INCLUDE:mi_version', language : 'cpp')
  deps += cc.find_library('mimalloc-override', dirs: get_option('mimalloc_libdir'), required: true)
elif get_option('malloc') != ''
  deps += cc.find_library(get_option('malloc'), required: true)
endif

#############################################################################
## Main files
#############################################################################
files += [
  'src/engine.cc',
  'src/version.cc',
  'src/benchmark/backendbench.cc',
  'src/benchmark/benchmark.cc',
  'src/chess/bitboard.cc',
  'src/chess/board.cc',
  'src/chess/position.cc',
  'src/chess/uciloop.cc',
  'src/mcts/node.cc',
  'src/mcts/params.cc',
  'src/mcts/search.cc',
  'src/mcts/stoppers/common.cc',
  'src/mcts/stoppers/factory.cc',
  'src/mcts/stoppers/legacy.cc',
  'src/mcts/stoppers/alphazero.cc',
  'src/mcts/stoppers/smooth.cc',
  'src/mcts/stoppers/stoppers.cc',
  'src/mcts/stoppers/timemgr.cc',
  'src/neural/cache.cc',
  'src/neural/encoder.cc',
  'src/neural/factory.cc',
  'src/neural/loader.cc',
  'src/neural/network_check.cc',
  'src/neural/network_demux.cc',
  'src/neural/network_legacy.cc',
  'src/neural/network_mux.cc',
  'src/neural/network_random.cc',
  'src/neural/network_record.cc',
  'src/neural/network_rr.cc',
  'src/neural/writer.cc',
  'src/selfplay/game.cc',
  'src/selfplay/loop.cc',
  'src/selfplay/tournament.cc',
  'src/syzygy/syzygy.cc',
  'src/utils/commandline.cc',
  'src/utils/configfile.cc',
  'src/utils/esc_codes.cc',
  'src/utils/histogram.cc',
  'src/utils/logging.cc',
  'src/utils/numa.cc',
  'src/utils/optionsdict.cc',
  'src/utils/optionsparser.cc',
  'src/utils/random.cc',
  'src/utils/string.cc',
  'src/utils/weights_adapter.cc',
]
includes += include_directories('src')

deps += dependency('threads')

#############################################################################
## Platform specific files
############################################################################
if host_machine.system() == 'windows'
  files += 'src/utils/filesystem.win32.cc'
else
  files += 'src/utils/filesystem.posix.cc'
endif

#############################################################################
## BACKENDS
#############################################################################

if get_option('build_backends')
  ## ~~~~~~~~~~
  ## Tensorflow
  ## ~~~~~~~~~~
  tf_dl_lib = cc.find_library('dl', required: false)
  # We had `is_system: true` to reduce warnings, but meson > 0.56.0 breaks.
  tf_tensorflow_cc_lib = dependency('tensorflow_cc', required: false)
  if get_option('tensorflow') and tf_dl_lib.found() and tf_tensorflow_cc_lib.found()
    deps += [tf_dl_lib, tf_tensorflow_cc_lib]
    files += 'src/neural/network_tf_cc.cc'
    has_backends = true
  endif

  ## ~~~~~
  ## Blas
  ## ~~~~~

  shared_files = []

  accelerate_lib = dependency('Accelerate', required: false)

  mkl_libdirs = get_option('mkl_libdirs')
  mkl_lib = cc.find_library('mkl_rt', dirs: mkl_libdirs, required: false)
  if not mkl_lib.found()
    mkl_lib = cc.find_library('mklml', dirs: mkl_libdirs, required: false)
  endif

  dnnl_libdirs = get_option('dnnl_dir') + '/lib'
  dnnl_lib = cc.find_library('dnnl', dirs: dnnl_libdirs, required: false)

  openblas_libdirs = get_option('openblas_libdirs')
  openblas_lib = cc.find_library('openblas.dll', dirs: openblas_libdirs, required: false)
  if not openblas_lib.found()
    openblas_lib = cc.find_library('openblas', dirs: openblas_libdirs, required: false)
  endif

  if get_option('blas')
    if get_option('mkl') and mkl_lib.found()
      add_project_arguments(['-DUSE_MKL', '-DUSE_BLAS'], language : 'cpp')
      mkl_inc = get_option('mkl_include')
      if run_command('scripts/checkdir.py', mkl_inc).returncode() == 0
        includes += include_directories(mkl_inc)
      endif
      deps += [ mkl_lib ]

    elif get_option('dnnl') and dnnl_lib.found()
      add_project_arguments(['-DUSE_DNNL', '-DUSE_BLAS'], language : 'cpp')
      includes += include_directories(get_option('dnnl_dir') + '/include')
      deps += [ dnnl_lib, dependency('openmp', required:true) ]

    elif get_option('accelerate') and accelerate_lib.found()
      deps += [ accelerate_lib ]
      add_project_arguments('-DUSE_BLAS', language : 'cpp')

    elif get_option('openblas') and openblas_lib.found()
      add_project_arguments(['-DUSE_OPENBLAS', '-DUSE_BLAS'], language : 'cpp')

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

    endif

    deps += dependency('eigen3', fallback: ['eigen', 'eigen_dep'])

    ispc = find_program('ispc', required: false)
    ispc_arch = 'x86-64'
    ispc_extra_args = []
    if get_option('ispc') and ispc.found()
      ispc_native_only = get_option('ispc_native_only')
      if host_machine.system() == 'windows'
        outputnames = [ '@BASENAME@.obj']
        if not ispc_native_only
          outputnames += ['@BASENAME@_sse2.obj', '@BASENAME@_sse4.obj',
                          '@BASENAME@_avx.obj', '@BASENAME@_avx2.obj',
                          '@BASENAME@_avx512knl.obj', '@BASENAME@_avx512skx.obj' ]
        endif
      else
        ispc_extra_args += ['--pic']
        outputnames = [ '@BASENAME@.o']
        if not ispc_native_only
          outputnames += ['@BASENAME@_sse2.o', '@BASENAME@_sse4.o',
                          '@BASENAME@_avx.o', '@BASENAME@_avx2.o',
                          '@BASENAME@_avx512knl.o', '@BASENAME@_avx512skx.o' ]
        endif
      endif
      if ispc_native_only
        ispc_target = 'host'
      else
        ispc_target = 'sse2-i32x8,sse4-i32x8,avx1-i32x8,avx2-i32x8,avx512knl-i32x16,avx512skx-i32x16'
      endif

      if host_machine.system() == 'android'
        ispc_extra_args += ['--target-os=android']
        if host_machine.cpu_family() == 'arm'
          outputnames = [ '@BASENAME@.o']
          if host_machine.cpu() == 'aarch64'
            ispc_target = 'neon-i32x8'
            ispc_arch = 'aarch64'
          else
            ispc_target = 'neon-i32x4'
            ispc_arch = 'arm'
          endif
        endif
      endif

      iscp_gen = generator(ispc,
        output: [ '@BASENAME@_ispc.h', outputnames ],
        arguments: [ '-O2', '--wno-perf', '--arch=' + ispc_arch,
                     '--target=' + ispc_target,
                     '@INPUT@', '-o', '@OUTPUT1@' ,'-h', '@OUTPUT0@' ]
                     + ispc_extra_args
      )
    endif

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

  opencl_framework=dependency('OpenCL', method: 'extraframework', required: false)
  if opencl_framework.found()
      opencl_dep = [ opencl_framework ]
      has_opencl = true

  elif opencl_lib.found() and cc.has_header('CL/opencl.h', args: '-I' + get_option('opencl_include'))
      opencl_dep = [ opencl_lib ]
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

    if not opencl_framework.found()
      includes += include_directories(get_option('opencl_include'))
    endif
    deps += opencl_dep
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

  if (get_option('cudnn') or get_option('plain_cuda')) and cu_blas.found() and cu_dart.found() and nvcc.found()
    deps += [cu_blas, cu_dart]
    if get_option('cudnn') and cu_dnn.found()
      deps += cu_dnn
      cuda_files += 'src/neural/cuda/network_cudnn.cc'
      add_project_arguments('-DUSE_CUDNN', language : 'cpp')
    endif
    if get_option('plain_cuda')
      cuda_files += 'src/neural/cuda/network_cuda.cc'
    endif
    foreach d : get_option('cudnn_include')
      if run_command('scripts/checkdir.py', d).returncode() == 0
        includes += include_directories(d)
      endif
    endforeach
    includes += include_directories('src/neural/cuda/')

    cuda_arguments = ['@EXTRA_ARGS@', '-c', '@INPUT@', '-o', '@OUTPUT@',
                      '-I', meson.current_source_dir() + '/src']
    if host_machine.system() == 'windows'
      if get_option('b_vscrt') == 'mt'
        cuda_arguments += ['-Xcompiler', '-MT']
      elif get_option('b_vscrt') == 'mtd'
        cuda_arguments += ['-Xcompiler', '-MTd']
      elif get_option('b_vscrt') == 'mdd' or (get_option('b_vscrt') == 'from_buildtype' and get_option('buildtype') == 'debug')
        cuda_arguments += ['-Xcompiler', '-MDd']
      elif get_option('b_vscrt') != 'none'
        cuda_arguments += ['-Xcompiler', '-MD']
      endif
    else
      cuda_arguments += ['--std=c++14', '-Xcompiler', '-fPIC']
    endif
    if get_option('nvcc_ccbin') != ''
      cuda_arguments += ['-ccbin=' + get_option('nvcc_ccbin')]
    endif
    cuda_cc = get_option('cc_cuda') # Unfortunately option cuda_cc is reserved.
    nvcc_extra_args = []
    if cuda_cc != ''
      nvcc_extra_args = ['-arch=compute_' + cuda_cc, '-code=sm_' + cuda_cc]
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
    files += cuda_gen.process(cuda_files_nvcc_common, extra_args: nvcc_extra_args)
    nvcc_arch = '-arch=compute_70'
    nvcc_sm_list = ['sm_80', 'sm_75', 'sm_86', 'sm_70']
    if host_machine.system() != 'windows'
      nvcc_arch = '-arch=compute_60'
      nvcc_sm_list += ['sm_60']
      if ['arm', 'aarch64'].contains(host_machine.cpu_family())
        # Add Jetson versions to the list.
        message('Jetson support enabled.')
        nvcc_arch = '-arch=compute_53'
        nvcc_sm_list += ['sm_72', 'sm_62', 'sm_53']
      endif
    endif
    # Ignore the given CC for fp16 when it is not in the supported list.
    if cuda_cc == '' or not nvcc_sm_list.contains('sm_' + cuda_cc)
      nvcc_extra_args = [nvcc_arch]
      nvcc_help = run_command(nvcc, '-h').stdout()
      foreach x : nvcc_sm_list
        if nvcc_help.contains(x)
          nvcc_extra_args += '-code=' + x
        endif
      endforeach
    endif
    files += cuda_gen.process(cuda_files_nvcc_fp16, extra_args: nvcc_extra_args)
    has_backends = true
  endif

  ## ~~~~~~~~
  ## DirectX
  ## ~~~~~~~~

  # we should always be able to build DirectX12 backend on windows platform
  if host_machine.system() == 'windows' and get_option('dx')
    dx_d3d12 = cc.find_library('d3d12')
    dx_dxgi = cc.find_library('dxgi')

    dx_files = [
      'src/neural/dx/network_dx.cc',
      'src/neural/dx/shader_wrapper.cc',
      'src/neural/dx/layers_dx.cc',
      'src/neural/dx/fp16_utils.cc',
    ]
    files += dx_files
    deps += [dx_d3d12, dx_dxgi]

    subdir('src/neural/dx/shaders')

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
  if host_machine.system() == 'windows'
    # In several cases where a zlib dependency was detected on windows, it
    # caused trouble (crashes or failed builds). Better safe than sorry.
    deps += subproject('zlib').get_variable('zlib_dep')
  else
    deps += dependency('zlib', fallback: ['zlib', 'zlib_dep'])
  endif

  ## ~~~~~~~~
  ## Profiler
  ## ~~~~~~~~
  if get_option('buildtype') != 'release'
    deps += cc.find_library('libprofiler',
      dirs: ['/usr/local/lib'], required: false)
  endif

  deps += cc.find_library('libatomic', required: false)

#############################################################################
## Main Executable
#############################################################################

if not get_option('popcnt')
  add_project_arguments('-DNO_POPCNT', language : 'cpp')
endif

if not get_option('pext')
  add_project_arguments('-DNO_PEXT', language : 'cpp')
endif

if get_option('embed')
  add_project_arguments('-DEMBED', language : 'cpp')
endif

executable('lc0', 'src/main.cc',
  files, include_directories: includes, dependencies: deps, install: true)

#############################################################################
## Tests
#############################################################################

if get_option('gtest')
  gtest = dependency('gtest', fallback: ['gtest', 'gtest_dep'])
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
  ), args: '--gtest_output=xml:optionsparser.xml', timeout: 90)

  test('SyzygyTest',
    executable('syzygy_test', 'src/syzygy/syzygy_test.cc',
    include_directories: includes, link_with: lc0_lib, dependencies: gtest
  ), args: '--gtest_output=xml:syzygy.xml', timeout: 90)

  test('EncodePositionForNN',
    executable('encoder_test', 'src/neural/encoder_test.cc', pb_files,
    include_directories: includes, link_with: lc0_lib,
    dependencies: [gtest]
  ), args: '--gtest_output=xml:encoder.xml', timeout: 90)
endif


#############################################################################
## Python bindings
#############################################################################

if get_option('python_bindings')
  pymod = import('python')
  python = pymod.find_installation('python3')
  if python.language_version() < '3.7'
    error('You need python 3.7 or newer')
  endif
  py_bindings_generator = find_program('scripts/gen_py_bindings.py')

  gen_py_bindings = custom_target('backends', input:[], output:['backends.cc'],
    command : [py_bindings_generator, '@OUTPUT0@'])

  py_files = [ gen_py_bindings ]

  cpython = dependency('python3')
  python.extension_module('backends',
    [py_files + files],
    include_directories: [includes],
    dependencies: [cpython] + deps)
endif
