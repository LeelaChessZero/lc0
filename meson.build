# This file is part of Leela Chess Zero.
# Copyright (C) 2018-2022 The LCZero Authors
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
        default_options : ['cpp_std=c++20', 'b_ndebug=if-release', 'warning_level=3', 'b_lto=true', 'b_vscrt=mt'],
        meson_version: '>=0.55')

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
if cc.get_id() != 'msvc'
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
  add_project_arguments(cc.get_supported_arguments(['/source-charset:utf-8']), language : 'cpp')
endif
if ['arm', 'aarch64'].contains(host_machine.cpu_family())
  if get_option('neon')
    add_project_arguments(cc.get_supported_arguments(['-mfpu=neon']), language : 'cpp')
    add_project_link_arguments(cc.get_supported_arguments(['-mfpu=neon']), language : 'cpp')
  endif
endif

# Files to compile.
deps = []
common_files = []
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
if run_command('scripts/checkdir.py', 'libs/lczero-common/proto', check : false).returncode() != 0
  if git.found()
    if run_command(git, 'status', check : false).returncode() == 0
      message('updating git submodule libs/lczero-common')
      run_command(git, 'submodule', 'update', '--init', '--recursive', check : false)
    else
      message('cloning lczero-common.git into libs/lczero-common')
      run_command(git, 'clone', '--depth=1',
                  'https://github.com/LeelaChessZero/lczero-common.git',
                  'libs/lczero-common/', check : false)
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
common_files += pb_files

# Extract git short revision.
short_rev = 'unknown'
if git.found()
  r = run_command(git, 'rev-parse', '--short', 'HEAD', check : false)
  if r.returncode() == 0
    # Now let's check if the working directory is clean.
    if run_command(git, 'diff-index', '--quiet', 'HEAD', check : false).returncode() == 0
      short_rev = r.stdout().strip()
      if run_command(git, 'describe', '--exact-match', '--tags', check : false).returncode() == 0
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
build_identifier = ''
if short_rev != ''
  build_identifier = 'git.' + short_rev
  message('Using build identifier "' + build_identifier + '".')
endif

conf_data = configuration_data()
conf_data.set_quoted('BUILD_IDENTIFIER', build_identifier)
configure_file(output: 'build_id.h', configuration: conf_data)

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

# ONNX and HLO protobufs.
gen_proto_src = generator(compile_proto, output: ['@BASENAME@.pb.h'],
  arguments : [
    '--proto_path=@CURRENT_SOURCE_DIR@/src',
    '--cpp_out=@BUILD_DIR@',
    '@INPUT@'])

files += gen_proto_src.process('src/neural/onnx/onnx.proto',
  preserve_path_from : meson.current_source_dir() + '/src/')

files += gen_proto_src.process('src/neural/xla/hlo.proto',
  preserve_path_from : meson.current_source_dir() + '/src/')

#############################################################################
## Main files
#############################################################################
common_files += [
  'src/chess/board.cc',
  'src/chess/gamestate.cc',
  'src/chess/position.cc',
  'src/chess/uciloop.cc',
  'src/neural/backend.cc',
  'src/neural/batchsplit.cc',
  'src/neural/decoder.cc',
  'src/neural/encoder.cc',
  'src/neural/factory.cc',
  'src/neural/loader.cc',
  'src/neural/register.cc',
  'src/neural/shared_params.cc',
  'src/neural/wrapper.cc',
  'src/search/classic/node.cc',
  'src/syzygy/syzygy.cc',
  'src/trainingdata/reader.cc',
  'src/trainingdata/trainingdata.cc',
  'src/trainingdata/writer.cc',
  'src/utils/commandline.cc',
  'src/utils/configfile.cc',
  'src/utils/esc_codes.cc',
  'src/utils/files.cc',
  'src/utils/logging.cc',
  'src/utils/optionsdict.cc',
  'src/utils/optionsparser.cc',
  'src/utils/random.cc',
  'src/utils/string.cc',
  'src/version.cc',
]

files += [
  'src/engine_loop.cc',
  'src/engine.cc',
  'src/neural/backends/network_check.cc',
  'src/neural/backends/network_demux.cc',
  'src/neural/backends/network_mux.cc',
  'src/neural/backends/network_random.cc',
  'src/neural/backends/network_record.cc',
  'src/neural/backends/network_rr.cc',
  'src/neural/backends/network_trivial.cc',
  'src/neural/memcache.cc',
  'src/neural/network_legacy.cc',
  'src/neural/onnx/adapters.cc',
  'src/neural/onnx/builder.cc',
  'src/neural/onnx/converter.cc',
  'src/neural/xla/hlo_builder.cc',
  'src/neural/xla/onnx2hlo.cc',
  'src/neural/xla/print_hlo.cc',
  'src/neural/xla/xla_tensor.cc',
  'src/search/classic/params.cc',
  'src/search/classic/search.cc',
  'src/search/classic/stoppers/alphazero.cc',
  'src/search/classic/stoppers/common.cc',
  'src/search/classic/stoppers/factory.cc',
  'src/search/classic/stoppers/legacy.cc',
  'src/search/classic/stoppers/simple.cc',
  'src/search/classic/stoppers/smooth.cc',
  'src/search/classic/stoppers/stoppers.cc',
  'src/search/classic/stoppers/timemgr.cc',
  'src/search/classic/wrapper.cc',
  'src/search/register.cc',
  'src/selfplay/game.cc',
  'src/selfplay/loop.cc',
  'src/selfplay/multigame.cc',
  'src/selfplay/tournament.cc',
  'src/tools/backendbench.cc',
  'src/tools/benchmark.cc',
  'src/tools/describenet.cc',
  'src/tools/leela2onnx.cc',
  'src/tools/onnx2leela.cc',
  'src/utils/histogram.cc',
  'src/utils/numa.cc',
  'src/utils/weights_adapter.cc',
]

files += [
  'src/search/instamove/instamove.cc',
]

includes += include_directories('src')

deps += dependency('threads')

#############################################################################
## Platform specific files
############################################################################
if host_machine.system() == 'windows'
  common_files += 'src/utils/filesystem.win32.cc'
else
  common_files += 'src/utils/filesystem.posix.cc'
endif

#############################################################################
## DAG CLASSIC SEARCH
#############################################################################
if get_option('dag_classic')
  files += [
    'src/search/dag_classic/node.cc',
    'src/search/dag_classic/params.cc',
    'src/search/dag_classic/search.cc',
    'src/search/dag_classic/stoppers/alphazero.cc',
    'src/search/dag_classic/stoppers/common.cc',
    'src/search/dag_classic/stoppers/factory.cc',
    'src/search/dag_classic/stoppers/legacy.cc',
    'src/search/dag_classic/stoppers/simple.cc',
    'src/search/dag_classic/stoppers/smooth.cc',
    'src/search/dag_classic/stoppers/stoppers.cc',
    'src/search/dag_classic/stoppers/timemgr.cc',
    'src/search/dag_classic/wrapper.cc',
  ]

  absl = subproject('abseil-cpp', default_options : ['warning_level=0', 'cpp_std=c++20'])
  deps += absl.get_variable('absl_container_dep').as_system()
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
    files += 'src/neural/backends/network_tf_cc.cc'
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

  dnnl_libdirs = [get_option('dnnl_dir') + '/lib64', get_option('dnnl_dir') + '/lib']
  dnnl_lib = cc.find_library('dnnl', dirs: dnnl_libdirs, required: false)

  openblas_libdirs = get_option('openblas_libdirs')
  openblas_lib = cc.find_library('openblas.dll', dirs: openblas_libdirs, required: false)
  if not openblas_lib.found()
    openblas_lib = cc.find_library('openblas', dirs: openblas_libdirs, required: false)
  endif

  if get_option('blas')
    if get_option('mkl') and mkl_lib.found()
      mkl_inc = get_option('mkl_include')
      if run_command('scripts/checkdir.py', mkl_inc, check : false).returncode() == 0
        includes += include_directories(mkl_inc)
      endif
      if cc.has_header('mkl.h')
        add_project_arguments(['-DUSE_MKL', '-DUSE_BLAS'], language : 'cpp')
        deps += [ mkl_lib ]
      endif

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

    deps += dependency('eigen3', fallback: ['eigen', 'eigen_dep']).as_system()

    ispc = find_program('ispc', required: false)
    ispc_arch = 'x86-64'
    ispc_extra_args = []
    if get_option('ispc') and ispc.found()
      ispc_native_only = get_option('ispc_native_only') and not meson.is_cross_build()  
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
      ispc_target = 'sse2-i32x8,sse4-i32x8,avx1-i32x8,avx2-i32x8,avx512knl-i32x16,avx512skx-i32x16'

      if host_machine.system() == 'android'
        ispc_extra_args += ['--target-os=android']
      endif

      if ['arm', 'aarch64'].contains(host_machine.cpu_family())
        outputnames = [ '@BASENAME@.o']
        if host_machine.cpu_family() == 'aarch64'
          ispc_target = 'neon-i32x8'
          ispc_arch = 'aarch64'
        else
          ispc_target = 'neon-i32x4'
          ispc_arch = 'arm'
        endif
      endif

      if ispc_native_only
        ispc_target = 'host'
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
    'src/neural/backends/blas/convolution1.cc',
    'src/neural/backends/blas/fully_connected_layer.cc',
    'src/neural/backends/blas/se_unit.cc',
    'src/neural/backends/blas/network_blas.cc',
    'src/neural/backends/blas/winograd_convolution3.cc'
    ]

    shared_files = [
    'src/neural/backends/shared/activation.cc',
    'src/neural/backends/shared/winograd_filter.cc',
    ]

    files += blas_files
    has_backends = true

    if get_option('ispc') and ispc.found()
      files += iscp_gen.process('src/neural/backends/blas/winograd_transform.ispc')
      files += iscp_gen.process('src/neural/backends/blas/layer_norm.ispc')
      files += iscp_gen.process('src/neural/backends/shared/activation.ispc')
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
      'src/neural/backends/opencl/network_opencl.cc',
      'src/neural/backends/opencl/OpenCL.cc',
      'src/neural/backends/opencl/OpenCLTuner.cc',
      'src/neural/backends/opencl/OpenCLBuffers.cc',
    ]

    shared_files = [
    'src/neural/backends/shared/activation.cc',
    'src/neural/backends/shared/winograd_filter.cc',
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

  if (get_option('cudnn') or get_option('plain_cuda')) and cu_blas.found() and cu_dart.found() and nvcc.found()
    deps += [cu_blas, cu_dart]
    cuda_files = ['src/neural/backends/cuda/layers.cc']
    if get_option('cudnn') and cu_dnn.found()
      deps += cu_dnn
      cuda_files += 'src/neural/backends/cuda/network_cudnn.cc'
      cuda_files += 'src/neural/backends/cuda/network_cuda.cc' # To support newer nets.
      add_project_arguments('-DUSE_CUDNN', language : 'cpp')
    elif get_option('plain_cuda')
      cuda_files += 'src/neural/backends/cuda/network_cuda.cc'
    endif
    foreach d : get_option('cudnn_include')
      if run_command('scripts/checkdir.py', d, check : false).returncode() == 0
        includes += include_directories(d, is_system: true)
      endif
    endforeach
    includes += include_directories('src/neural/backends/cuda/')

    cuda_arguments = ['-c', '@INPUT@', '-o', '@OUTPUT@',
                      '-I', meson.current_source_dir() + '/src']
    nvcc_help = run_command(nvcc, '-h', check : false).stdout()
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
    elif get_option('native_cuda') and nvcc_help.contains('-arch=native')
      nvcc_extra_args = ['-arch=native']
    elif nvcc_help.contains('-arch=all-major')
      nvcc_extra_args = ['-arch=all-major', '-Wno-deprecated-gpu-targets']
    else
      nvcc_extra_args = ['-Wno-deprecated-gpu-targets']
      # Fallback for cuda versions without -arch=all-major.
      foreach x : ['35', '50', '60', '70', '80']
        if nvcc_help.contains('sm_' + x)
          nvcc_extra_args += '-gencode=arch=compute_' + x + ',code=sm_' + x
        endif
      endforeach
      # For forward compatibility.
      if nvcc_help.contains('sm_80') # Cuda 11+
        nvcc_extra_args += '-gencode=arch=compute_80,code=compute_80'
      elif nvcc_help.contains('sm_75') # Cuda 10+
        nvcc_extra_args += '-gencode=arch=compute_75,code=compute_75'
      endif
    endif
    foreach x : get_option('cudnn_include')
      cuda_arguments += ['-I', x]
    endforeach
    if host_machine.system() == 'windows'
      outputname = '@BASENAME@.obj'
    else
      outputname = '@BASENAME@.o'
    endif
	 files += cuda_files
    files += custom_target('cuda fp32 code',
      input : 'src/neural/backends/cuda/common_kernels.cu',
      output : outputname,
      depend_files: 'src/neural/backends/cuda/winograd_helper.inc',
      command : [nvcc, nvcc_extra_args, cuda_arguments]
    )

    files += custom_target('cuda fp16 code',
      input : 'src/neural/backends/cuda/fp16_kernels.cu',
      output : outputname,
      depend_files: 'src/neural/backends/cuda/winograd_helper.inc',
      command : [nvcc, nvcc_extra_args, cuda_arguments]
    )
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
      'src/neural/backends/dx/network_dx.cc',
      'src/neural/backends/dx/shader_wrapper.cc',
      'src/neural/backends/dx/layers_dx.cc',
    ]
    files += dx_files
    deps += [dx_d3d12, dx_dxgi]

    subdir('src/neural/backends/dx/shaders')

    has_backends = true
  endif

  if get_option('onednn') and dnnl_lib.found()
    includes += include_directories(get_option('dnnl_dir') + '/include')
    deps += [ dnnl_lib, dependency('openmp', required:true) ]
    files += [
      'src/neural/backends/onednn/network_onednn.cc',
      'src/neural/backends/onednn/layers.cc',
    ]
    has_backends = true
  endif

  ## ~~~~~~~~~~
  ## ONNX
  ## ~~~~~~~~~~
  if get_option('onnx_libdir') != '' and get_option('onnx_include') != ''
    deps += cc.find_library('onnxruntime', dirs: get_option('onnx_libdir'),
                            required: true)
    includes += include_directories(get_option('onnx_include'), is_system: true)
    cc.has_header('onnxruntime_cxx_api.h', required: true,
                  args: '-I' + get_option('onnx_include'))
    if not cc.has_header('cpu_provider_factory.h',
                         args: '-I' + get_option('onnx_include'))
      cc.has_header('../providers/cpu/cpu_provider_factory.h', required: true,
                    args: '-I' + get_option('onnx_include'))
      includes += include_directories(get_option('onnx_include') + '/../providers/cpu',
                                      is_system: true)
    endif
    files += 'src/neural/backends/network_onnx.cc'
    if cc.find_library('onnxruntime_providers_rocm',
                       dirs: get_option('onnx_libdir'), required: false).found()
      add_project_arguments('-DUSE_ROCM', language : 'cpp')
    endif
    has_backends = true
  endif

  ## ~~~~~~~~
  ## Metal
  ## ~~~~~~~~
  # Metal backend only available on MacOS.
  # Check for required frameworks - Foundation, Metal and MPS.
  metal_frameworks = dependency('appleframeworks',
                                modules : ['Foundation', 'Metal', 'MetalPerformanceShaders', 'MetalPerformanceShadersGraph'],
                                required: get_option('metal'))

  if (metal_frameworks.found() and add_languages('objc', 'objcpp'))
    deps += metal_frameworks

    files += [
      'src/neural/backends/metal/network_metal.cc',
      'src/neural/backends/metal/mps/NetworkGraph.mm',
      'src/neural/backends/metal/mps/MetalNetworkBuilder.mm',
    ]

    has_backends = true
    add_project_arguments('-fobjc-arc', language : 'objc')
    add_project_arguments('-fobjc-arc', language : 'objcpp')
  endif

  ## ~~~~~~~~
  ## XLA
  ## ~~~~~~~~
  if get_option('xla')
      files += [
        'src/neural/backends/xla/network_xla.cc',
        'src/neural/backends/xla/pjrt.cc',
        'src/neural/backends/xla/xla_runner.cc',
      ]
      deps += cc.find_library('dl', required: false)
      has_backends = true
  endif

  ## ~~~~
  ## Sycl
  ## ~~~~
  if get_option('sycl') != 'off'
      has_backends = true
      message('Building SYCL')
      add_project_arguments('-fsycl', language : 'cpp')
      add_project_link_arguments('-fsycl', language : 'cpp')

      files += 'src/neural/backends/sycl/layers.cc.dp.cpp'
      files += 'src/neural/backends/sycl/network_sycl.cc.dp.cpp'
      files += 'src/neural/backends/sycl/common_kernels.dp.cpp'
      files += 'src/neural/backends/sycl/fp16_kernels.dp.cpp'

      if get_option('sycl') == 'l0'
        message('Building SYCL for the L0 backend')
        add_project_arguments('-DMKL_ILP64', language : 'cpp')
        deps += cc.find_library('mkl_sycl', required: true)
        deps += cc.find_library('mkl_intel_ilp64', required: true)
        deps += cc.find_library('mkl_sequential', required: true)
        deps += cc.find_library('mkl_core', required: true)
        deps += cc.find_library('OpenCL', required: true)
      elif get_option('sycl') == 'amd'
        error('Building SYCL for AMD backend not yet supported')
      else
        error('Building SYCL for the NVIDIA backend not yet supported')
      endif
      if host_machine.system() == 'windows'
        # For sycl under windows we need to link using icx to generate the device code.
        # This script edits build.ninja for this and for an icx dependency issue.
        meson.add_postconf_script('scripts/sycl_build_hack.py')
      endif
  endif

endif # if get_option('build_backends')

if not has_backends and get_option('lc0') and get_option('build_backends')
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

if not get_option('f16c')
  add_project_arguments('-DNO_F16C', language : 'cpp')
endif

if not get_option('pext')
  add_project_arguments('-DNO_PEXT', language : 'cpp')
endif

if get_option('embed')
  add_project_arguments('-DEMBED', language : 'cpp')
endif

if get_option('default_search') != ''
  add_project_arguments('-DDEFAULT_SEARCH=' +
                        get_option('default_search'), language : 'cpp')
endif

if get_option('lc0')
  files += common_files
  executable('lc0', 'src/main.cc',
       files, include_directories: includes, dependencies: deps, install: true)
endif

#############################################################################
## Rescorer Executable
#############################################################################

if get_option('rescorer')
  deps += subproject('gaviotatb').get_variable('gaviotatb_dep')
  executable('rescorer', 'src/rescorer_main.cc',
       [common_files, 'src/trainingdata/rescorer.cc'],
       include_directories: includes, dependencies: deps, install: true)
endif

#############################################################################
## Tests
#############################################################################

if get_option('gtest')
  gtest = dependency('gtest', fallback: ['gtest', 'gtest_dep'])
  gmock = dependency('gmock', fallback: ['gtest', 'gmock_dep'])
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

  test('EngineTest',
    executable('engine_test', 'src/engine_test.cc', pb_files,
    include_directories: includes, link_with: lc0_lib, dependencies: [gtest, gmock]),
    args: '--gtest_output=xml:engine_test.xml', timeout: 90)
endif


#############################################################################
## Python bindings
#############################################################################

if get_option('python_bindings')
  pymod = import('python')
  python = pymod.find_installation('python3')
  if not python.language_version().version_compare('>3.7')
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
    dependencies: [cpython] + deps,
    subdir: 'lczero',
    install: true)
endif
