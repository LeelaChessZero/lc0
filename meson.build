project('lc0', 'cpp')
  #        default_options : ['cpp_std=c++17'])

add_global_arguments('-std=c++17', '-Wthread-safety', language : 'cpp')
cc = meson.get_compiler('cpp')

# Installed from https://github.com/FloopCZ/tensorflow_cc
tensorflow_cc = declare_dependency(
  include_directories: include_directories(
    '/usr/local/include/tensorflow',
    '/usr/local/include/tensorflow/bazel-genfiles',
    '/usr/local/include/tensorflow/tensorflow/contrib/makefile/downloads',
    '/usr/local/include/tensorflow/tensorflow/contrib/makefile/downloads/eigen',
    '/usr/local/include/tensorflow/tensorflow/contrib/makefile/downloads/gemmlowp',
    '/usr/local/include/tensorflow/tensorflow/contrib/makefile/downloads/nsync/public',
    '/usr/local/include/tensorflow/tensorflow/contrib/makefile/gen/protobuf-host/include',
  ),
  dependencies: [
      cc.find_library('libtensorflow_cc', dirs: '/usr/local/lib/tensorflow_cc/'),
      cc.find_library('dl'),
      cc.find_library('pthread'),
      cc.find_library('libprotobuf', dirs: '/usr/local/lib/tensorflow_cc/'),
  ],
)

deps = []
deps += tensorflow_cc
deps += cc.find_library('stdc++fs')
deps += cc.find_library('libcublas', dirs: ['/opt/cuda/lib64/', '/usr/local/cuda/lib64/'])
deps += cc.find_library('libcudnn', dirs: ['/opt/cuda/lib64/', '/usr/local/cuda/lib64/'])
deps += cc.find_library('libcudart', dirs: ['/opt/cuda/lib64/', '/usr/local/cuda/lib64/'])
# deps += dependency('libprofiler')

nvcc = find_program('nvcc')
cuda_files = [
  'src/neural/network_cudnn.cu',
]

cuda_gen = generator(nvcc,
    output: '@BASENAME@.o',
    arguments: ['--std=c++14', '-c', '@INPUT@', '-o', '@OUTPUT@', '-I', '../src'],
)

files = [
  'src/chess/bitboard.cc',
  'src/chess/board.cc',
  'src/mcts/node.cc',
  'src/mcts/search.cc',
  'src/neural/cache.cc',
  'src/neural/factory.cc',
  'src/neural/loader.cc',
  'src/neural/writer.cc',
  'src/neural/network_mux.cc',
  'src/neural/network_random.cc',
  'src/neural/network_tf.cc',
  'src/utils/transpose.cc',
  'src/utils/commandline.cc',
  'src/utils/optionsdict.cc',
  'src/utils/random.cc',
  'src/engine.cc',
  'src/optionsparser.cc',
  'src/selfplay/game.cc',
  'src/selfplay/tournament.cc',
  'src/selfplay/loop.cc',
  'src/uciloop.cc',
  cuda_gen.process(cuda_files)
]

includes = []
includes += include_directories('src')

executable('lc0', 'src/main.cc',
  files, include_directories: includes, dependencies: deps)


### Tests

test_deps = deps
test_deps += dependency('gtest')

test('ChessBoard',
  executable('chessboard_test', 'src/chess/board_test.cc',
  files, include_directories: includes, dependencies: test_deps
))

test('HashCat',
  executable('hashcat_test', 'src/utils/hashcat_test.cc',
  files, include_directories: includes, dependencies: test_deps
))
