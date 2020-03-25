# Build lc0 with tensorflow
The first step is to add `-Dtensorflow=true` to the meson comamnd line. If you are lucky meson will pick up the dependency and everything will work right away. If not, then there is some more meson options are required:
1. Set `-Dtensorflow_include=/path/to/tensorflow/include/files`. This may not be the top tensorflow include directory, in some installations it was `tensorflow/bazel-bin/tensorflow/include` further down.
2. Set `-Dtensorflow_libdir=/path/to/tensorflow_cc/library`. If you get link errors for protobuf functions, you should also add the path to protobuf library compiled during the tensorflow build, so use `-Dtensorflow_libdir=/path/to/tensorflow_cc/library,/path/to/protobuf/library` instead.
3. Make sure you don't pass `-Deigen=true`, it will most probably create conflcits.
