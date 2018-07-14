FROM floopcz/tensorflow_cc:ubuntu-shared-cuda

RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list' && apt-get update && apt-get install -y intel-mkl-64bit-2018.2-046
RUN apt-get install -y clang-6.0 ninja-build python3-pip nvidia-opencl-dev libopenblas-dev libboost-dev nvidia-cuda-dev nvidia-cuda-toolkit libgtest-dev git ssh tar gzip ca-certificates sudo
RUN pip3 install meson
RUN ln -s /usr/include/ /usr/include/openblas

RUN curl -OL https://github.com/google/protobuf/releases/download/v3.5.1/protoc-3.5.1-linux-x86_64.zip
RUN unzip protoc-3.5.1-linux-x86_64.zip -d protoc3
RUN sudo mv protoc3/bin/* /usr/local/bin/
RUN sudo mv protoc3/include/* /usr/local/include/
