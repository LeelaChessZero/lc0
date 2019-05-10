
# Building lc0 on Android, with Termux

This is a quick way of compiling lc0 on Android by using Termux, it might not be the best way to do it but it works now. First, install Termux from Google Play:

[https://play.google.com/store/apps/details?id=com.termux](https://play.google.com/store/apps/details?id=com.termux)

After the install you should be ready to start adding packages.

## Add repos and required packages

You will need to add a custom repo that has clang, openblas and other goodies. Don't worry, the repo is listed on the Termux wiki [https://wiki.termux.com/wiki/Package_Management](https://wiki.termux.com/wiki/Package_Management).

```
pkg install coreutils curl git
curl -L https://its-pointless.github.io/setup-pointless-repo.sh | sh
```

After the repo is added you should add the required packages:

```
pkg install clang ninja openblas python
pip3 install meson
```

## Get the project from git

Get lc0 from git, you might use depth 1 for a smaller download:

```
git clone --depth 1 https://github.com/LeelaChessZero/lc0.git
cd lc0
git submodule update --init --recursive
```

## Tweak some parameters and build

Now you can start the build process, it will take a long time depending on your phone.

```
CC=clang CXX=clang++ ./build.sh -Ddefault_library=static -Dandroid=true
```

Hopefully no error will appear and you will have a successful build.

## Test the binary

You can test the binary right away from Termux. It just needs a weights file, let's say you copied it to the root of your phone:

```
lc0/build/release/lc0 benchmark -w /sdcard/11258-48x5-se.pb.gz
```

## Building the OpenCL back-end

This step is optional and requires a little more knowledge of your Android system, also a phone that supports OpenCL. First get these three things:

* Locate the library libOpenCL.so and remember its path, usually it's on `/system/vendor/lib64`
* Locate the library liblzma.so and remember its path, usually `/system/lib64/`
* Get the OpenCL headers from here [https://github.com/KhronosGroup/OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers) expand & copy them let's say on the parent folder of your lc0 source. Remember its relative path, for example `../include/`

Now edit the `meson.build` file and find the following line:

```
add_project_link_arguments('/system/lib64/liblzma.so', language:'cpp')
```

you should replace the path with the exact one you've got for the `libzma` library. Now there is a longer command for performing the build:

```
CC=clang CXX=clang++ ./build.sh -Ddefault_library=static -Dandroid=true -Dopencl_include=../include/ -Dopencl_libdirs=/system/vendor/lib64
```

replacing the two paths for the include and lib dirs with your own. That's it, if you got all the paths right there's a good chance the build will succeed and you'll have a working OpenCL back-end.

## Issues

The generated binary is dynamically linked. Most chess GUIs will fail when loading it because they'll be missing the libraries it has in the Termux environment. Even when you copy the libraries along with the binary, the GUIs won't search the libraries on the current path. In my phone the dependencies are the following:

```
ldd lc0/build/release/lc0
libc.so
libopenblas.so
libOpenCL.so
liblog.so
libc++_shared.so
libm.so
libdl.so
```

The solution is to make a static-linked binary. I tried passing -static on LDFLAGS but then more errors appeared.

## Future work

The ideal outcome of this is to generate an installable package that's compatible with Android chess GUIs, for example by implementing the Open Exchange Protocol for engines [https://github.com/garawaa/chessenginesupport-androidlib/](https://github.com/garawaa/chessenginesupport-androidlib/)

It's really a very straightforward implementation. Libraries and dependent files remain something to be tested.

