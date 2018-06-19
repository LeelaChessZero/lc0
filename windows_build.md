## Windows MKL/OpenCL

0. Install Microsoft Visual Studio. For vs2017 make sure the option "Desktop development with C++"
   is installed (you can add it if not)

1. Install git for windows - this can be used to get lc0 but also needed for meson

2. Install MKL - dont forget to run `mklvars.bat intel64` to set up the paths to the dlls

3. For OpenCL you also need to install:
   * Boost
   * OpenCL libraries

4. Install Python3 - be sure to check the box to add python to the path

5. Install Meson: `pip3 install --upgrade meson`

6. Edit `build-cl.cmd`:

*  If you use MSVS other than 2017 community edition (or if it's installed into non-standard location)
   replace the path to vcvarsall.bat and MSBuild.exe. If you can't find vcvarsall.bat on VS2017, you
   need to install option "Desktop development with C++". Some example paths are in comments.
*  `--backend 2017` replace 2017 with the correct version.
*  `mkl_include` and `mkl_libdirs` should point to the MKL include and lib directories
*  If building with OpenCL `opencl_libdirs` and `opencl_include` should point to the OpenCL include and
   lib directories. The include directory is the one with the `CL` directory containing `opencl.h`, not
   directly the one containing `opencl.h`.

7. Run `build-cl.cmd`. It will generate MSVS project and pause.

8. Hit <Enter> to build it.

9. Resulting binary will be `build/lc0.exe`

Alternatively you can

8. open generated solution `build/lc0.sln` in Visual Studio and build yourself.


