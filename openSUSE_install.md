![openSUSE Logo](https://github.com/openSUSE/artwork/blob/master/logos/official/logo-color.png?style=centerme)
# lc0 on openSUSE 

openSUSE is a popular RPM based Linux distro, the install sources can be downloaded from [https://software.opensuse.org](https://software.opensuse.org) .

The steps described here are minimal, enough to install and run lc0 on openSUSE. The reader is encouraged to skim the Supplementary Information after completing the Install where additional information can be found to download an icon to beautify and enter complete information about the lc0 engine into Arena. I doubt anyone would want to run on a 32-bit machine, but for these kinds of oddball installs, modifying the script should be next to trivial.

If the User finds anything in this guide unclear, the original documentation is linked under "Supplemental Information" in the last section. And, don't forget to recommend(as an Issue) or submit a change (as a Pull).

## RPM packages vs Building from Source

An extremely versatile Build script that makes building from source simple and trivial is provided below which should run on practically any version of openSUSE but supports only the openBLAS backend, which means it can be run on any openSUSE without any hardware dependencies, and supports any version of openSUSE LEAP(ie 42.x, 15.1, etc, possibly ARM) or Tumbleweed. Most of the procedure after starting the script is just waiting to finish, no technical knowledge is required.

For those who instead prefer to not install the many files to build lc0, experimental packages (as of this writing) are available for LEAP 15 and Tumbleweed only (as of this writing). Currently the RPM supports OpenCL(AMD GPU only) and BLAS (which can be installed on any hardware). For Users who wish to use a pre-built binary from a package, skip down to ["RPM packages"](https://github.com/putztzu/lc0/blob/master/openSUSE_install.md#rpm-packages)

If someone wishes to do the work to investigate the procedures to install other backends (As of this writing, Tensorflow and CUDA are possible) findings and updates to this guide are welcomed.

## Supported openSUSE versions

The following until the "RPM Packages" section describes running the build script which would be the more flexible and versativle option today, which creates a binary for more than just the LEAP 15.0 and Tumbleweed x64 platforms

These build script instructions have been tested on current versions of LEAP 15.0 (The stable release) and Tumbleweed (The Development Rolling Release) on 64-bit machines but these instructions and the build script should work with all versions of LEAP and Tumbleweed both past and into the forseeable future provided the repositories and packages are available..

## Supported lc0 Backends

Although lc0 supports many backends including GPU computing, this guide currently describes using only the openBLAS backend for its lack of hardware dependencies. This means that this guide can be used to install on any openSUSE regardless of CPU or GPU, on physical or virtual machines. Anyone who wishes to build CUDA, OpenCL or Tensorflow backends are invited to contribute and modify this page.

## For Players who don't want to know the details, Just set up and Go!

Not all Players are Tech-heads, they just want something that will work ASAP without knowing the details. An Install Script is provided for you, which should have you playing in a very short period of time.

Summary of steps

* Compile the lc0 Engine
* Download the Networks file and place in the same folder as the compiled lc0 binary
* Move or Copy the folder containing the lc0 binary to a convenient location for setting up the graphical chessboard
* Configure the graphical chessboard pointing to the lc0 binary

#### Compile the lc0 Engine

Download and save the following file to your machine by clicking on the following link

[install_openSUSE_lc0.sh](install_openSUSE_lc0.sh)

The file you just downloaded can be run from any location, but must be executed in a root console.

Open a root console by first opening whatever terminal or console was installed in your Desktop, examples might be Xterm, Konsole, Qterminal, Gnome Terminal.
Then in your console type

`
su
`

For the following two commands and elsewhere below, the User can either copy the text into an open console or type the commands by hand.<br>

Change directory to where your downloaded script is (most likely your Downloads folder) and execute.
If necessary, modify the execute permission of the script with "chmod+x install_openSUSE_lc0.sh"

```

cd ~/Downloads
./install_openSUSE_lc0.sh
```


## RPM Packages

If you ran the script that builds from source, you can ignore this section and skip to the next section which describes installing a networks file. 

Otherwise, if you skipped most of what is described above because you want to use pre-built packages, this is where you should start!
1. Using an openSUSE provided Web browser (recommend Firefox), find the package for your version of openSUSE and download and/or install the package. The actual location of packages may change depending on project status, so you may also want to use the web package search at https://software.opensuse.org/package/lc0 and click on the "One-click install for your version of openSUSE.
2. Once the RPM file has been downloaded, you can install using YaST, zypper or RPM by simply pointing the install command to the file.
3. If successfully installed, you will find the lc0 binary at the following location and you can now proceed to the next section.
```

/usr/bin/lco
```



## The One Thing you must do Manually

#### Download and install the Networks file

The Install script automates practically everything needed for the lc0 Engine to run when hooked up to a graphical chessboard. The one thing missing that you, the User has to do on your own is to select a "Networks" file which contains the game data for its thinking. New data is generated continuously and players may want to try different files so this can't be automated. You need to select a file from the following page (usually under 50MBytes) and drop the file into the same folder as your lc0 binary.

[https://lczero.org/play/networks/bestnets/](https://lczero.org/play/networks/bestnets/)

## An example setup with the Arena graphical chessboard

The following applies if you compiled your own lc0 binary or if you are using the pre-built lc0 binary on a machine with an AMD processor. Special instructions to use Arena with the RPM is provided below, otherwise if you built from source Arena should "just work." An alternative is to use another graphical chessboard, [Cute Chess](https://cutechess.com/) has been tested and verified to work. Setting up Cute Chess is generally similar to setting up on Arena, with fewer options but generally the same major steps. If Users are unable to figure out how to set up with Cute Chess, a section will be added later.

Download the Arena Linux app from [Download Arena for Linux](http://www.playwitharena.com/?Download:Arena_for_Linux)

If your web browser offers to extract the archive, you should decline. Perform the following which installs in your User's Home Directory

```

mkdir ~/Arena
mv arenalinux_64bit_1.1.tar.gz ~/Arena/
cd ~/Arena/
tar -xf arenalinux_64bit_1.1.tar.gz
```
Now, a special command to reset permissions on ~/.configure/ so that a regular User account can write to this directory. After you run the following command as root, you will be able to run Arena as a regular User.


```

chown -R $USER:$(id -gn $USER) ~/.config
```



Now you can execute Arena as a normal, non-root User with the following command.

```

./Arena_x86_64_linux
```

#### Copying the lc0 binary to a location for Arena access

The above completes the installation of Arena, but does not hook it up to any Chess Engines. You can set up the Chess Engines that come with Arena, but the following describes how to set up lc0.
Assuming that your command console is still open and at the root of the Arena application, the next steps set up lc0 to connect to Arena

The "network file" in the following is the file you should have downloaded in the above section "The One thing you must do Manually"

If you built using the "install_openSUSE_lc0.sh" script, run the following
```

mkdir Engines/lc0
cp -r /opt/lc0/build/release/* Engines/lc0/
cp "network file" Engines/lc0/
```
If you installed using a RPM, run the following
```

mkdir Engines/lc0
cp /usr/bin/lc0 Engines/lc0/
cp "network file" Engines/lc0/
```



#### Configure Arena to point to the lc0 binary

Your files are now pre-positioned and ready for Arena.

Launch the Arena Engine Install Wizard with the F11 key. Or, if that doesn't work, then from the Arena menubar, 
Engines > Manage > Details tab > Installation Wizard button

Enter the information as required, pointing to your lc0 binary at

```

[Arena_root]/Engines/lc0/lc0
```

__RPM Package installs only__:

* In Arena, "Load" the lc0 engine
* In Arena, Press Ctl-1 or 
   Engines > Engine 1 > Configure
   Find the "backend" setting and select your preference, currently should be either opencl or blas
* While still in "Engine 1" start the engine

You should now be able to play!




## Removal / Uninstall

If RPMs were used to install lc0, you can remove using ordinary package management commands, eg the YaST Software Manager or the following command line
...


zypper rm lc0
...

The following describes removing files manually, either to verify total removal or if installed from source

The following commands remove parts of the install, the User can decide which to implement

The script installs lc0 files in /opt/lc0
So the following removes these files

```

rm -r /opt/lc0/
```

The script adds the following repositories. If you don't have something else using these repositories, they can be removed with the following command

```

zypper rr LEAP:devel:gcc LEAP:devel:languages:python:backports LEAP:science
```

The following packages were installed by the script and can be uninstalled. Note that uninstalling packages doesn't remove any dependencies of these packages, and libzl is excluded from the list because it's commonly used for other things.

```

zypper rm git gcc-c++ gcc7-c++ meson ninja python3-abseil openblas_pthreads-devel-static
```

## Re-install

Unless the Install script is interrupted, it's unlikely that the script has to be run again. If the script is run on a system more than once, non-critical errors (might look ugly, but will not harm a system) will be seen. Depending on the reason to run the Install script more than once, the User should consider either removing what has been installed (see Uninstall section) or commenting the lines in the Install script that would cause errors (primarily the Add Repo lines).

## Supplementary Information

Full Instructions connecting lc0 to graphical chessboards including Arena

[Running Leela Chess Zero in a Chess GUI"](https://github.com/LeelaChessZero/lc0/wiki/Running-Leela-Chess-Zero-in-a-Chess-GUI)

Original instructions for setting up lc0 on all platforms

[Getting Started](https://github.com/LeelaChessZero/lc0/wiki/Getting-Started)

Original compilation Instructions

[Build Instructions](https://github.com/LeelaChessZero/lc0)
