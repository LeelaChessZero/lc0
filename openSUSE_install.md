# Building lc0 on openSUSE

openSUSE is a popular RPM based Linux distro, the install sources can be downloaded from [https://software.opensuse.org](https://software.opensuse.org) .

The steps described here are minimal, enough to install and run lc0 on openSUSE. The reader is encouraged to skim the Supplementary Information after completing the Install where additional information can be found to download an icon to beautify and enter complete information about the lc0 engine into Arena. I doubt anyone would want to run on a 32-bit machine, but for these kinds of oddball installs, modifying the script should be next to trivial.

If the User finds anything in this guide unclear, the original documentation is linked under "Supplemental Information" in the last section. And, don't forget to recommend(as an Issue) or submit a change (as a Pull).

## Supported openSUSE versions

These instructions have been tested on current versions of LEAP 15.0 (The stable release) and Tumbleweed (The Development Rolling Release) on 64-bit machines but these instructions and the build script should work with all versions of LEAP and Tumbleweed both past and into the forseeable future provided the repositories and packages are available..

## Supported lc0 Backends

Although lc0 supports many backends including GPU computing, this guide currently describes using only the openBLAS backend for its lack of hardware dependencies. This means that this guide can be used to install on any openSUSE regardless of CPU or GPU, on physical or virtual machines. Anyone who wishes to build CUDA, OpenCL or Tensorflow backends are invited to contribute and modify this page.

## For Players who don't want to know the details, Just set up and Go!

Not all Players are Tech-heads, they just want something that will work ASAP without knowing the details. An Install Script is provided for you, which should have you playing in a very short period of time.

Summary of steps

* Compile the lc0 Engine
* Download the Networks file and place in the same folder as the compile lc0 binary
* Move or Copy the folder containing the lc0 binary to a convenient location for setting up the graphical chessboard
* Configure the graphical chessboard pointing to the lc0 binary

#### Compile the lc0 Engine

Download and save the following file to your machine by clicking on the following link

[Download install_openSUSE_lc0.sh](https://github.com/LeelaChessZero/lc0/install_openSUSE_lc0.sh)

The file you just downloaded can be run from any location, but must be executed in a root console.

Open a root console by first opening whatever terminal or console was installed in your Desktop, examples might be Xterm, Konsole, Qterminal, Gnome Terminal.
Then in your console type

`
su
`

For the following two commands and elsewhere below, the User can either copy the text into an open console or type the commands by hand.<br>

Change directory to where your downloaded script is (most likely your Downloads folder) and execute

```

cd ~/Downloads
./install_openSUSE_lc0
```

## The One Thing you must do Manually

#### Download and install the Networks file

The Install script automates practically everything needed for the lc0 Engine to run when hooked up to a graphical chessboard. The one thing missing that you, the User has to do on your own is to select a "Networks" file which contains the game data for its thinking. New data is generated continuously and players may want to try different files so this can't be automated. You need to select a file from the following page (usually under 50MBytes) and drop the file into the same folder as your lc0 binary.

[http://lczero.org/networks](http://lczero.org/networks/)

#### An example settup up with the Arena graphical chessboard

Download the Arena Linux app from [Download Arena for Linux](http://www.playwitharena.com/?Download:Arena_for_Linux)

If your web browser offers to extract the archive, you should decline. Perform the following which installs in your User's Home Directory

```

mkdir ~/Arena
mv arenalinux_64bit_1.1.tar.gz ~/Arena/
cd ~/Arena/
tar -xf arenalinux_64bit_1.1.tar.gz
```

If you were able to successfully create a Desktop shortcut to launch Arena, you can test it now, otherwise in the Arena root folder you can execute the following in a console

```

./Arena_x86_64_linux
```

#### Copying the lc0 binary to a location for Arena access

The above completes the installation of Arena, but does not hook it up to any Chess Engines. You can set up the Chess Engines that come with Arena, but the following describes how to set up lc0.
Assuming that your command console is still open and at the root of the Arena application, the next steps set up lc0 to connect to Arena

The "network file" is the file you should have downloaded in the above section "The One thing you must do Manually"

```

mkdir Engine/lc0
cp -r /opt/lc0/build/release/* Engines/lc0/
cp <i>"network file"</i> Engines/lc0/
```

#### Configure Arena to point to the lc0 binary

Your files are now pre-positioned and ready for Arena.

Launch the Arena Engine Install Wizard hit the F11 key. Or, if that doesn't work from the Arena menubar, Engines > Manage > Details tab > Installation Wizard button

Enter the information as required, pointing to your lc0 binary at

```

<i>Arena_root</i>/Engines/lc0/lc0
```

## Removal / Uninstall

The following commands remove parts of the install, the User can decide which to implement

The script installs lc0 files in /opt/lc0

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

Original instructions for setting lc0 on all platforms

[Getting Started](https://github.com/LeelaChessZero/lc0/wiki/Getting-Started)

Original compilation Instructions

[Build Instructions](https://github.com/LeelaChessZero/lc0)
