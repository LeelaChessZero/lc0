#!/bin/bash


# This script can be run from any location, but must be run in a root console.

# Start OS detection and setting variables
ID=$(grep -w ID= /etc/os-release | sed -e 's/ID=//')
VERSION_ID=$(grep -w VERSION_ID= /etc/os-release | sed -e 's/VERSION_ID=//' | sed -e 's/"//g')

#If Tumbleweed is detected, the Tumbleweed install
if [ "$ID" = '"opensuse-tumbleweed"' ]; then

# Update system, always a good idea before a major install
zypper dup -y

echo "No repositories will be added"
# Tumbleweed repositories are currently disabled because of a Tumbleweed bug, but may not be significant due to the bleeding edge and rolling release nature of Tumbleweed. Currently, lc0 builds fine without these repos	
# zypper -n ar -f https://download.opensuse.org/repositories/devel:/gcc/openSUSE_Factory/ Tumbleweed:devel:gcc 
# zypper -n ar -f	https://download.opensuse.org/repositories/devel:/languages:/python:/Factory/openSUSE_Tumbleweed/ Tumbleweed:devel:languages:python
# zypper -n ar -f	http://download.opensuse.org/repositories/science/openSUSE_Tumbleweed/ openSUSE_Tumbleweed:science 

#Enable to access Tensorflow packages
# zypper -n ar -f https://download.opensuse.org/repositories/science:/machinelearning/openSUSE_Leap_$VERSION_ID/ Tumbleweed::science:machinelearning


else
#If LEAP is detected, then the LEAP install for that specific version of LEAP
if [ "$ID" = '"opensuse-leap"' ]; then

# Update system, always a good idea before a major install
zypper up -y

zypper -n ar -f https://download.opensuse.org/repositories/devel:/gcc/openSUSE_Leap_"$VERSION_ID"/ LEAP:devel:gcc 
zypper -n ar -f	https://download.opensuse.org/repositories/devel:/languages:/python:/backports/openSUSE_Leap_"$VERSION_ID"/ LEAP:devel:languages:python:backports
zypper -n ar -f	http://download.opensuse.org/repositories/science/openSUSE_Leap_"$VERSION_ID"/ LEAP:science 

#Enable to access Tensorflow packages
#zypper -n ar -f https://download.opensuse.org/repositories/science:/machinelearning/openSUSE_Leap_$VERSION_ID/ Tumbleweed::science:machinelearning

fi
# Package repositories are activated
zypper --gpg-auto-import-keys ref
fi


# install dependencies
# If any build dependencies aren't met, enable the following which installs the complete C/C++ Development environment
# zypper in -y -t pattern devel_C_C++ 

# The following dependencies are sometimes found, sometimes not found by the lc0 meson build, so are not installed by default
# When the following are not found already on the system, the lc0 build towards the end of this script will do its own install
# zypper in -y python3-protobuf libprotoc17 libprotobuf17 

zypper in -y --allow-vendor-change --allow-downgrade git gcc-c++ gcc7-c++ meson ninja python3-abseil openblas_pthreads-devel-static libz1



# Clone lc0 github repo
# The following clones the development branch.
# If you want to instead clone a stable release branch, visit the following URL in a web browser, select and copy the release you want, replacing the URL that follows "git clone" 5 lines below, but leaving the "/opt/lc0" at the end of the line untouched(In other words, everything between "https" and "git" inclusively)
# https://github.com/LeelaChessZero/lc0

# Note following installs into /opt. Modify location as desired
mkdir /opt/lc0
git clone https://github.com/LeelaChessZero/lc0.git /opt/lc0
cd /opt/lc0/ || exit


# Execute
./build.sh

# Final Message
echo "Completed. If you see no errors above, you can attach your chessboard to the lc0 binary at /opt/lc0/build/release/lc0"
echo "Or, follow the instructions for your specific chessboard frontend at"
echo "https://github.com/LeelaChessZero/lc0/wiki/Running-Leela-Chess-Zero-in-a-Chess-GUI"


