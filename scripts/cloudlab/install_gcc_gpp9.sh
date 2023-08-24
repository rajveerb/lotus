#!/bin/bash
# Install g++9 and gcc9 and replace the symbolic link of gcc and g++
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y gcc-9
/usr/bin/gcc-9 --version
sudo apt install -y g++-9
/usr/bin/g++-9 --version
rm /usr/bin/gcc
rm /usr/bin/g++
ln -s /usr/bin/gcc-9 /usr/bin/gcc
ln -s /usr/bin/g++-9 /usr/bin/g++