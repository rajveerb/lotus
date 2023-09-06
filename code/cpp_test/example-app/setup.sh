#!/bin/sh
rm -rf ./build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=../libtorch ..
cmake --build . --config Release