#!/bin/sh
rm -rf ./build
mkdir build
cp -r dataset build/
cp -r imagenet build/
cp python_test.py build/
cd build
cmake -DCMAKE_PREFIX_PATH=/home/mayurpl/kexin_rong/github_code/ml-pipeline-benchmark/code/cpp_test/libtorch ..
cmake --build . --config Release
./correctness_checker
python3 python_test.py

