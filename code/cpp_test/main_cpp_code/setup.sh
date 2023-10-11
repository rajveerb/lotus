# #!/bin/sh
# rm -rf ./build
# mkdir build
# cp model_serialiser_tracing.py build/
# cp -r dataset build/
# cp -r imagenet build/
# cd build
# cmake -DCMAKE_PREFIX_PATH=/home/mayurpl/kexin_rong/github_code/ml-pipeline-benchmark/code/cpp_test/libtorch ..
# cmake --build . --config Release
# python3 model_serialiser_tracing.py
# ./main traced_resnet_model.pt



#!/usr/bin/env bash

# Set up libtorch
../setup_libtorch.sh

# Set up opencv
../setup_opencv.sh

rm -rf ./build
mkdir build
# cp model_serialiser_tracing.py build/
cp traced_resnet_model.pt build/
cp -r dataset build/
cp -r imagenet build/
cp info.txt build/
cd build

username=$(whoami)
base_dir="/data"

libtorch_install_dir="libtorch_main/libtorch"
libtorch_install_path="$base_dir/$username/$libtorch_install_dir"

opencv_install_dir="opencv_cpp_main/opencv_install"
opencv_install_path="$base_dir/$username/$opencv_install_dir"

export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

cmake -DCMAKE_PREFIX_PATH="$libtorch_install_path;$opencv_install_path" ..

cmake --build . --config Release

# python3 model_serialiser_tracing.py
./main traced_resnet_model.pt