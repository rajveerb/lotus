#!/bin/sh
rm -rf ./build
mkdir build
cp model_serialiser_tracing.py build/
cd build
cmake -DCMAKE_PREFIX_PATH=/home/mayurpl/kexin_rong/github_code/ml-pipeline-benchmark/code/cpp_test/libtorch ..
cmake --build . --config Release
python3 model_serialiser_tracing.py
./model_deserialiser traced_resnet_model.pt

