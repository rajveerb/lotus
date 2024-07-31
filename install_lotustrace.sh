#!/bin/bash
echo "Installing in $(which python)"

pushd code/LotusTrace
conda install -y cmake ninja
pip install -r requirements.txt
conda install -y mkl mkl-include
conda install -c pytorch magma-cuda110 -y
git submodule sync
git submodule update --init --recursive --depth 1
# Below command can cause issues
export CMAKE_PREFIX_PATH=$(dirname $(dirname $(which conda)))
echo "CMAKE_PREFIX_PATH is set to $CMAKE_PREFIX_PATH, it should be set to dir which contains the conda installation"  
TORCH_CUDA_ARCH_LIST="8.6" REL_WITH_DEB_INFO=1 python setup.py install
popd
echo "Finished!"