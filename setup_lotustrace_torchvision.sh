# !/bin/bash
echo "Installing in $(which python)"

pushd code/LotusTrace
conda install -y cmake ninja
pip install -r requirements.txt
conda install -y mkl mkl-include
git submodule sync
git submodule update --init --recursive --depth 1
# Below command can cause issues
export CMAKE_PREFIX_PATH=$(dirname $(dirname $(which conda)))
echo "CMAKE_PREFIX_PATH is set to $CMAKE_PREFIX_PATH, it should be set to dir which contains the conda installation"  
sudo apt install -y g++
REL_WITH_DEB_INFO=1 MAX_JOBS=$(nproc) CC=/usr/bin/gcc-7 CXX=/usr/bin/g++-7 python setup.py install
popd
# Sanity check
pip list | grep "torch" | grep "2.0.0a0"


pushd code/torchvision
conda install -y -c conda-forge libjpeg-turbo
conda install -y pillow=10.3.0
python setup.py install
popd
echo "Finished!"