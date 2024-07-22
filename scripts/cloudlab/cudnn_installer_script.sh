#!/bin/bash
sudo apt-get install -y zlib1g
cudnn_binary_dir="/mydata/iiswc24/installation_binaries"
tarball_link="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz"
wget $tarball_link -P $cudnn_binary_dir
echo "Downloaded the tarball for CuDNN installation!"
pushd $cudnn_binary_dir
tar xvf cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz
# below instructions are from https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-870/install-guide/index.html#installlinux-tar
sudo cp -r cudnn-linux-x86_64-8.7.0.84_cuda11-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-linux-x86_64-8.7.0.84_cuda11-archive/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
popd
echo "Finished installating CuDNN to the system!"