#!/bin/bash
sudo apt-get install -y zlib1g
cudnn_binary_dir="/mydata/Lotus/installation_binaries"

# if [ ! -f $cudnn_binary_dir/cudnn-local-repo-ubuntu2004-8.7.0.84_1.0-1_amd64.deb ]; then
#     wget https://developer.download.nvidia.com/compute/cudnn/secure/8.7.0/local_installers/11.8/cudnn-local-repo-ubuntu2004-8.7.0.84_1.0-1_amd64.deb -P $cudnn_binary_dir
# fi
# sudo dpkg -i $cudnn_binary_dir/cudnn-local-repo-ubuntu2004-8.7.0.84_1.0-1_amd64.deb
# sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
# sudo apt-get update
# sudo apt-get install -y libcudnn8=8.7.0.84-1+cuda11.8
# sudo apt-get install -y libcudnn8-dev=8.7.0.84-1+cuda11.8


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