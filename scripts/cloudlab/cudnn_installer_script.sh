#!/bin/bash
sudo apt-get install -y zlib1g
cudnn_binary_dir="/mydata/P3Tracer/installation_binaries/"
if [ ! -f $cudnn_binary_dir/cudnn-local-repo-ubuntu2004-8.7.0.84_1.0-1_amd64.deb ]; then
    wget https://developer.download.nvidia.com/compute/cudnn/secure/8.7.0/local_installers/11.8/cudnn-local-repo-ubuntu2004-8.7.0.84_1.0-1_amd64.deb -P $cudnn_binary_dir
fi
sudo dpkg -i $cudnn_binary_dir/cudnn-local-repo-ubuntu2004-8.7.0.84_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install -y libcudnn8=8.7.0.84-1+cuda11.8
sudo apt-get install -y libcudnn8-dev=8.7.0.84-1+cuda11.8