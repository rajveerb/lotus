#!/bin/bash
sudo apt-get install -y zlib1g
if [ ! -f /proj/prismgt-PG0/rbachkaniwala3/installation_binaries/cudnn-local-repo-ubuntu2004-8.7.0.84_1.0-1_amd64.deb ]; then
    wget https://developer.download.nvidia.com/compute/cudnn/secure/8.7.0/local_installers/11.8/cudnn-local-repo-ubuntu2004-8.7.0.84_1.0-1_amd64.deb -P /proj/prismgt-PG0/rbachkaniwala3/installation_binaries/
fi
sudo dpkg -i /proj/prismgt-PG0/rbachkaniwala3/installation_binaries/cudnn-local-repo-ubuntu2004-8.7.0.84_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install -y libcudnn8=8.7.0.84-1+cuda11.8
sudo apt-get install -y libcudnn8-dev=8.7.0.84-1+cuda11.8
# sudo rm /proj/prismgt-PG0/rbachkaniwala3/installation_binaries/cudnn-local-repo-ubuntu2004-8.7.0.84_1.0-1_amd64.deb