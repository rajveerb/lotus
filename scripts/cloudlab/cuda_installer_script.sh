#!/bin/bash
sudo apt-get install linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
if [ ! -f /proj/prismgt-PG0/rbachkaniwala3/installation_binaries/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb ]; then
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb -P /proj/prismgt-PG0/rbachkaniwala3/installation_binaries/
fi
sudo dpkg -i /proj/prismgt-PG0/rbachkaniwala3/installation_binaries/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
# Below is for specific node
if [ "$(uname -n)" = "c4130-node.v100ubuntu20.prismgt-pg0.wisc.cloudlab.us" ]; then
    # Run below command due to https://groups.google.com/g/cloudlab-users/c/B6rNj7Vhltk/m/rwkHf_kwAgAJ
    sudo systemctl disable NetworkManager
    sudo systemctl unmask NetworkManager
    sudo systemctl disable NetworkManager-wait-online.service
    sudo ln -s /dev/null /etc/systemd/system/NetworkManager.service
    # Below command is used to comment out the nvidia driver in xorg.conf.d otherwise xorg will use GPU for display
    sudo sed -i 's/^/#/g' /usr/share/X11/xorg.conf.d/10-nvidia.conf
    # Some more issues relate to c4130 could be resolved by contacting cloudlab support check below link:
    # https://groups.google.com/g/cloudlab-users/c/ndz3XJCcIcg/m/kOa9iM4kAQAJ
    sudo systemctl mask gdm.service
    # Below command to refresh after updates to the xorg.conf.d
    sudo systemctl restart display-manager
fi
sudo apt-get update
echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc