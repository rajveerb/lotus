#!/bin/bash
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"
sudo apt update
sudo apt install -y --allow-downgrades intel-oneapi-vtune=2023.2.0-49484
sudo sed -i 's/kernel.yama.ptrace_scope = 1/kernel.yama.ptrace_scope = 0/' /etc/sysctl.d/10-ptrace.conf
sudo sysctl -w kernel.kptr_restrict=0
sudo sysctl -w kernel.perf_event_paranoid=0
sudo sysctl -w kernel.yama.ptrace_scope=0 
echo "Run below command"
echo "Enable vtune using below command:"
echo 'source /opt/intel/oneapi/setvars.sh'
echo 'sudo usermod -aG vtune $USER && newgrp vtune'
