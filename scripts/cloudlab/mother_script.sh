#!/bin/bash
env RESIZEROOT=100 /proj/prismgt-PG0/rbachkaniwala3/scripts/grow-rootfs.sh
# update Import the debug symbol archive signing key
sudo apt install ubuntu-dbgsym-keyring
sudo apt update
# check if login user is rbachkaniwala3
me=$(whoami)
if me="rajveerb"
then
    echo "export GIT_SSH_COMMAND='ssh -i /proj/prismgt-PG0/rbachkaniwala3/ssh_keys/id_rsa'" >> /users/rajveerb/.bashrc
    echo "export GIT_SSH_COMMAND='ssh -i /proj/prismgt-PG0/rbachkaniwala3/ssh_keys/id_rsa'" >> /root/.bashrc
    echo "export GIT_SSH_COMMAND='ssh -i /proj/prismgt-PG0/rbachkaniwala3/ssh_keys/id_rsa'" >> /root/.profile
    echo "export GIT_SSH_COMMAND='ssh -i /proj/prismgt-PG0/rbachkaniwala3/ssh_keys/id_rsa'" >> /users/rajveerb/.profile
    # configure git
    git config --global user.email "46040700+rajveerb@users.noreply.github.com"
    git config --global user.name "rajveerb"
fi
# Need below because libc is not built with -fno-omit-frame-pointer. Check https://bugs.launchpad.net/ubuntu/+source/glibc/+bug/1908307.
# Use as `env LD_LIBRARY_PATH=/lib/libc6-prof/x86_64-linux-gnu ./prog prog_args`` 
# sudo apt-get install -y  libc6-prof
# bash /proj/prismgt-PG0/rbachkaniwala3/scripts/cuda_installer_script.sh
# bash /proj/prismgt-PG0/rbachkaniwala3/scripts/cudnn_installer_script.sh
# bash /proj/prismgt-PG0/rbachkaniwala3/scripts/install_gcc_gpp9.sh
# bash /proj/prismgt-PG0/rbachkaniwala3/scripts/install_vtune.sh
# bash /proj/prismgt-PG0/rbachkaniwala3/scripts/enable_conda.sh
# bash /proj/prismgt-PG0/rbachkaniwala3/scripts/fetching_vmlinux.sh