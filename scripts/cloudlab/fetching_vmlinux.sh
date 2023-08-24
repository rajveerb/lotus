#!/bin/bash
# echo "deb http://ddebs.ubuntu.com $(lsb_release -cs)-updates main restricted universe multiverse
# deb http://ddebs.ubuntu.com $(lsb_release -cs)-security main restricted universe multiverse
# deb http://ddebs.ubuntu.com $(lsb_release -cs)-proposed main restricted universe multiverse" | \
# sudo tee -a /etc/apt/sources.list.d/ddebs.list

# sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 428D7C01
# sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys C8CAB6595FDFF622
# sudo apt-get update
# sudo apt-get install linux-image-$(uname -r)-dbgsym

# # check if file does not exist /usr/lib/debug/boot/vmlinux-$(uname -r)
# if [ ! -f /usr/lib/debug/boot/vmlinux-$(uname -r) ]; then
#     # add log to file with date
#     echo "linux-image-$(uname -r)-dbgsym file not found!" > /proj/prismgt-PG0/rbachkaniwala3/system_startup_script_logs/log_$(date +%d-%m-%Y).txt
#     exit 1
# fi

# check if below two files exist

installation_dir="/proj/prismgt-PG0/rbachkaniwala3/installation_binaries"

if [ ! -f $installation_dir/linux-image-unsigned-5.4.0-139-generic-dbgsym_5.4.0-139.156_amd64.ddeb ]; then
    wget -P $installation_dir/ https://launchpad.net/~canonical-signing/+archive/ubuntu/primary-2022v1/+build/25511687/+files/linux-image-unsigned-5.4.0-139-generic-dbgsym_5.4.0-139.156_amd64.ddeb
fi

if [ ! -f $installation_dir/linux-image-5.4.0-139-generic-dbgsym_5.4.0-139.156_amd64.ddeb ]; then
    wget -P $installation_dir/ https://launchpad.net/~canonical-signing/+archive/ubuntu/primary-2022v1/+build/25511688/+files/linux-image-5.4.0-139-generic-dbgsym_5.4.0-139.156_amd64.ddeb
fi

sudo dpkg -i $installation_dir/linux-image-unsigned-5.4.0-139-generic-dbgsym_5.4.0-139.156_amd64.ddeb
sudo dpkg -i $installation_dir/linux-image-5.4.0-139-generic-dbgsym_5.4.0-139.156_amd64.ddeb 
sudo cp /usr/lib/debug/boot/vmlinux-5.4.0-139-generic /boot/