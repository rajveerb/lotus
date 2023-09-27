#!/usr/bin/env bash

curr_dir=$(pwd)

# Get the username using whoami
username=$(whoami)

# set base as /data
base_dir="/data"

# create a directory for open_cv
dir_name="libtorch_main"

# Combine the base directory and username to create the full path
folder_path="$base_dir/$username/$dir_name"

# Check if the folder already exists
if [ ! -d "$folder_path" ]; then
  # If it doesn't exist, create it
	mkdir -p "$folder_path"
	echo "Created directory: $folder_path"
	cd "$folder_path"
	echo $(pwd)
	wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip
	unzip *.zip
	rm *.zip
else
	echo "Directory already exists: $folder_path"
fi

cd $curr_dir