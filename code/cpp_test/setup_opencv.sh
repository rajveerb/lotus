#!/usr/bin/env bash

# Get the username using whoami
username=$(whoami)

# set base as /data
base_dir="/data"

# create a directory for open_cv
dir_name="opencv_cpp_main"

# Combine the base directory and username to create the full path
folder_path="$base_dir/$username/$dir_name"

# Check if the folder already exists
if [ ! -d "$folder_path" ]; then
	# If it doesn't exist, create it
	mkdir -p "$folder_path"
	echo "Created directory: $folder_path"
	cd "$folder_path"
	echo $(pwd)
	git clone https://github.com/opencv/opencv.git
	git clone https://github.com/opencv/opencv_contrib.git
	cd opencv && mkdir build && cd build

	install_dir="opencv_install"
	install_path="$folder_path/$install_dir"

	cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX="$install_path" ..
	make -j8
	make install
else
	echo "Directory already exists: $folder_path"
fi

cd $curr_dir
