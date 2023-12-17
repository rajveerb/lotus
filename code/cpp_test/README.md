# Setup

<!-- Directory Structure -->

The **setup_libtorch.sh** and **setup_opencv.sh** script files help setup libtorch and opencv respectively. Set **base_dir** appropriately in each of the files and also in the **setup.sh** inside **main_cpp_code**  directory

The config of the setup is
- cpp11
- cude 11.8

The **setup.sh** file inside **main_cpp_code** runs both the libtorch and opencv setup scripts and hence you can run that directly. Running the scripts multiple times is allowed and hence will not leave the system in a broken state.