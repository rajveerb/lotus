# Setup

<!-- Directory Structure -->

The **setup_libtorch.sh** and **setup_opencv.sh** script files help setup libtorch and opencv respectively. Set **base_dir** appropriately in each of the files and also in the **setup.sh** inside **main_cpp_code**  directory. The script also uses commands like `whoami`. Do check if those need to be updated.

The config of the setup is
- cpp11
- cude 11.8

The **setup.sh** file inside **main_cpp_code** runs both the libtorch and opencv setup scripts and hence you can run that directly. Running the scripts multiple times is allowed and hence will not leave the system in a broken state.

I'll clean the directory in the next couple of days and add comments in appropriate locations

Before running the code, update the dataset path in **main.cpp** to the appropriate location of imagenet dataset

Once setup, a **build** directory is created inside **main_cpp_code** which contains the executable name ***main***. Run the executable as `./main`. 

For quicker development time, once compiled using setup.sh, run `make` inside the build directory to compile the most recent version of the code. You do not need to run **setup.sh** all over again