# Pending Tasks

Add functions to do the necessary preprocessing 

```
1. Crop
2. Rotate
3. Convert to tensor
4. normalize
```

Convert the python file to train the model completely in C++

# Setup

Run the setup_cuda.sh if cuda has not been set up yet - Yet to do

Run the setup_libtorch.sh to setup libtorch and be able to run a simple cpp program using libtorch - Yet to do

Manual Steps:

<https://pytorch.org/cppdocs/installing.html> - Follow the steps listed here to set up libtorch. Download the appropriate libtorch library for your distribution and cuda version. On Kepler2, the link used was https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.8.0%2Bcu101.zip

<https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local> - Setup Cuda to be able to train models on GPU if needed. This selection is for WSL but choose the necessary distribution and version. - On Kepler2, it was already installed and the version was 10.1
