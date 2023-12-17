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

<!-- Directory Structure -->

The __setup_libtorch.sh__and __setup_opencv.sh__ script files help setup libtorch and opencv respectively. Set **base_dir** appropriately in each of the files and also in the __setup.sh__ inside __main_cpp_code__  directory

The config of the setup is
- cpp11
- cude 11.8

The __setup.sh__ file inside __main_cpp_code__ runs both the libtorcha nd opencv setup scripts and hence you can run that directly. Running the scripts multiple times is allowed and hence will not leave the system in a broken state.



### Update:

#### Completed Tasks:

Added functions to do the following preprocessing in C++
- [x] RandomResizedCrop
- [x] RandomHorizontalFlip
- [x] Tensor
- [x] Normalisation

Checked and confirmed the correctness of:
- [x] RandomResizedCrop
- [x] RandomHorizontalFlip
- [x] Tensor
- [x] Normalisation

#### Pending Tasks:

Train the model on python using the preprocessed data from C++