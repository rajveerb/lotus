We provide the code/scripts to replicate Lotus experiment results in the cloudlab testbed using a c4130 node available in Wisconsin cluster.

### Steps

1. Use our [`Lotus_c4130_cloudlab.profile`](Lotus_c4130_cloudlab.profile) as the geni script for cloudlab experiment profile. This profile uses a long term cloudlab dataset as a mounted remote filesystem with 916 GB storage. Mounted on `/mydata`.

2. Grow root partition of c4130 node on cloudlab (If this step is not performed, there will be no space left in root for installations):
    ```bash
    export RESIZEROOT=100
    sudo scripts/cloudlab/grow-rootfs.sh
    export RESIZEROOT=
    ```
3. Install Intel VTune:
    ```bash
    sudo bash scripts/cloudlab/install_vtune.sh
    # Make sure to run below commands 
    echo "source /opt/intel/oneapi/setvars.sh" > ~/.bashrc
    sudo usermod -aG vtune $USER && newgrp vtune
    ```
    Note: we used `Intel(R) VTune(TM) Profiler 2024.0.1 (build 627177)`
4. Install CUDA 11.8:
    ```bash
    sudo bash scripts/cloudlab/cuda_installer_script.sh
    ```
5. Install CuDNN 8.7.0:
    ```bash
    sudo bash scripts/cloudlab/cudnn_installer_script.sh
    ```
6. Reboot the machine:
    ```bash
    sudo reboot
    ```
7. Install [`vmtouch`](https://linux.die.net/man/8/vmtouch):
    ```bash
    sudo apt install vmtouch
    ```
8. Clone this repository!
9. Get submodules:

    ```git
    git submodule update --init --recursive
    ```
10. Install anaconda from [here](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html): 
11. Create a conda environment
    ```bash
    conda create -n lotus python=3.10 -y
    conda activate lotus
    ```
12. Install **itt-python** using build instructions below:
    ```bash
    pushd code/itt-python
    export ITT_LIBRARY_DIR=/opt/intel/oneapi/vtune/latest/lib64/
    export ITT_INCLUDE_DIR=/opt/intel/oneapi/vtune/latest/include
    python setup.py install
    # Check if installed
    pip list | grep "itt"
    popd
    ```

13. Install PyTorch (LotusTrace):
    ```bash
    pushd code/LotusTrace
    conda install -y cmake ninja
    pip install -r requirements.txt
    conda install -y mkl mkl-include
    git submodule sync
    git submodule update --init --recursive --depth 1
    # Below command can cause issues
    export CMAKE_PREFIX_PATH=$(dirname $(dirname $(which conda)))
    echo "CMAKE_PREFIX_PATH is set to $CMAKE_PREFIX_PATH, it should be set to dir which contains the conda installation"  
    sudo apt install -y g++
    REL_WITH_DEB_INFO=1 MAX_JOBS=1 python setup.py install
    popd
    # Sanity check
    pip list | grep "torch" | grep "2.0.0a0"
    ```
14. Install torchvision:
    ```bash
    pushd code/torchvision
    conda install -y -c conda-forge libjpeg-turbo
    conda install -y pillow=10.3.0
    python setup.py install
    pip list | grep "torchvision" | grep "0.15.1a0"
    popd
    ```
15. Get the mapping logs for the preprocessing operations:
    ```bash
    # Activate VTune
    source /opt/intel/oneapi/setvars.sh
    bash code/image_classification/LotusMap/Intel/LotusMap.sh
    ```
16. Generate JSON file with mapping info by running all cells in [`code/image_classification/LotusMap/Intel/logsToMapping.ipynb`](code/image_classification/LotusMap/Intel/logsToMapping.ipynb)
17. You have successfully obtained the mapping ([`code/image_classification/LotusMap/mapping_funcs.json`](code/image_classification/LotusMap/Intel/mapping_funcs.json)) using **LotusMap**!
18. Download ImageNet for next set of experiments:
    ```bash
    bash scripts/cloudlab/download_imagenet.sh
    ```
19. Run the experiment where batch size and number of gpus are varied and LotusTrace is enabled:
    ```bash
    bash scripts/cloudlab/LotusTrace_imagenet_vary_batch_and_gpu.sh
    ```
    Note: # of DataLoader workers is equal to # of gpus in this experiment.
20. Run the below commands for observations in `High variance in Preprocessing Time`:
    ```bash
    python code/image_classification/analysis/LotusTrace_imagenet_vary_batch_and_gpu/preprocessing_time_stats.py --remove_outliers
    python code/image_classification/analysis/LotusTrace_imagenet_vary_batch_and_gpu/iqr_and_stddev_preprocessing_time_stats.py --remove_outliers
    python code/image_classification/analysis/LotusTrace_imagenet_vary_batch_and_gpu/box_plot_preprocessing_time.py --remove_outliers
    ```
21. Run the below commands for observations in `Significant wait time`:
    ```bash
    python code/image_classification/analysis/LotusTrace_imagenet_vary_batch_and_gpu/delay_and_wait_time_stats_and_plot.py --sort_criteria duration
    ```