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
    ```bash
    curl -O https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
    # Follow instructions (add Anaconda to `/mydata/iiswc24/anaconda3` when prompted)
    bash Anaconda3-2024.06-1-Linux-x86_64.sh
    source ~/.bashrc
    ```

11. Download and setup ImageNet dataset:
    ```bash
    bash scripts/cloudlab/download_imagenet.sh
    ```

12. Create a conda environment
    ```bash
    conda create -n lotus python=3.10 -y
    conda activate lotus
    ```
13. Install **itt-python** using build instructions below:
    ```bash
    pushd code/itt-python
    export ITT_LIBRARY_DIR=/opt/intel/oneapi/vtune/latest/lib64/
    export ITT_INCLUDE_DIR=/opt/intel/oneapi/vtune/latest/include
    python setup.py install
    # Check if installed
    pip list | grep "itt"
    popd
    ```

14. Install PyTorch (LotusTrace):
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
15. Install torchvision:
    ```bash
    pushd code/torchvision
    conda install -y -c conda-forge libjpeg-turbo
    conda install -y pillow=10.3.0
    python setup.py install
    pip list | grep "torchvision" | grep "0.15.1a0"
    popd
    ```
16. Get the mapping logs for the preprocessing operations:
    ```bash
    # Activate VTune
    source /opt/intel/oneapi/setvars.sh
    bash code/image_classification/LotusMap/Intel/LotusMap.sh
    ```
17. Install below packages:
    ```bash
    conda install ipykernel pandas=2.0.3 -y
    pip install -y matplotlib==3.9.0 natsort==8.4.0
    ```
18. Generate JSON file with mapping info by running all cells in [`code/image_classification/LotusMap/Intel/logsToMapping.ipynb`](code/image_classification/LotusMap/Intel/logsToMapping.ipynb)

19. You have successfully obtained the mapping ([`code/image_classification/LotusMap/Intel/mapping_funcs.json`](code/image_classification/LotusMap/Intel/mapping_funcs.json)) using **LotusMap** (Table 1)!

20. Run the experiment where batch size and number of gpus are varied and LotusTrace is enabled:
    ```bash
    bash scripts/cloudlab/LotusTrace_imagenet.sh
    ```
    Note: # of DataLoader workers is equal to # of gpus in this experiment.
21. Run the below commands for observations in `High variance in Preprocessing Time` (fig 4 (a) and the statistics):
    ```bash
    python code/image_classification/analysis/LotusTrace_imagenet_vary_batch_and_gpu/preprocessing_time_stats.py\
     --remove_outliers\
     --data_dir lotustrace_result/b1024_gpu4/\
     --output_file lotustrace_result/preprocessing_time_stats.log 
    python code/image_classification/analysis/LotusTrace_imagenet_vary_batch_and_gpu/box_plot_preprocessing_time.py\
     --remove_outliers\
     --data_dir lotustrace_result/b1024_gpu4\
     --output_file lotustrace_result/box_plot_preprocessing_time.png
    ```
22. Run the below commands for observations in `Significant wait time` (fig 4 (b), (c) and the statistics):
    ```bash
    python code/image_classification/analysis/LotusTrace_imagenet_vary_batch_and_gpu/delay_and_wait_time_stats_and_plot.py\
     --sort_criteria duration\
     --data_dir lotustrace_result/b512_gpu4\
     --fig_dir lotustrace_result/figures\
     --output_file lotustrace_result/delay_and_wait_time_stats_and_plot.log
    ```
23. Run the visualization script (Figure 2):
    ```bash
    python code/visualize_LotusTrace/visualization_augmenter.py\
     --coarse\
     --lotustrace_trace_dir lotustrace_result/b512_gpu4\
     --custom_log_prefix lotustrace_log\
     --output_lotustrace_viz_file lotustrace_result/viz_file.lotustrace
    ```
    Open the file in chrome trace viewer for visualization (Navigate to `chrome://tracing` URL in Google Chrome, upload the `viz_file.lotustrace` and visualize the trace)

24. Run the below to generate hardware performance numbers (Figure 5):
    ```bash
    bash scripts/cloudlab/LotusTrace_imagenet_vtune.sh
    ```
25. Plot (Fig 5 (a)) by running `code/image_classification/analysis/combine_lotus/elapsed_time_plot.ipynb` notebook

26. Plot (Fig 5 (b)) by running `code/image_classification/analysis/combine_lotus/per_python_func_plot_vary_dataloaders.ipynb` notebook

27. 
