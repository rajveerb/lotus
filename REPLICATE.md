We provide the code/scripts to replicate Lotus experiment results in the cloudlab testbed using a c4130 node available in Wisconsin cluster.

The following experiments are targetting an Intel Processor chip with 4x V100 GPUs. The experiments are performed on the ImageNet dataset for the Image Classification task. We focus on a single configuration for the below experiments because the same process/method can be applied to each of them. Please note that the figures generated via below experiments correspond to one configuration of the figures found in the paper.

We have setup software dependencies such as CUDA, CuDNN, Intel VTune, Anaconda, and ImageNet dataset on the c4130 node. You can check out the instructions to do so in the [SETUP.md](SETUP.md) file.

### Installation steps

1. Clone this repository
    ```git
    git clone --depth 1 git@github.com:rajveerb/lotus.git -b iiswc24ae
    cd lotus
    git submodule update --init --recursive
    ```

2. Create a conda environment
    ```bash
    conda create -n lotus python=3.10 -y
    conda activate lotus
    ```
3. Install **itt-python** using build instructions below:
    ```bash
    pushd code/itt-python
    export ITT_LIBRARY_DIR=/opt/intel/oneapi/vtune/latest/lib64/
    export ITT_INCLUDE_DIR=/opt/intel/oneapi/vtune/latest/include
    python setup.py install
    # Check if installed
    pip list | grep "itt"
    popd
    ```

4. Install PyTorch (LotusTrace):
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
    sudo apt install -y gcc-7 g++-7
    REL_WITH_DEB_INFO=1 MAX_JOBS=1 CC=/usr/bin/gcc-7 CXX=/usr/bin/g++-7 python setup.py install
    popd
    # Sanity check
    pip list | grep "torch" | grep "2.0.0a0"
    ```
5. Install torchvision:
    ```bash
    pushd code/torchvision
    conda install -y -c conda-forge libjpeg-turbo
    conda install -y pillow=10.3.0
    python setup.py install
    pip list | grep "torchvision" | grep "0.15.1a0"
    popd
    ```
    
6. Install below packages:
    ```bash
    conda install ipykernel pandas=2.0.3 -y
    pip install -y matplotlib==3.9.0 natsort==8.4.0 seaborn==0.13.2
    ```

### Experiment steps

1. Get the mapping logs for the preprocessing operations:
    ```bash
    # Activate VTune
    source /opt/intel/oneapi/setvars.sh
    bash code/image_classification/LotusMap/Intel/LotusMap.sh
    ```


2. Generate JSON file with mapping info by running all cells in [`code/image_classification/LotusMap/Intel/logsToMapping.ipynb`](code/image_classification/LotusMap/Intel/logsToMapping.ipynb)

3. You have successfully obtained the mapping ([`code/image_classification/LotusMap/Intel/mapping_funcs.json`](code/image_classification/LotusMap/Intel/mapping_funcs.json)) using **LotusMap** (Table 1)!

4. Run the Image Classification pipeline experiment where batch size and number of gpus are varied and LotusTrace is enabled:
    ```bash
    bash scripts/cloudlab/LotusTrace_imagenet.sh
    ```
    Note: # of DataLoader workers is equal to # of gpus in this experiment.

5. Run the below commands for observations in `High variance in Preprocessing Time` for fig 4 (a) and the statistics:
    ```bash
    python code/image_classification/analysis/LotusTrace_imagenet_vary_batch_and_gpu/preprocessing_time_stats.py\
     --remove_outliers\
     --data_dir lotustrace_result/512_gpu4/\
     --output_file lotustrace_result/preprocessing_time_stats.log 
    python code/image_classification/analysis/LotusTrace_imagenet_vary_batch_and_gpu/box_plot_preprocessing_time.py\
     --remove_outliers\
     --data_dir lotustrace_result/512_gpu4\
     --output_file lotustrace_result/box_plot_preprocessing_time.png
    ```
6. Run the below commands for observations in `Significant wait time` for fig 4 (b), (c) and the statistics:
    ```bash
    python code/image_classification/analysis/LotusTrace_imagenet_vary_batch_and_gpu/delay_and_wait_time_stats_and_plot.py\
     --sort_criteria duration\
     --data_dir lotustrace_result/b512_gpu4\
     --fig_dir lotustrace_result/figures\
     --output_file lotustrace_result/delay_and_wait_time_stats_and_plot.log
    ```
7. Run the visualization script for Fig 2:
    ```bash
    python code/visualize_LotusTrace/visualization_augmenter.py\
     --coarse\
     --lotustrace_trace_dir lotustrace_result/b512_gpu4\
     --custom_log_prefix lotustrace_log\
     --output_lotustrace_viz_file lotustrace_result/viz_file.lotustrace
    ```
    Open the file in chrome trace viewer for visualization (Navigate to `chrome://tracing` URL in Google Chrome, upload the `viz_file.lotustrace` and visualize the trace)

8. Run the below command for Image Classification pipeline to generate hardware performance numbers for Fig 5:
    ```bash
    bash scripts/cloudlab/LotusTrace_imagenet_vtune.sh
    ```

9. Follow the below steps to get a CSV of hw performance numbers (has to be performed manually):
    ```bash
    # Below step will provide a link, open a browser window, and login to the VTune GUI (set the password to anything you like)
    vtune-backend --web-port 8080 --data-directory ./vtune_mem_access_vary_dataloader/b1024_gpu4_dataloader20
    ```
    - Navigate to Microarchitecture Exploration tab

    - Perform grouping by Source Function / Function / Call Stack

    - Select all cells and paste it in a CSV file called `code/image_classification/analysis/combine_lotus/lotustrace_uarch/b1024_gpu4_dataloader20.csv`

10. Plot Fig 5 (a) by running `code/image_classification/analysis/combine_lotus/elapsed_time_plot.ipynb` notebook
    Check out the plot at the bottom of the notebook.

11. Plot Fig 5 (b) by running `code/image_classification/analysis/combine_lotus/per_python_func_plot_vary_dataloaders.ipynb` notebook
    Check out the plot at the bottom of the notebook.

12. Plot Fig 5 (c) by running below command:
    ```bash
    python code/image_classification/analysis/combine_lotus/hw_event_analyzer.py\
     --mapping_file code/image_classification/LotusMap/Intel/mapping_funcs.json\
     --uarch_dir code/image_classification/analysis/combine_lotus/lotustrace_uarch\
     --combined_hw_events code/image_classification/analysis/combine_lotus/combined_lotustrace_uarch.csv\
     --cpp_hw_events_plot_dir code/image_classification/analysis/combine_lotus/cpp_hw_events_figs
    ```
    Check out the `code/image_classification/analysis/combine_lotus/cpp_hw_events_figs` directory for the plots.

13. Plot Fig 5 (e)-(h) by running `code/image_classification/analysis/combine_lotus/c_to_python_analyser.ipynb` notebook
    Check out the plots in the `code/image_classification/analysis/combine_lotus/mapped_python_figs` directory.

14. That completes the experiment for LotusTrace on ImageNet dataset for Image Classification task!