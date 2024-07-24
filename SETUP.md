We provide the code/scripts to setup Lotus experiment in the cloudlab testbed using a c4130 node available in Wisconsin cluster.

The following instructions are for setting up/installing software dependencies on the c4130 node. The experiments are performed on the ImageNet dataset for the Image Classification task.

### Setup steps


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

8. Install anaconda from [here](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html):
    ```bash
    curl -O https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
    # Follow instructions (add Anaconda to `/mydata/iiswc24/anaconda3` when prompted)
    bash Anaconda3-2024.06-1-Linux-x86_64.sh
    source ~/.bashrc
    ```

9. Download and setup ImageNet dataset:
    ```bash
    bash scripts/cloudlab/download_imagenet.sh
    ```