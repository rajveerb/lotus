# !/bin/bash

mem_access_result_dir="/mydata/vtune_logs/pytorch_vtune_logs/vtune_mem_access_vary_dataloader"
custom_log_result_dir_="/mydata/pytorch_custom_log_and_vtune_one_epoch_imagenet_dataset"

python_path="/proj/prismgt-PG0/anaconda3/envs/torch2/bin/python3"
program_path="/mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/code/image_classification/code/pytorch_main.py"
dataset_path="/mydata/imagenet"

batch_size="1024"
num_gpu="4"
num_dataloaders=("8" "12" "16" "20" "24" "28")

num_epochs=1

# check if result directory exists
if [ ! -d ${mem_access_result_dir} ]; then
    mkdir -p ${mem_access_result_dir}
fi

# set below env variable to pin data loader threads to cores
# export TORCH_DATALOADER_PIN_CORE=1

for num_dataloader in "${num_dataloaders[@]}"
do
        custom_log_result_dir="${custom_log_result_dir_}/b${batch_size}_gpu${num_gpu}_dataloader${num_dataloader}"

        # check if result directory exists
        if [ ! -d ${custom_log_result_dir} ]; then
            mkdir -p ${custom_log_result_dir}
        fi
        vtune_result_dir=${mem_access_result_dir}/b${batch_size}_gpu${num_gpu}_dataloader${num_dataloader}
    vmtouch -e ${dataset_path}
    vtune -collect memory-access -data-limit 0 -result-dir ${vtune_result_dir} ${python_path} ${program_path} ${dataset_path} -b ${batch_size} --gpus ${num_gpu} -j ${num_dataloader} --epochs ${num_epochs} --log-train-file ${custom_log_result_dir}/custom_log --val-loop 0;
done