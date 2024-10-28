# !/bin/bash

mem_access_result_dir="/mydata/iiswc24/lotus/vtune_mem_access_vary_dataloader"
custom_log_result_dir_="/mydata/iiswc24/lotus/lotustrace_vtune_result"

python_path=$(which python)
program_path="code/image_classification/code/pytorch_main.py"
dataset_path="/mydata/iiswc24/imagenet"

batch_size="1024"
num_gpu="4"
num_dataloaders=("20")

num_epochs=1

# check if result directory exists
if [ ! -d ${mem_access_result_dir} ]; then
    mkdir -p ${mem_access_result_dir}
fi

# set below env variable to pin data loader threads to cores
# export TORCH_DATALOADER_PIN_CORE=1

for num_dataloader in "${num_dataloaders[@]}"
do
    echo "LOL"
    custom_log_result_dir="${custom_log_result_dir_}/b${batch_size}_gpu${num_gpu}_dataloader${num_dataloader}"

    # check if result directory exists
    if [ ! -d ${custom_log_result_dir} ]; then
        mkdir -p ${custom_log_result_dir}
    fi
    vtune_result_dir=${mem_access_result_dir}/b${batch_size}_gpu${num_gpu}_dataloader${num_dataloader}
    # vmtouch -e ${dataset_path}
    echo "vtune -collect memory-access -data-limit 0 -result-dir ${vtune_result_dir} ${python_path} ${program_path} ${dataset_path} -b ${batch_size} --gpus ${num_gpu} -j ${num_dataloader} --epochs ${num_epochs} --log-train-file ${custom_log_result_dir}/lotustrace_log --val-loop 0;"
    summary_file=${custom_log_result_dir_}/e2e/summary_b${batch_size}_gpu${num_gpu}_dataloader${num_dataloader}.csv
    echo "vtune -report summary -report-output ${summary_file}  -format csv -csv-delimiter comma -r ${vtune_result_dir}"
done