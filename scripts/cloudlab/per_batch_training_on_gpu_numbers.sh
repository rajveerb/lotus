# !/bin/bash

result_dir_="/mydata/per_batch_training_time_on_gpu"
python_path="/proj/prismgt-PG0/anaconda3/envs/torch2/bin/python"
program_path="/mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/code/image_classification/code/pytorch_main_batch_time_instrumentation.py"
dataset_dir="/mydata/imagenet"


batch_sizes=("128" "256" "512" "1024")
num_gpus=("1" "2" "3" "4")
num_epochs=1

# check if result directory exists
if [ ! -d ${result_dir_} ]; then
    mkdir -p ${result_dir_}
fi

for batch_size in "${batch_sizes[@]}"
do
    for num_gpu in "${num_gpus[@]}"
    do

        # if batch size is 1024 and num_gpu is 1 then skip
        # Below is done due to CUDA out of memory error for V100 GPUs
        if [ ${batch_size} -eq 1024 ] && [ ${num_gpu} -eq 1 ]; then
            continue
        fi

        result_file="${result_dir_}/b${batch_size}_gpu${num_gpu}.log"

        # check if result directory exists
        if [ ! -d ${result_dir} ]; then
            mkdir -p ${result_dir}
        fi

        ${python_path} ${program_path} ${dataset_dir} -b ${batch_size} --gpus ${num_gpu} -j ${num_gpu} --epochs ${num_epochs} --train-loop 10 --val-loop 0 --per-batch-train-time-log-file ${result_file};
    done
done