#!/bin/bash

result_dir="/mydata/lotustrace_result_priority"
e2e_log_dir="${result_dir}/e2e"
python_path=$(which python)
program_path="code/image_classification/code/pytorch_main.py"
time_binary="/usr/bin/time"
dataset_dir="/mydata/iiswc24/imagenet"

batch_sizes=("128" "256" "512" "1024")
num_gpus=("1" "2" "3" "4")
num_workers=("4" "8" "12" "16" "20" "24" "28")
num_epochs=1
time_format="wall(s),user(s),kernel(s)\n%e,%U,%S"

# check if result directory exists
if [ ! -d ${result_dir} ]; then
    mkdir -p ${result_dir}
fi

# check if e2e log directory exists
if [ ! -d ${e2e_log_dir} ]; then
    mkdir -p ${e2e_log_dir}
fi

for batch_size in "${batch_sizes[@]}"
do
    for num_gpu in "${num_gpus[@]}"
    do
        # if batch size is 1024 and num_gpu is 1 then skip
        # Below is done due to CUDA out of memory error for V100 GPUs
        if [ ${batch_size} -eq 1024 ] && [ ${num_gpu} -eq 1 ]; then
            echo "Skipping batch size 1024 and num_gpu 1 due to OOM"
            continue
        fi

        for worker in "${num_workers[@]}"
        do
            run_result_dir="${result_dir}/b${batch_size}_gpu${num_gpu}_w${worker}"
            # check if result directory exists
            if [ ! -d ${run_result_dir} ]; then
                mkdir -p ${run_result_dir}
            fi
            echo "Result for run with batch size ${batch_size}, num_gpu ${num_gpu}, num_workers ${worker} will be stored in ${run_result_dir}" 
            ${time_binary} --quiet --format=${time_format} -o "${e2e_log_dir}/lotustrace_log_b${batch_size}_gpu${num_gpu}_w${worker}.log" \
                ${python_path} ${program_path} ${dataset_dir} \
                -b ${batch_size} --gpus ${num_gpu} -j ${worker} --epochs ${num_epochs} \
                --log-train-file ${run_result_dir}/lotustrace_log \
                --val-loop 0 --gpu-idle-times ${result_dir}/gpu_idle_times \
                --gpu-util-times ${result_dir}/gpu_util_times
            echo "Finished running for batch size ${batch_size}, num_gpu ${num_gpu}, num_workers ${worker}"
            echo "Check ${e2e_log_dir}/lotustrace_log_b${batch_size}_gpu${num_gpu}_w${worker}.log for end to end time for this run"
        done
    done
done

echo "Finished running all experiments for lotustrace_imagenet_vary_batch_gpu_workers.sh"
