# !/bin/bash

result_dir_="/mydata/profiler_benchmark"
python_path="/proj/prismgt-PG0/anaconda3/envs/torch2/bin/python"
program_path="/proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py"
time_binary="/usr/bin/time"
dataset_dir="/mydata/austin_imagenet"

batch_sizes=("512")
num_gpus=("1")

num_epochs=1

time_format="wall(s),user(s),kernel(s)\n%e,%U,%S"
e2e_log_dir="${result_dir_}/e2e"

# check if result directory exists
if [ ! -d ${result_dir_} ]; then
    mkdir -p ${result_dir_}
fi

# check if e2e log directory exists
if [ ! -d ${e2e_log_dir} ]; then
    mkdir -p ${e2e_log_dir}
fi

# for batch_size in "${batch_sizes[@]}"
# do
#     for num_gpu in "${num_gpus[@]}"
#     do
#         result_dir=${result_dir_}/pytorch_profiler_imagenet_subset_b${batch_size}_gpu${num_gpu}
#         ${time_binary} --quiet --format=${time_format} -o "${e2e_log_dir}/pytorch_profiler_imagenet_subset_b${batch_size}_gpu${num_gpu}.log" ${python_path} ${program_path} ${dataset_dir} -b ${batch_size} -j ${num_gpu} --gpus ${num_gpu} --epochs ${num_epochs} --val-loop 0 --profile --profile-steps-all --profile-log-prefix ${result_dir};
#     done
# done

dataset_dir="/mydata/imagenet"

for batch_size in "${batch_sizes[@]}"
do
    for num_gpu in "${num_gpus[@]}"
    do
        result_dir=${result_dir_}/pytorch_profiler_b${batch_size}_gpu${num_gpu}
        ${time_binary} --quiet --format=${time_format} -o "${e2e_log_dir}/pytorch_profiler_b${batch_size}_gpu${num_gpu}.log" ${python_path} ${program_path} ${dataset_dir} -b ${batch_size} -j ${num_gpu} --gpus ${num_gpu} --epochs ${num_epochs} --val-loop 0 --profile --profile-steps-all --profile-log-prefix ${result_dir};
    done
done