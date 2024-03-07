# !/bin/bash

mem_access_result_dir="/mydata/vtune_logs/pytorch_vtune_logs/vtune_mem_access_unpinned"
python_path="/proj/prismgt-PG0/anaconda3/envs/torch2/bin/python3"
program_path="/mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/code/image_classification/code/pytorch_main.py"
dataset_path="/mydata/imagenet"

batch_sizes=("128" "256" "512" "1024")
num_gpus=("1" "2" "3" "4")

num_epochs=1

# check if result directory exists
if [ ! -d ${mem_access_result_dir} ]; then
    mkdir -p ${mem_access_result_dir}
fi

# set below env variable to pin data loader threads to cores
# export TORCH_DATALOADER_PIN_CORE=1

for batch_size in "${batch_sizes[@]}"
do
    for num_gpu in "${num_gpus[@]}"
    do
        # if batch size is 1024 and num_gpu is 1 then skip
        # Below is done due to CUDA out of memory error for V100 GPUs
        if [ ${batch_size} -eq 1024 ] && [ ${num_gpu} -eq 1 ]; then
            continue
        fi
        result_dir=${mem_access_result_dir}/b${batch_size}_gpu${num_gpu}
        vtune -collect memory-access -data-limit 0 -result-dir ${result_dir} ${python_path} ${program_path} ${dataset_path} -b ${batch_size} --gpus ${num_gpu} -j ${num_gpu} --epochs ${num_epochs} --val-loop 0;
    done
done

# unset below env variable to pin data loader threads to cores
# export TORCH_DATALOADER_PIN_CORE=0


# Below is not needed for now!

# mem_consumption_result_dir="/mydata/vtune_logs/pytorch_vtune_logs/vtune_mem_consumption/"
# hotspots_result_dir="/mydata/pytorch_vtune_logs/vtune_hotspots/"

# for batch_size in "${batch_sizes[@]}"
# do
#     for num_gpu in "${num_gpus[@]}"
#     do
#         result_dir_=${mem_consumption_result_dir}b${batch_size}_gpu${num_gpu}
#         vtune -collect memory-consumption -quiet -result-dir ${result_dir_} ${python_path} ${program_path} ${dataset_path} -b ${batch_size} --gpus ${num_gpu} -j ${num_gpu} --epochs ${num_epochs}
#     done
# done

# for batch_size in "${batch_sizes[@]}"
# do
#     for num_gpu in "${num_gpus[@]}"
#     do
#         declare result_dir_=${hotspots_result_dir}b${batch_size}_gpu${num_gpu}
#         vtune -collect hotspots -quiet -result-dir ${result_dir_} ${python_path} ${program_path} ${dataset_path} -b ${batch_size} --gpus ${num_gpu} -j ${num_gpu} --epochs ${num_epochs}
#         echo "[hotspots] Exited with code $? for batch_size=${batch_size} and num_gpu=${num_gpu}"
#     done
# done