# !/bin/bash

result_dir_="/mydata/pytorch_profiles_imagenet_dataset"
python_path="/proj/prismgt-PG0/anaconda3/envs/torch2/bin/python"
program_path="/proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py"

batch_sizes=("128" "256" "512" "1024")
num_gpus=("1" "2" "3" "4")

num_epochs=1


# check if result directory exists
if [ ! -d ${result_dir_} ]; then
    mkdir ${result_dir_}
fi

for batch_size in "${batch_sizes[@]}"
do
    for num_gpu in "${num_gpus[@]}"
    do
        result_dir=${result_dir_}/b${batch_size}_gpu${num_gpu}
        TORCH_DATALOADER_PIN_CORE=1 ${python_path} ${program_path} /mydata/imagenet/ -b ${batch_size} -j ${num_gpu} --epochs ${num_epochs} --val-loop 0 --profile --profile-log-prefix ${result_dir};
    done
done