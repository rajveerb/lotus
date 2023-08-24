# !/bin/bash

mem_access_result_dir_="/mydata/vtune_logs/pytorch_vtune_logs/vtune_large_image_vary_batch_gpus"

python_path="/proj/prismgt-PG0/anaconda3/envs/torch2/bin/python"

program_path="/proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py"

num_epochs=1
file_sizes=("10" "100")
batch_sizes=("256")
num_gpus=("1" "2" "3" "4")

vtune -collect memory-access -data-limit 0 -r /mydata/vtune_logs/pytorch_vtune_logs/vtune_large_image_vary_batch_gpus/filesize100MB/b128_gpu4 ${python_path} ${program_path} /mydata/synthetic_data_100MBeach_8192files_symlink/ -b 128 -j 4 --epochs 1 --val-loop 0;
chmod -R 777 /mydata/vtune_logs/pytorch_vtune_logs/vtune_large_image_vary_batch_gpus/filesize100MB/b128_gpu4;

for batch in "${batch_sizes[@]}"
do
    for gpu in "${num_gpus[@]}"
    do
        for file_size in "${file_sizes[@]}"
            do
                    mem_access_result_dir=${mem_access_result_dir_}/filesize${file_size}MB
                    # check if the result directory exists
                    if [ ! -d "${mem_access_result_dir}" ]; then
                        mkdir -p ${mem_access_result_dir}
                    fi

                    dataset_path=/mydata/synthetic_data_${file_size}MBeach_8192files_symlink/
                    result_dir_=${mem_access_result_dir}/b${batch}_gpu${gpu}
                    vtune -collect memory-access -data-limit 0 -r ${result_dir_} ${python_path} ${program_path} ${dataset_path} -b ${batch} -j ${gpu} --epochs ${num_epochs} --val-loop 0;
                    chmod -R 777 ${result_dir_}
            done
    done
done

