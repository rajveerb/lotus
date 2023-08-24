# !/bin/bash

result_dir_="/mydata/pytorch_profiles_synthetic_dataset"

python_path="/proj/prismgt-PG0/anaconda3/envs/torch2/bin/python"

program_path="/proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py"

batch_size=128
num_gpu=4
num_epochs=1

file_sizes=("1" "10" "100")

memory_allocators=("glibc" "tcmalloc" "jemalloc")
memory_allocator_paths=("" "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4" "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2")

for ((i=0;i<${#memory_allocators[@]};++i))
    do
        memory_allocator=${memory_allocators[i]}
        memory_allocator_path=${memory_allocator_paths[i]}
        mem_result_dir=${result_dir_}/${memory_allocator}

        for file_size in "${file_sizes[@]}"
            do
                    # check if the result directory exists
                    if [ ! -d "${mem_result_dir}" ]; then
                        mkdir -p ${mem_result_dir}
                    fi

                    echo "Setting memory allocator to ${memory_allocator}"

                    dataset_path=/mydata/synthetic_data_${file_size}MBeach_8192files_symlink/
                    result_dir=${mem_result_dir}/filesize${file_size}MB_b${batch_size}_gpu${num_gpu}
                    LD_PRELOAD=${memory_allocator_path} ${python_path} ${program_path} ${dataset_path} -b ${batch_size} -j ${num_gpu} --epochs ${num_epochs} --val-loop 0 --profile --profile-log-prefix ${result_dir};
            done
    done

numa_nodes=("0" "1")

for node in "${numa_nodes[@]}"
    do
        numa_result_dir=${result_dir_}/numa_node_${node}

        if [ ! -d "${numa_result_dir}" ]; then
            mkdir -p ${numa_result_dir}
        fi

        for file_size in "${file_sizes[@]}"
            do
                dataset_path=/mydata/synthetic_data_${file_size}MBeach_8192files_symlink/
                result_dir=${numa_result_dir}/filesize${file_size}MB_b${batch_size}_gpu${num_gpu}
                numactl --cpunodebind=${node} --membind=${node} ${python_path} ${program_path} ${dataset_path} -b ${batch_size} -j ${num_gpu} --epochs ${num_epochs} --val-loop 0 --profile --profile-log-prefix ${result_dir};
            done
    done

# Dataset: (10MB), vary batch size and fixed no. of batches
batch_size=128
result_dir=${result_dir_}/vary_batch_size_fixed_num_batches/filesize10MB_b${batch_size}_gpu${num_gpu}
${python_path} ${program_path} /mydata/synthetic_data_10MBeach_4096files_symlink -b ${batch_size} -j ${num_gpu} --epochs ${num_epochs} --val-loop 0 --profile --profile-log-prefix ${result_dir};
batch_size=256
result_dir=${result_dir_}/vary_batch_size_fixed_num_batches/filesize10MB_b${batch_size}_gpu${num_gpu}
${python_path} ${program_path} /mydata/synthetic_data_10MBeach_8192files_symlink -b ${batch_size} -j ${num_gpu} --epochs ${num_epochs} --val-loop 0 --profile --profile-log-prefix ${result_dir};