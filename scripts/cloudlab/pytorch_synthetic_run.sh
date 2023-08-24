# !/bin/bash

mem_access_result_dir_="/mydata/vtune_logs/pytorch_vtune_logs/vtune_vary_image_file_size_100Kfiles"

python_path="/proj/prismgt-PG0/anaconda3/envs/torch2/bin/python"

program_path="/proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py"

batch_size=128
num_gpu=4
num_epochs=1

file_sizes=("1" "10" "100")

memory_allocators=("" "tcmalloc" "jemalloc")
memory_allocator_paths=("" "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4" "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2")

numa_nodes=("0" "1")

for ((i=0;i<${#memory_allocators[@]};++i))
    do
        memory_allocator=${memory_allocators[i]}
        memory_allocator_path=${memory_allocator_paths[i]}
        mem_access_result_dir=${mem_access_result_dir_}/${memory_allocator}

        for file_size in "${file_sizes[@]}"
            do
                    # check if the result directory exists
                    if [ ! -d "${mem_access_result_dir}" ]; then
                        mkdir ${mem_access_result_dir}
                    fi

                    echo "Setting memory allocator to ${memory_allocator}"

                    dataset_path=/mydata/synthetic_data_${file_size}MBeach_100Kfiles_symlink/
                    result_dir_=${mem_access_result_dir}/filesize${file_size}MB_b${batch_size}_gpu${num_gpu}
                    LD_PRELOAD=${memory_allocator_path} vtune -collect memory-access -data-limit 0 -r ${result_dir_} ${python_path} ${program_path} ${dataset_path} -b ${batch_size} -j ${num_gpu} --epochs ${num_epochs} --val-loop 0;
                    # vtune -collect memory-access -quiet -r ${result_dir_} -data-limit 0 -finalization-mode none -source-search-dir $(dirname ${program_path}) -- ${python_path} ${program_path} ${dataset_path} -b ${batch_size} -j ${num_gpu} --epochs ${num_epochs} --val-loop 0;
                    # vtune -finalize -finalization-mode full -quiet -r ${result_dir_};
                    # vtune -limit 7 -format csv -csv-delimiter comma -report hotspots -quiet -group-by function -r ${result_dir_} -report-output ${result_dir_}/hotspots.csv
                    chmod -R 777 ${result_dir_}
                fi
            done
    done

for node in "${numa_nodes[@]}"
    do
        mem_access_result_dir=${mem_access_result_dir_}/numa_node_${node}

        if [ ! -d "${mem_access_result_dir}" ]; then
            mkdir ${mem_access_result_dir}
        fi

        for file_size in "${file_sizes[@]}"
            do
                dataset_path=/mydata/synthetic_data_${file_size}MBeach_100Kfiles_symlink/
                result_dir_=${mem_access_result_dir}/filesize${file_size}MB_b${batch_size}_gpu${num_gpu}
                vtune -collect memory-access -data-limit 0 -r ${result_dir_} numactl --cpunodebind=${node} --membind=${node} ${python_path} ${program_path} ${dataset_path} -b ${batch_size} -j ${num_gpu} --epochs ${num_epochs} --val-loop 0;
                chmod -R 777 ${result_dir_}
            done
    done

# Dataset: (10MB), vary batch size and fixed no. of batches
vtune -collect memory-access -data-limit 0 -r /mydata/vtune_logs/pytorch_vtune_logs/vary_batch_size_fixed_number_batches/filesize10MB_b128_gpu4 /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python /proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py /mydata/synthetic_data_10MBeach_4096files_symlink -b 128 -j 4 --epochs 1 --val-loop 0;
vtune -collect memory-access -data-limit 0 -r /mydata/vtune_logs/pytorch_vtune_logs/vary_batch_size_fixed_number_batches/filesize10MB_b256_gpu4 /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python /proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py /mydata/synthetic_data_10MBeach_100Kfiles_symlink -b 256 -j 4 --epochs 1 --val-loop 0;
