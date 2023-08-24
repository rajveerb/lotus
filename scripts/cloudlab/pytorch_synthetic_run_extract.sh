#!/bin/bash
mem_access_result_dir="/mydata/vtune_logs/pytorch_vtune_logs/vtune_vary_image_file_size_100Kfiles"
hw_events_result_dir="/mydata/vtune_logs/pytorch_vtune_logs/vtune_hw_events_vary_image_file_size_100Kfiles"

# check if the directory exists
if [ ! -d "${hw_events_result_dir}" ]; then
    mkdir ${hw_events_result_dir}
fi


# vtune report generated have a first line which says "war:Column filter is ON."
# if left as it is, the flame graph will not be generated
# this function removes the first line from the csv file
function remove_first_line() {
    # read first line of a file
    first_line=$(sudo head -n 1 $1)

    # check if the first line contains "war:Column filter is ON."
    if [[ $first_line == *"war:Column filter is ON."* ]]; then
        echo "Column filter is ON"
        # remove the first line from the file
        sudo sed -i '1d' $1
    fi
}

result_dirs=("filesize1MB_b1024_gpu4" "filesize10MB_b1024_gpu4" "filesize100MB_b1024_gpu4")


for result_dir in "${result_dirs[@]}"
do
    declare csv_file_=${hw_events_result_dir}/${result_dir}.csv
    vtune -report hw-events -quiet -result-dir ${mem_access_result_dir}/${result_dir} -format csv -csv-delimiter comma -report-output ${csv_file_}
    echo "[report hw-events] Exited with code $? for result_dir=${result_dir}"
    remove_first_line $csv_file_
done


# for batch_size in "${batch_sizes[@]}"
# do
#     for num_gpu in "${num_gpus[@]}"
#     do
#         declare result_dir_=${mem_access_result_dir}b${batch_size}_gpu${num_gpu}
#         declare csv_file_=${hw_events_result_dir}/b${batch_size}_gpu${num_gpu}.csv
#         vtune -report hw-events -quiet -result-dir ${result_dir_} -format csv -csv-delimiter comma -report-output ${csv_file_}
#         echo "[report hw-events] Exited with code $? for batch_size=${batch_size} and num_gpu=${num_gpu}"
#         remove_first_line $csv_file_
#     done
# done

# for batch_size in "${batch_sizes[@]}"
# do
#     for num_gpu in "${num_gpus[@]}"
#     do
#         declare result_dir_=${mem_access_result_dir}b${batch_size}_gpu${num_gpu}
#         declare csv_file_=${cpu_time_result_dir}b${batch_size}_gpu${num_gpu}.csv
#         vtune -report hotspots -quiet -result-dir ${result_dir_} --column="CPU Time","Module","Function (Full)" -format csv -csv-delimiter comma -report-output ${csv_file_}
#         echo "[report hotspots] Exited with code $? for batch_size=${batch_size} and num_gpu=${num_gpu}"
#         remove_first_line $csv_file_
#     done
# done