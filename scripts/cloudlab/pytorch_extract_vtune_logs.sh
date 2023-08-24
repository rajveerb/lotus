#!/bin/bash

batch_sizes=("128" "256" "512" "1024")
num_gpus=("1" "2" "3" "4")

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

# For not pinned case
mem_access_result_dir="/mydata/vtune_logs/pytorch_vtune_logs/vtune_mem_access_not_pinned"
hw_events_result_dir="/mydata/vtune_logs/pytorch_vtune_logs/csv/vtune_mem_access_not_pinned"

# check if hw_events_result_dir exists
if [ ! -d "$hw_events_result_dir" ]; then
    mkdir -p $hw_events_result_dir
fi

for batch_size in "${batch_sizes[@]}"
do
    for num_gpu in "${num_gpus[@]}"
    do
        declare result_dir_=${mem_access_result_dir}/b${batch_size}_gpu${num_gpu}
        declare csv_file_=${hw_events_result_dir}/b${batch_size}_gpu${num_gpu}.csv
        vtune -report hw-events -quiet -result-dir ${result_dir_} -format csv -csv-delimiter comma -report-output ${csv_file_}
        echo "[report hw-events] Exited with code $? for batch_size=${batch_size} and num_gpu=${num_gpu}"
        remove_first_line $csv_file_
    done
done

# For pinned case
mem_access_result_dir="/mydata/vtune_logs/pytorch_vtune_logs/vtune_mem_access_pinned"
hw_events_result_dir="/mydata/vtune_logs/pytorch_vtune_logs/csv/vtune_mem_access_pinned"

# check if hw_events_result_dir exists
if [ ! -d "$hw_events_result_dir" ]; then
    mkdir -p $hw_events_result_dir
fi

for batch_size in "${batch_sizes[@]}"
do
    for num_gpu in "${num_gpus[@]}"
    do
        declare result_dir_=${mem_access_result_dir}/b${batch_size}_gpu${num_gpu}
        declare csv_file_=${hw_events_result_dir}/b${batch_size}_gpu${num_gpu}.csv
        vtune -report hw-events -quiet -result-dir ${result_dir_} -format csv -csv-delimiter comma -report-output ${csv_file_}
        echo "[report hw-events] Exited with code $? for batch_size=${batch_size} and num_gpu=${num_gpu}"
        remove_first_line $csv_file_
    done
done

# Don't need to generate hotspots report for now!
# cpu_time_result_dir="/mydata/vtune_logs/pytorch_vtune_logs/??"

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