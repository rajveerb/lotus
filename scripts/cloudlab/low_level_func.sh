# !/bin/bash

program_path_prefix="/proj/prismgt-PG0/rbachkaniwala3/code/low_level_func"
python_path="/proj/prismgt-PG0/anaconda3/envs/torch2/bin/python"
programs=("convertRGB.py" "Normalize.py" "RandomHorizontalFlip.py" "RandomResizedCrop.py" "ToTensor.py")
vtune_record="vtune -collect hotspots -start-paused"
vtune_report="vtune -report hotspots"
csv_dir="/proj/prismgt-PG0/rbachkaniwala3/code/low_level_func/logs"

# check if all the above directories exist
if [ ! -d "$program_path_prefix" ]; then
    echo "Program path prefix does not exist"
    exit 1
fi

if [ ! -d "$csv_dir" ]; then
    echo "CSV directory does not exist"
    exit 1
fi

for program_ in "${programs[@]}"
do
    program=$program_path_prefix/$program_
    if [ ! -f "$program" ]; then
        echo "$program does not exist"
        exit 1
    fi
done


vtune_result_dir="/root/low_level_func"

# Running multiple times and taking "AND" operation of the reported function
total_runs=20

for run in $(seq 1 $total_runs)
do
    for program_ in "${programs[@]}"
    do
        program=$program_path_prefix/$program_
        # remove .py from program_
        result_dir=$csv_dir/${program_::-3}
        # check if result_dir exists
        if [ ! -d "$result_dir" ]; then
            mkdir $result_dir
        fi
        csv_file=$result_dir/$run.csv
        echo "Running $program for run $run"
        $vtune_record -result-dir $vtune_result_dir -- $python_path $program
        $vtune_report -result-dir $vtune_result_dir -format csv -csv-delimiter comma -report-output $csv_file
        rm -rf $vtune_result_dir
    done
done
chmod 777 -R $csv_dir
echo "Done running all programs"

# vtune -collect hotspots -start-paused -result-dir ~/collation_tester_v2 -- /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python rbachkaniwala3/code/collation_tester.py
# chmod 777 -R /proj/prismgt-PG0/rbachkaniwala3/code/low_level_func/
# vtune -report hotspots -result-dir /root/collation_tester_v2/ -format csv -csv-delimiter comma -report-output ./low_level_func_v2.csv