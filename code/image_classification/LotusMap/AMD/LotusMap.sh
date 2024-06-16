# !/bin/bash

program_path_prefix="/home/rbachkaniwala3/work/rajveerb_AMDProfileControl-python/AMD"
python_path="/home/rbachkaniwala3/work/anaconda3/envs/amduprof/bin/python"
programs=("Loader.py" "Normalize.py" "RandomHorizontalFlip.py" "RandomResizedCrop.py" "ToTensor.py" "Collation.py")
amduprof_record="AMDuProfCLI collect --config tbp --start-paused"
amduprof_report="AMDuProfCLI report"
csv_dir="/home/rbachkaniwala3/work/rajveerb_AMDProfileControl-python/AMD/logs"

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


amduprof_result_dir="/home/rbachkaniwala3/AMD"

# Running multiple times and taking "AND" operation of the reported function
total_runs=2

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
        $amduprof_record --output-dir $amduprof_result_dir $python_path $program
        for outputdir in $(ls $amduprof_result_dir) 
        do
            $amduprof_report --input-dir $amduprof_result_dir/$outputdir --report-output $csv_file --cutoff 100 -f csv
            rm -rf $amduprof_result_dir/$outputdir
        done
    done
done
chmod 777 -R $csv_dir
echo "Done running all programs"