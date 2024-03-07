#  !/bin/bash

base_log_dir="/mydata/profiler_benchmark"
e2e_log_dir="${base_log_dir}/e2e"
# removed max rss becuase reported incorrectly
time_format="wall(s),user(s),kernel(s)\n%e,%U,%S"

# Binary paths
time_binary="/usr/bin/time"
python_binary="/proj/prismgt-PG0/anaconda3/envs/torch2/bin/python"
scalene_binary="/proj/prismgt-PG0/anaconda3/envs/torch2/bin/scalene"
py_spy_binary="/proj/prismgt-PG0/anaconda3/envs/torch2/bin/py-spy"
austin_binary="/proj/prismgt-PG0/anaconda3/envs/torch2/bin/austin"
austin2speedscope_binary="/proj/prismgt-PG0/anaconda3/envs/torch2/bin/austin2speedscope"

code_file="/mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/code/image_classification/code/pytorch_main.py"
dataset_dir="/mydata/imagenet/"
# austin's imagenet subset was created using `mkdir -p /mydata/austin_imagenet/train/ && cp -R /mydata/imagenet/train/n019* /mydata/austin_imagenet/train/`
austin_imagenet_subset_dir="/mydata/austin_imagenet/"
epochs=1
batch_size=512
num_gpus=1
num_workers=1

# check if base_log_dir exists
if [ -d "${base_log_dir}" ] 
then
    echo "Directory ${base_log_dir} exists." 
else
    mkdir -p ${base_log_dir}
    echo "Directory ${base_log_dir} created."
fi

# check if e2e_log_dir exists
if [ -d "${e2e_log_dir}" ] 
then
    echo "Directory ${e2e_log_dir} exists." 
else
    mkdir -p ${e2e_log_dir}
    echo "Directory ${e2e_log_dir} created."
fi

echo "py-spy results can be inaccurate and the profiler will log a warning suggesting 'behind in sampling, results may be inaccurate.' in which case the entire run's profile as well as time cannot be trusted"

echo "Running profiler benchmark"
echo "Running py-spy with batch size ${batch_size} and num_gpus ${num_gpus}"
${time_binary} --quiet --format=${time_format} -o "${e2e_log_dir}/py-spy_b${batch_size}_gpu${num_gpus}.log" ${py_spy_binary} record -s --format speedscope -o "${base_log_dir}/pyspy_b${batch_size}_gpu${num_gpus}.profile" -- ${python_binary} ${code_file} ${dataset_dir} -b ${batch_size} -j ${num_workers} --gpus ${num_gpus} --epochs ${epochs} --val-loop 0;

echo "Running custom instrumentation with batch size ${batch_size} and num_gpus ${num_gpus}"
# check if directory exists
if [ -d "${base_log_dir}/custom_log_profiles_b${batch_size}_gpu${num_gpus}" ] 
then
    echo "Directory ${base_log_dir}/custom_log_profiles_b${batch_size}_gpu${num_gpus} exists." 
else
    mkdir -p ${base_log_dir}/custom_log_profiles
    echo "Directory ${base_log_dir}/custom_log_profiles_b${batch_size}_gpu${num_gpus} created."
fi
${time_binary} --quiet --format=${time_format} -o "${e2e_log_dir}/custom_log_b${batch_size}_gpu${num_gpus}.log" ${python_binary} ${code_file} ${dataset_dir} -b ${batch_size} -j ${num_workers} --gpus ${num_gpus} --epochs ${epochs} --val-loop 0 --log-train-file "${base_log_dir}/custom_log_profiles_b${batch_size}_gpu${num_gpus}/custom_log";

echo "Running without profilers with batch size ${batch_size} and num_gpus ${num_gpus}"
${time_binary} --quiet --format=${time_format} -o "${e2e_log_dir}/no_profiler_b${batch_size}_gpu${num_gpus}.log" ${python_binary} ${code_file} ${dataset_dir} -b ${batch_size} -j ${num_workers} --gpus ${num_gpus} --epochs ${epochs} --val-loop 0;

echo "Running scalene with batch size ${batch_size} and num_gpus ${num_gpus}"
${time_binary} --quiet --format=${time_format} -o "${e2e_log_dir}/scalene_b${batch_size}_gpu${num_gpus}.log" $scalene_binary --cpu --profile-all --html --outfile "${base_log_dir}/imagenet_b512_b1_scalene_default_sampling_profile.html" ${code_file} --- ${dataset_dir} -b ${batch_size} -j ${num_workers} --gpus ${num_gpus} --epochs ${epochs} --val-loop 0;

echo "Running austin with batch size ${batch_size} and num_gpus ${num_gpus}"
${time_binary} --quiet --format=${time_format} -o "${e2e_log_dir}/austin_b${batch_size}_gpu${num_gpus}.log" ${austin_binary} -C --output="${base_log_dir}/austin_b${batch_size}_gpu${num_gpus}.profile" ${python_binary} ${code_file} ${dataset_dir} -b ${batch_size} -j ${num_workers} --gpus ${num_gpus} --epochs ${epochs} --val-loop 0;

echo "Converting austin profile to speedscope format [This will take some time]"
${austin2speedscope_binary} "${base_log_dir}/austin_b${batch_size}_gpu${num_gpus}.profile" "${base_log_dir}/austin_b${batch_size}_gpu${num_gpus}.speedscope.json"

echo "Running austin with batch size ${batch_size} and num_gpus ${num_gpus} with a subset of imagenet dataset"
${time_binary} --quiet --format=${time_format} -o "${e2e_log_dir}/austin_imagenet_subset_b${batch_size}_gpu${num_gpus}.log" ${austin_binary} -C --output="${base_log_dir}/austin_imagenet_subset_b${batch_size}_gpu${num_gpus}.profile" ${python_binary} ${code_file} ${austin_imagenet_subset_dir} -b ${batch_size} -j ${num_workers} --gpus ${num_gpus} --epochs ${epochs} --val-loop 0;

echo "Converting austin profile to speedscope format [This will take some time]"
${austin2speedscope_binary} "${base_log_dir}/austin_imagenet_subset_b${batch_size}_gpu${num_gpus}.profile" "${base_log_dir}/austin_imagenet_subset_b${batch_size}_gpu${num_gpus}.speedscope.json"
echo "Finished running austin2speedscope"

echo "Running custom instrumentation with batch size ${batch_size} and num_gpus ${num_gpus} with a subset of imagenet dataset"
# check if directory exists
if [ -d "${base_log_dir}/custom_log_profiles_imagenet_subset_b${batch_size}_gpu${num_gpus}" ] 
then
    echo "Directory ${base_log_dir}/custom_log_profiles_imagenet_subset_b${batch_size}_gpu${num_gpus} exists." 
else
    mkdir -p ${base_log_dir}/custom_log_profiles_imagenet_subset_b${batch_size}_gpu${num_gpus}
    echo "Directory ${base_log_dir}/custom_log_profiles_imagenet_subset_b${batch_size}_gpu${num_gpus} created."
fi
${time_binary} --quiet --format=${time_format} -o "${e2e_log_dir}/custom_log_imagenet_subset_b${batch_size}_gpu${num_gpus}.log" ${python_binary} ${code_file} ${austin_imagenet_subset_dir} -b ${batch_size} -j ${num_workers} --gpus ${num_gpus} --epochs ${epochs} --val-loop 0 --log-train-file "${base_log_dir}/custom_log_profiles_imagenet_subset_b${batch_size}_gpu${num_gpus}/custom_log";

echo "Running without profilers with batch size ${batch_size} and num_gpus ${num_gpus} with a subset of imagenet dataset"
${time_binary} --quiet --format=${time_format} -o "${e2e_log_dir}/no_profiler_imagenet_subset_b${batch_size}_gpu${num_gpus}.log" ${python_binary} ${code_file} ${austin_imagenet_subset_dir} -b ${batch_size} -j ${num_workers} --gpus ${num_gpus} --epochs ${epochs} --val-loop 0;

batch_size=1024
num_gpus=4
num_workers=4
echo "Running py-spy with batch size ${batch_size} and num_gpus ${num_gpus}"
${time_binary} --quiet --format=${time_format} -o "${e2e_log_dir}/py-spy_b${batch_size}_gpu${num_gpus}.log" ${py_spy_binary} record -s --format speedscope -o "${base_log_dir}/pyspy_b${batch_size}_gpu${num_gpus}.profile" -- ${python_binary} ${code_file} ${dataset_dir} -b ${batch_size} -j ${num_workers} --gpus ${num_gpus} --epochs ${epochs} --val-loop 0;

echo "Running custom instrumentation with batch size ${batch_size} and num_gpus ${num_gpus}"
# check if directory exists
if [ -d "${base_log_dir}/custom_log_profiles_b${batch_size}_gpu${num_gpus}" ] 
then
    echo "Directory ${base_log_dir}/custom_log_profiles_b${batch_size}_gpu${num_gpus} exists." 
else
    mkdir -p ${base_log_dir}/custom_log_profiles_b${batch_size}_gpu${num_gpus}
    echo "Directory ${base_log_dir}/custom_log_profiles_b${batch_size}_gpu${num_gpus} created."
fi
${time_binary} --quiet --format=${time_format} -o "${e2e_log_dir}/custom_log_b${batch_size}_gpu${num_gpus}.log" ${python_binary} ${code_file} ${dataset_dir} -b ${batch_size} -j ${num_workers} --gpus ${num_gpus} --epochs ${epochs} --val-loop 0 --log-train-file "${base_log_dir}/custom_log_profiles/custom_log_b${batch_size}_gpu${num_gpus}";

echo "Running without profilers with batch size ${batch_size} and num_gpus ${num_gpus}"
${time_binary} --quiet --format=${time_format} -o "${e2e_log_dir}/no_profiler_b${batch_size}_gpu${num_gpus}.log" ${python_binary} ${code_file} ${dataset_dir} -b ${batch_size} -j ${num_workers} --gpus ${num_gpus} --epochs ${epochs} --val-loop 0;
echo "Done"