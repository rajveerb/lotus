# !/bin/bash
echo "Running profiler benchmark with different profiler options"
echo "No profiler"
time -p /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python code/image_classification/code/pytorch_main.py /mydata/imagenet/ -j 4 --gpus 4 -b 1024 --epochs 1 --val-loop 0
echo "custom log"
time -p python code/image_classification/code/pytorch_main.py /mydata/imagenet/ --log-train-file /mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/custom_log/log -j 1 --gpus 1 -b 512 --epochs 1 --val-loop 0 &
echo "vanilla scalene"
time -p scalene --cpu --profile-all --html --outfile imagenet_b512_b1_scalene_default_sampling_profile.html code/image_classification/code/pytorch_main.py --- /mydata/imagenet/ -j 1 --gpus 1 -b 512 --epochs 1 --val-loop 0 &
echo "fast sampling scalene"
time -p scalene --cpu --profile-all --html --outfile imagenet_b512_b1_scalene_fast_sampling_profile.html --cpu-sampling-rate 0.0001 --cpu-percent-threshold 0.01 code/image_classification/code/pytorch_main.py --- /mydata/imagenet/ -j 1 --gpus 1 -b 512 --epochs 1 --val-loop 0 &
echo "Scalene with more than one dataloader worker"
time -p scalene --cpu --profile-all --html --outfile profile.html --- code/image_classification/code/pytorch_main.py /mydata/imagenet/ -j 4 --gpus 4 -b 1024 --epochs 1 --val-loop 0 &
echo "vtune run with smaller sampling interval"
vtune -collect memory-access -data-limit 0 -knob sampling-interval=0.01 -result-dir /mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/vtune_log /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python /mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/code/image_classification/code/pytorch_main.py /mydata/imagenet -b 1024 --gpus 4 -j 4 --epochs 1 --val-loop 0