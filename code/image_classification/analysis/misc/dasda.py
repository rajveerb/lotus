import subprocess, psutil, time, sys

# run a command and detach
cmd = 'taskset -c 0-24 /usr/bin/time --quiet --format="wall(s),user(s),kernel(s),max_rss(KB)\n%e,%U,%S,%M" python code/image_classification/code/pytorch_main.py /mydata/imagenet -b 1024 --gpus 4 -j 32 --val-loop 0 --epochs 1'

p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=subprocess.STDOUT, shell=True)

max_mem_usage = 0
#  while processes are running
while p.poll() is None:
    # get all child processes
    children = psutil.Process(p.pid).children(recursive=True)
    curr_mem_usage = 0
    for child in children:
        # get child process rss
        curr_mem_usage += child.memory_full_info().uss
    if curr_mem_usage > max_mem_usage:
        max_mem_usage = curr_mem_usage
        print([c.pid for c in children])
        print(max_mem_usage/1024/1024/1024, "GB")
    # sleep for 500 ms
    time.sleep(0.5)


