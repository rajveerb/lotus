# !/bin/bash

# logs are stored in /mydata/rbachkaniwala3/code/e2e_time.log

batch_size=128
gpus=4
epochs=1
train_loop=16
log_dir=/mydata/rbachkaniwala3/code/e2e_time_logs
jemalloc_path=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
tcmalloc_path=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

# check if log_dir exists
if [ ! -d "$log_dir" ]; then
  mkdir $log_dir
fi

# malloc
echo "malloc: 1 MB, 128 batch_size, 4 gpus, $train_loop" > $log_dir/e2e_time.log;
/usr/bin/time -f "Maximum resident set size (kbytes): %M" /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python /proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py /mydata/synthetic_data_1MBeach_8192files_symlink/ -b $batch_size -j $gpus --epochs $epochs --train-loop $train_loop --val-loop 0 --log-train-file $log_dir/b128_gpu4_1MBeach_malloc.log >> $log_dir/e2e_time.log 2>&1;
echo "malloc: 10 MB, 128 batch_size, 4 gpus, $train_loop" >> $log_dir/e2e_time.log;
/usr/bin/time -f "Maximum resident set size (kbytes): %M" /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python /proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py /mydata/synthetic_data_10MBeach_8192files_symlink/ -b $batch_size -j $gpus --epochs $epochs --train-loop $train_loop --val-loop 0 --log-train-file $log_dir/b128_gpu4_10MBeach_malloc.log >> $log_dir/e2e_time.log 2>&1;
echo "malloc: 100 MB, 128 batch_size, 4 gpus, $train_loop" >> $log_dir/e2e_time.log;
/usr/bin/time -f "Maximum resident set size (kbytes): %M" /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python /proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py /mydata/synthetic_data_100MBeach_8192files_symlink/ -b $batch_size -j $gpus --epochs $epochs --train-loop $train_loop --val-loop 0 --log-train-file $log_dir/b128_gpu4_100MBeach_malloc.log >> $log_dir/e2e_time.log 2>&1;

# jemalloc
echo "jemalloc: 1 MB, 128 batch_size, 4 gpus, $train_loop" >> $log_dir/e2e_time.log;
LD_PRELOAD=$jemalloc_path /usr/bin/time -f "Maximum resident set size (kbytes): %M" /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python /proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py /mydata/synthetic_data_1MBeach_8192files_symlink/ -b $batch_size -j $gpus --epochs $epochs --train-loop $train_loop --val-loop 0 --log-train-file $log_dir/b128_gpu4_1MBeach_jemalloc.log >> $log_dir/e2e_time.log 2>&1;
echo "jemalloc: 10 MB, 128 batch_size, 4 gpus, $train_loop" >> $log_dir/e2e_time.log;
LD_PRELOAD=$jemalloc_path /usr/bin/time -f "Maximum resident set size (kbytes): %M" /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python /proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py /mydata/synthetic_data_10MBeach_8192files_symlink/ -b $batch_size -j $gpus --epochs $epochs --train-loop $train_loop --val-loop 0 --log-train-file $log_dir/b128_gpu4_10MBeach_jemalloc.log >> $log_dir/e2e_time.log 2>&1;
echo "jemalloc: 100 MB, 128 batch_size, 4 gpus, $train_loop" >> $log_dir/e2e_time.log;
LD_PRELOAD=$jemalloc_path /usr/bin/time -f "Maximum resident set size (kbytes): %M" /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python /proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py /mydata/synthetic_data_100MBeach_8192files_symlink/ -b $batch_size -j $gpus --epochs $epochs --train-loop $train_loop --val-loop 0 --log-train-file $log_dir/b128_gpu4_100MBeach_jemalloc.log >> $log_dir/e2e_time.log 2>&1;

# tcmalloc
echo "tcmalloc: 1 MB, 128 batch_size, 4 gpus, $train_loop" >> $log_dir/e2e_time.log;
LD_PRELOAD=$tcmalloc_path /usr/bin/time -f "Maximum resident set size (kbytes): %M" /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python /proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py /mydata/synthetic_data_1MBeach_8192files_symlink/ -b $batch_size -j $gpus --epochs $epochs --train-loop $train_loop --val-loop 0 --log-train-file $log_dir/b128_gpu4_1MBeach_tcmalloc.log >> $log_dir/e2e_time.log 2>&1;
echo "tcmalloc: 10 MB, 128 batch_size, 4 gpus, $train_loop" >> $log_dir/e2e_time.log;
LD_PRELOAD=$tcmalloc_path /usr/bin/time -f "Maximum resident set size (kbytes): %M" /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python /proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py /mydata/synthetic_data_10MBeach_8192files_symlink/ -b $batch_size -j $gpus --epochs $epochs --train-loop $train_loop --val-loop 0 --log-train-file $log_dir/b128_gpu4_10MBeach_tcmalloc.log >> $log_dir/e2e_time.log 2>&1;
echo "tcmalloc: 100 MB, 128 batch_size, 4 gpus, $train_loop" >> $log_dir/e2e_time.log;
LD_PRELOAD=$tcmalloc_path /usr/bin/time -f "Maximum resident set size (kbytes): %M" /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python /proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py /mydata/synthetic_data_100MBeach_8192files_symlink/ -b $batch_size -j $gpus --epochs $epochs --train-loop $train_loop --val-loop 0 --log-train-file $log_dir/b128_gpu4_100MBeach_tcmalloc.log >> $log_dir/e2e_time.log 2>&1;

# numactl
# nvlink connection is on node 0
echo "numactl cpunodebind 0 membind 0: 1 MB, 128 batch_size, 4 gpus, $train_loop" >> $log_dir/e2e_time.log;
numactl --cpunodebind=0 --membind=0 /usr/bin/time -f "Maximum resident set size (kbytes): %M" /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python /proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py /mydata/synthetic_data_1MBeach_8192files_symlink/ -b $batch_size -j $gpus --epochs $epochs --train-loop $train_loop --val-loop 0 --log-train-file $log_dir/b128_gpu4_1MBeach_numanode0.log >> $log_dir/e2e_time.log 2>&1;
echo "numactl cpunodebind 0 membind 0: 10 MB, 128 batch_size, 4 gpus, $train_loop" >> $log_dir/e2e_time.log;
numactl --cpunodebind=0 --membind=0 /usr/bin/time -f "Maximum resident set size (kbytes): %M" /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python /proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py /mydata/synthetic_data_10MBeach_8192files_symlink/ -b $batch_size -j $gpus --epochs $epochs --train-loop $train_loop --val-loop 0 --log-train-file $log_dir/b128_gpu4_10MBeach_numanode0.log >> $log_dir/e2e_time.log 2>&1;
echo "numactl cpunodebind 0 membind 0: 100 MB, 128 batch_size, 4 gpus, $train_loop" >> $log_dir/e2e_time.log;
numactl --cpunodebind=0 --membind=0 /usr/bin/time -f "Maximum resident set size (kbytes): %M" /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python /proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py /mydata/synthetic_data_100MBeach_8192files_symlink/ -b $batch_size -j $gpus --epochs $epochs --train-loop $train_loop --val-loop 0 --log-train-file $log_dir/b128_gpu4_100MBeach_numanode0.log >> $log_dir/e2e_time.log 2>&1;
# nvlink connection is on node 1
echo "numactl cpunodebind 1 membind 1: 1 MB, 128 batch_size, 4 gpus, $train_loop" >> $log_dir/e2e_time.log;
numactl --cpunodebind=1 --membind=1 /usr/bin/time -f "Maximum resident set size (kbytes): %M" /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python /proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py /mydata/synthetic_data_1MBeach_8192files_symlink/ -b $batch_size -j $gpus --epochs $epochs --train-loop $train_loop --val-loop 0 --log-train-file $log_dir/b128_gpu4_1MBeach_numanode1.log >> $log_dir/e2e_time.log 2>&1;
echo "numactl cpunodebind 1 membind 1: 10 MB, 128 batch_size, 4 gpus, $train_loop" >> $log_dir/e2e_time.log;
numactl --cpunodebind=1 --membind=1 /usr/bin/time -f "Maximum resident set size (kbytes): %M" /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python /proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py /mydata/synthetic_data_10MBeach_8192files_symlink/ -b $batch_size -j $gpus --epochs $epochs --train-loop $train_loop --val-loop 0 --log-train-file $log_dir/b128_gpu4_10MBeach_numanode1.log >> $log_dir/e2e_time.log 2>&1;
echo "numactl cpunodebind 1 membind 1: 100 MB, 128 batch_size, 4 gpus, $train_loop" >> $log_dir/e2e_time.log;
numactl --cpunodebind=1 --membind=1 /usr/bin/time -f "Maximum resident set size (kbytes): %M" /proj/prismgt-PG0/anaconda3/envs/torch2/bin/python /proj/prismgt-PG0/rbachkaniwala3/code/pytorch_main.py /mydata/synthetic_data_100MBeach_8192files_symlink/ -b $batch_size -j $gpus --epochs $epochs --train-loop $train_loop --val-loop 0 --log-train-file $log_dir/b128_gpu4_100MBeach_numanode1.log >> $log_dir/e2e_time.log 2>&1;