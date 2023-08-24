# !/bin/bash

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 python rbachkaniwala3/code/image_ops_time_splitter.py --log-file-suffix jemalloc
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 python rbachkaniwala3/code/image_ops_time_splitter.py --log-file-suffix tcmalloc
python rbachkaniwala3/code/image_ops_time_splitter.py --log-file-suffix malloc