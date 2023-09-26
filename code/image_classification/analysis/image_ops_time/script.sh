# !/bin/bash

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 python image_ops_time_splitter.py --log-file-suffix jemalloc
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 python image_ops_time_splitter.py --log-file-suffix tcmalloc
python image_ops_time_splitter.py --log-file-suffix malloc