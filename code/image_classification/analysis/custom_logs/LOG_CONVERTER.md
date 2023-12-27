## To convert old log to an approximate new log

Refer to the GitHub issue [here](https://github.com/rajveerb/ml-pipeline-benchmark/issues/28) for more details/motivation.


1. Use below command to find out options:

```python
    python convert_old_log_to_new_log.py -h
```

2. You only need the data directory where custom logs are stored and the script will recursively find all the logs and generate in-place i.e replace old logs with new.
