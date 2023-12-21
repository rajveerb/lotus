## Compress PyTorch profiler files in this directory

To compress:

```bash
make compress
```

To decompress:

```bash
make decompress
```

## Generate the PyTorch profiler data


```python
python <image_pipeline_code_file> <imagenet_dir> -b 128 --gpu 1 --epochs 1 --val-loop 0 --profile --profile-log-prefix <profile_log_prefix>
```



## Parse the PyTorch profiler data example:

```python
python gpu_time_parser.py --profiler_file ml-pipeline-benchmark/code/image_classification/analysis/sample_pytorch_profiler_json_data/withfetcher.json
```

```python
python gpu_time_parser.py --profiler_file ml-pipeline-benchmark/code/image_classification/analysis/sample_pytorch_profiler_json_data/withoutfetcher.json
```

## Visualize the json file using Google chrome's tracing feature

Navigate to `chrome://tracing/` in the Google chrome browser and load any of the json files in this directory.