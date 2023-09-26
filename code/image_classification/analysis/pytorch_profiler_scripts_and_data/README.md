## Generate the PyTorch profiler data


```python
python <image_pipeline_code_file> <imagenet_dir> -b 128 --gpu 1 --epochs 1 --val-loop 0 --profile --profile-log-prefix <profile_log_prefix>
```

For below without fetcher case, find the file `torch/utils/data/dataloader.py`, for example `anaconda3/envs/torch2/lib/python3.10/site-packages/torch/utils/data/dataloader.py`, in your anaconda environment and change the code

```python
# prime the prefetch loop
for _ in range(self._prefetch_factor * self._num_workers):
    self._try_put_index()
```

to

```python
# prime the prefetch loop
for _ in range(1):
    self._try_put_index()
```

and then run the profiling command.

## Parse the PyTorch profiler data example:

```python
python gpu_time_parser.py --profiler_file ml-pipeline-benchmark/code/image_classification/analysis/sample_pytorch_profiler_json_data/withfetcher.json
```

```python
python gpu_time_parser.py --profiler_file ml-pipeline-benchmark/code/image_classification/analysis/sample_pytorch_profiler_json_data/withoutfetcher.json
```

## Visualize the json file using Google chrome's tracing feature

Navigate to `chrome://tracing/` in the Google chrome browser and load any of the json files in this directory.