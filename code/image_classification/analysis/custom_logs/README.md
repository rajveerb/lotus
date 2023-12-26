## How to get graphs and summary stats from the custom logs of our instrumentation

1. Use below command to find out options:

```python
    python measure_elapsed_time_plot.py -h
```


2. You only need the data directory where custom logs are stored and the script will recursively find all the logs and generate graphs and summary stats.