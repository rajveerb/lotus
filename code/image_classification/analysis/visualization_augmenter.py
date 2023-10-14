
# read file lol_pid_1670885 as csv
import pandas as pd
import json
import os
import argparse

# add argument to pass pytorch_profiler_data_file
parser = argparse.ArgumentParser()
parser.add_argument('--pytorch_profiler_data_file', type=str,
                    default='pytorch_profile_log_default_fetch/kepler2_2126935.1696353112530.pt.trace.json', help='pytorch_profiler_data_file')
# custom log
parser.add_argument('--custom_log', type=str, default='',
                    help='custom_log, requirement is the file should end with pid_<pidnumber>, for example, abcdef_pid_1234')


def update_pytorch_profile_data(pytorch_profiler_data_file, pid):

    # add header to csv
    df = pd.read_csv(f, header=None)

    # add column names
    df.columns = ['name', 'ts', 'dur']

    res = ''
    res_list = []
    for row in df.itertuples():
        # print(row)
        # print row with column name 'name'
        # print(row.name)
        # if line starts with "S"
        if row.name.startswith("S"):
            data = {"ph": "X", "cat": "user_annotation", "name": row.name, "pid": int(
                pid), "tid": int(pid), "args": {}, "ts": row.ts//1000, "dur": row.dur//1000}
            res_list.append(data)
            res += str(data).replace("'", '"') + ','

    # open a file and read as json
    json_file = open(pytorch_profiler_data_file, 'r')
    pytorch_profiler_data = json.load(json_file)
    # get trace events
    traceEvents = pytorch_profiler_data['traceEvents']
    traceEvents = traceEvents + res_list
    # replace trace events
    pytorch_profiler_data['traceEvents'] = traceEvents
    json_file.close()
    # write to file
    with open(pytorch_profiler_data_file, 'w') as outfile:
        json.dump(pytorch_profiler_data, outfile)

args = parser.parse_args()

for f in os.listdir("."):
    if f.startswith(args.custom_log):
        pid = f.split("_")[-1]
        update_pytorch_profile_data(args.pytorch_profiler_data_file, pid)
