
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

parser.add_argument('--compact', action='store_true', help='compact option will not include fine grained breakdown of batch preprocessing ops such as \
                    loader time,\
                    transformation time,\
                    collation time,\
                    individual transformation time')


def update_pytorch_profile_data(custom_log_file, pid, compact=True):

    # add header to csv
    df = pd.read_csv(custom_log_file, header=None)

    # add column names
    df.columns = ['name', 'ts', 'dur']

    res = ''
    res_list = []
    for row in df.itertuples():
        # if line starts with "S", for our added instrumentation
        if row.name.startswith("SBatchPreprocessed"):
            batch_id = row.name.split("_")[1]
            synthetic_id = -1 + (int(batch_id) * -1)

            data = {"ph": "X", "cat": "user_annotation", "name": row.name, "pid": int(
                pid), "tid": int(pid), "args": {"External id": synthetic_id, "correlation": synthetic_id}, "ts": row.ts//1000, "dur": row.dur//1000}
            res_list.append(data)
            data = {"ph": "s","id": synthetic_id,"pid": int(pid),"tid": int(pid),"ts": row.ts//1000,"cat": "ac2g","name": "ac2g"}
            res_list.append(data)
        
        elif row.name.startswith("SBatchConsumed"):
            batch_id = row.name.split("_")[1]
            synthetic_id = (int(batch_id) * -1)

            data = {"ph": "X", "cat": "user_annotation", "name": row.name, "pid": int(
                pid), "tid": int(pid), "args": {"External id": synthetic_id, "correlation": synthetic_id}, "ts": row.ts//1000, "dur": row.dur//1000}
            res_list.append(data)
            data = {"ph": "f","id": synthetic_id,"pid": int(pid),"tid": int(pid),"ts": row.ts//1000,"cat": "ac2g","name": "ac2g"}
            res_list.append(data)
            
        if not compact:
            if not row.name.startswith("SBatch"):
                data = {"ph": "X", "cat": "user_annotation", "name": row.name, "pid": int(
                    pid), "tid": int(pid), "args": {}, "ts": row.ts//1000, "dur": row.dur//1000}
                res_list.append(data)
        
    return res_list

def save_augmented_profiler_data(pytorch_profiler_data_file,compact,result):
    # open a file and read as json
    if compact:
        file_name_split = pytorch_profiler_data_file.split(".json")[0]
        log_file_name = file_name_split+f"_compact.json"
    else:
        file_name_split = pytorch_profiler_data_file.split(".json")[0]
        log_file_name = file_name_split+f"_elaborate.json"

    json_file = open(pytorch_profiler_data_file, 'r')
    pytorch_profiler_data = json.load(json_file)
    # get trace events
    traceEvents = pytorch_profiler_data['traceEvents']
    traceEvents = traceEvents + result
    # replace trace events
    pytorch_profiler_data['traceEvents'] = traceEvents
    json_file.close()
    # write to file
    with open(log_file_name, 'w') as outfile:
        json.dump(pytorch_profiler_data, outfile)


args = parser.parse_args()

result = []
for f in os.listdir("."):
    if f.startswith(args.custom_log):
        pid = f.split("_")[-1]
        result += update_pytorch_profile_data(f, pid, args.compact)

save_augmented_profiler_data(args.pytorch_profiler_data_file,args.compact,result)