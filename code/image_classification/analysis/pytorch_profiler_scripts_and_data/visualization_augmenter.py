
# read file lol_pid_1670885 as csv
import pandas as pd
import json
import os
import argparse

# add argument to pass pytorch_profiler_data_file
parser = argparse.ArgumentParser()
parser.add_argument('--pytorch_profiler_data_dir', type=str,
                    default='pytorch_profiles_imagenet_dataset', help='Root directory which stores pytorch profiler data along with custom_log for different configs')
# custom log
parser.add_argument('--custom_log_prefix', type=str, default='custom_log',
                    help='custom_log, requirement is the file should begin with the prefix passed as an argument\
                          and end with pid_<pidnumber> without any extension such as .json, for example, custom_log_abcdef_pid_1234')

parser.add_argument('--compact', action='store_true', help='compact option will not include fine grained breakdown of batch preprocessing ops such as \
                    loader time,\
                    transformation time,\
                    collation time,\
                    individual transformation time')
# add argument to pass custom_log json vizualization file
parser.add_argument('--custom_log_json_viz_file', type=str,
                    default='custom_log.json', help='[Warning] A json file will be generated which can be visualized using chrome://tracing \
                    \nNote: The file should not contain custom_log prefix in the name or end with json')


def update_pytorch_profile_data(custom_log_file, pid, compact=True):

    # add header to csv
    df = pd.read_csv(custom_log_file, header=None)

    # add column names
    df.columns = ['name', 'ts', 'dur']

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
            synthetic_id = -1 + (int(batch_id) * -1)

            data = {"ph": "X", "cat": "user_annotation", "name": row.name, "pid": int(
                pid), "tid": int(pid), "args": {"External id": synthetic_id, "correlation": synthetic_id}, "ts": row.ts//1000, "dur": row.dur//1000}
            res_list.append(data)
            data = {"ph": "f","id": synthetic_id,"pid": int(pid),"tid": int(pid),"ts": row.ts//1000,"cat": "ac2g","name": "ac2g", "bp": "e"}
            res_list.append(data)
        
        elif row.name.startswith("SBatchWait"):
            batch_id = row.name.split("_")[1]
            synthetic_id = -1 + (int(batch_id) * -1)

            data = {"ph": "X", "cat": "user_annotation", "name": row.name, "pid": int(
                pid), "tid": int(pid), "args": {"External id": synthetic_id, "correlation": synthetic_id}, "ts": row.ts//1000, "dur": row.dur//1000}
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

def save_custom_log_profiler_format(custom_log_only_profiler_format_file,result):
    # write to file
    with open(custom_log_only_profiler_format_file, 'w') as outfile:
        json.dump({'traceEvents': result}, outfile)

args = parser.parse_args()

# check for custom log file viz warning
if args.custom_log_prefix in args.custom_log_json_viz_file or args.custom_log_json_viz_file.endswith(".json"):
    print("[Warning] The custom_log_json_viz_file should not contain custom_log prefix in the name or end with json")
    exit()

# recursively search for pytorch profiler data files
for root, dirs, files in os.walk(args.pytorch_profiler_data_dir):
    result = []
    result_pytorch_profiler_data_file = None
    for file in files:
        if file.startswith(args.custom_log_prefix):
            pid = file.split("_")[-1]
            result += update_pytorch_profile_data(os.path.join(root, file), pid, args.compact)
        elif file.endswith(".json"):
            result_pytorch_profiler_data_file = os.path.join(root, file) 
    if result_pytorch_profiler_data_file:
        save_augmented_profiler_data(result_pytorch_profiler_data_file,args.compact,result)
    else:
        custom_log_only_profiler_format_file = os.path.join(root, args.custom_log_json_viz_file)
        save_custom_log_profiler_format(custom_log_only_profiler_format_file,result)