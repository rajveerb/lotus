
import json,argparse
import numpy as np
import pandas as pd
# plot the data
import matplotlib.pyplot as plt
import natsort

# get profile file from command line
parser = argparse.ArgumentParser()
parser.add_argument('--profiler_data_dir', type=str, help='path to profiler file', default='pytorch_profiler_json_data')
parser.add_argument('--output_csv_file', type=str, help='path to output csv file', default='output_csv_file.csv')
parser.add_argument('--output_plot_file', type=str, help='path to output png file', default='output_plot_file.png')

args = parser.parse_args()

def get_parsed_json_data(json_file_path: str) -> dict:
    profile_step_to_event_map = {}
    # read json file
    with open(json_file_path) as f:
        data = json.load(f)
        events = data['traceEvents']

        # if user annotation is present, then store it in a dictionary from timestamp to name
        for index,event in enumerate(events):
            if 'cat' in event and event['name'].startswith('Profiler') and not event['name'].startswith('Profiler#0'):
                start = event['ts']
                end = event['ts'] + event['dur']
                profile_step_to_event_map[event['name']] = {}
                for i in range(index+1,len(events)):
                    # if start of another event is greater than end of current event, then break
                    if events[i]['ts'] > end:
                        break

                    # if user annotation is present, then store it in a dictionary from timestamp to name
                    if 'cat' in events[i] and 'user_annotation' == events[i]['cat'] and not events[i]['name'].startswith('Optimizer'):
                        event_name = events[i]['name'] if 'DataLoader' not in events[i]['name'] else "DataLoaderWait"
                        profile_step_to_event_map[event['name']][event_name] = events[i]
                # if profile_step_to_event_map[event['name']] is empty dictionary, then remove it
                if not profile_step_to_event_map[event['name']]:
                    del profile_step_to_event_map[event['name']]

    data_wait_time = []
    forward_and_backward_pass_time = []
    move_data_to_device_time = []
    for step in profile_step_to_event_map:
        data_wait_time.append(profile_step_to_event_map[step]['DataLoaderWait']['dur'])
        forward_pass_start = profile_step_to_event_map[step]['model_forward_pass']['ts']
        backward_pass_end = profile_step_to_event_map[step]['model_backward_pass']['ts'] + profile_step_to_event_map[step]['model_backward_pass']['dur']
        forward_and_backward_pass_time.append(backward_pass_end - forward_pass_start)
        move_data_to_device_time.append(profile_step_to_event_map[step]['move_data_to_device']['dur'])
    
    # convert to numpy array and divide by 1000 to convert to milliseconds
    data_wait_time = np.array(data_wait_time)/1000
    forward_and_backward_pass_time = np.array(forward_and_backward_pass_time)/1000
    move_data_to_device_time = np.array(move_data_to_device_time)/1000
    
    # print avg
    print("Avg. Data wait time:\n","{:.4f}".format(np.average(data_wait_time)))
    print("Avg. Forward + backward pass time:\n","{:.4f}".format(np.average(forward_and_backward_pass_time)))
    print("Avg. Move data to device time:\n","{:.4f}".format(np.average(move_data_to_device_time)))

    parsed_data = {"data_wait_time":np.average(data_wait_time),"forward_and_backward_pass_time":np.average(forward_and_backward_pass_time),"move_data_to_device_time":np.average(move_data_to_device_time)}

    return parsed_data

# given a dir recursively, get all json files
def get_all_json_files(dir_path: str) -> list:
    import os
    json_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files

json_files = get_all_json_files(args.profiler_data_dir)
json_files.sort()
# create a datafraem with columns as json files, data_wait_time, forward_and_backward_pass_time, and move_data_to_device_time
df = pd.DataFrame(columns=['json_file','data_wait_time','forward_and_backward_pass_time','move_data_to_device_time'])

for json_file in json_files:
    print("\n\n***Below is elapsed time in milliseconds for file: \n{}".format(json_file))
    parsed_data = get_parsed_json_data(json_file)
    json_file = "/".join(json_file.split('/')[:-1]).replace(args.profiler_data_dir,'').lstrip('/')
    df = pd.concat([df,pd.DataFrame([[json_file,parsed_data['data_wait_time'],parsed_data['forward_and_backward_pass_time'],parsed_data['move_data_to_device_time']]],columns=['json_file','data_wait_time','forward_and_backward_pass_time','move_data_to_device_time'])],ignore_index=True)

# sort by json_file name such that a number in the string is used for sorting
# df['json_file'] = natsort.natsorted(df['json_file'], alg=natsort.ns.IGNORECASE)
# need to sort the rows based on json_file column but using natsort ignores the case and sorts the rows
df = df.sort_values(by='json_file', key=lambda x: natsort.natsort_key(x.str.lower()), ignore_index=True)

# sort by json_file name such that a number in the string is used for sorting
df = df.sort_values(by='json_file', key=lambda x: natsort.natsort_key(x.str.lower()), ignore_index=True)
df.to_csv(args.output_csv_file,index=False)

# plot a stacked bar chart
plt.figure(figsize=(20,10))
# log scale
plt.yscale('log')
# make label vertical
plt.xticks(rotation=90)
plt.bar(df['json_file'],df['data_wait_time'],label='data_wait_time')
plt.bar(df['json_file'],df['forward_and_backward_pass_time'],bottom=df['data_wait_time'],label='forward_and_backward_pass_time')
plt.bar(df['json_file'],df['move_data_to_device_time'],bottom=df['data_wait_time']+df['forward_and_backward_pass_time'],label='move_data_to_device_time')
plt.legend()
plt.savefig(args.output_plot_file)