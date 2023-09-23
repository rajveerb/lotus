
import json,argparse

# get profile file from command line
parser = argparse.ArgumentParser()
parser.add_argument('--profiler_file', type=str, required=True, help='path to profiler file')
args = parser.parse_args()

def get_gpu_time(profiler_file: str)->int:
    time_to_name ={}
    timestamps = []
    annotation_types = []
    # read json file
    with open(profiler_file) as f:
        data = json.load(f)
        events = data['traceEvents']

        # if user annotation is present, then store it in a dictionary from timestamp to name
        for event in events:
            if 'cat' in event and event['cat'] == 'user_annotation' and not event['name'].startswith('Profiler'):
                annotation_types.append(event['name'])
                if event['name'] == 'model_backward_pass':
                    timestamps.append(event['ts']+event['dur'])
                    time_to_name[event['ts']+event['dur']] = event['name']
                else:
                    timestamps.append(event['ts'])
                    time_to_name[event['ts']] = event['name']

    # to print all available annotation types
    # print(set(annotation_types))

    timestamps.sort()

    time_to_train_on_gpu = []
    for timestamp in timestamps:
        if time_to_name[timestamp] == 'model_forward_pass':
            start = timestamp
        elif time_to_name[timestamp] == 'model_backward_pass':
            end = timestamp
            time_to_train_on_gpu.append(end - start)

    return sum(time_to_train_on_gpu)/len(time_to_train_on_gpu)

# profiler data output
profiler_data_file = args.profiler_file

gpu_time = get_gpu_time(profiler_data_file)
print("Elapsed time of CPU finishing the issue of forward + backward pass related kernel ops:")
print(profiler_data_file)
print(gpu_time/1000, "ms")