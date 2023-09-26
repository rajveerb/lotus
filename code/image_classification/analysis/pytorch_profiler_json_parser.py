
import json,argparse

# get profile file from command line
parser = argparse.ArgumentParser()
parser.add_argument('--profiler_file', type=str, required=True, help='path to profiler file')

args = parser.parse_args()

def print_parsed_json_data(json_file_path: str) -> None:
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

    avg_data_wait_time = sum(data_wait_time)/len(data_wait_time)/1000
    avg_forward_and_backward_pass_time = sum(forward_and_backward_pass_time)/len(forward_and_backward_pass_time)/1000
    avg_move_data_to_device_time = sum(move_data_to_device_time)/len(move_data_to_device_time)/1000

    # print in milliseconds
    print("Average data wait time in milliseconds:","{:.4f}".format(avg_data_wait_time))
    print("Average forward + backward pass time in milliseconds:","{:.4f}".format(avg_forward_and_backward_pass_time))
    print("Average move data to device time in milliseconds:","{:.4f}".format(avg_move_data_to_device_time))

# profiler data output
profiler_data_file = args.profiler_file

# get parsed json data
print_parsed_json_data(profiler_data_file)