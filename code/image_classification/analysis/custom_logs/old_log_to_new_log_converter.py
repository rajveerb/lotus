# %%
import os,natsort,argparse

parser = argparse.ArgumentParser(description='Convert old log to new log in-place')
parser.add_argument('--target_dir', type=str,\
                     default='/users/rajveerb/pytorch_custom_log_one_epoch_imagenet_dataset/',\
                          help='target directory')

args = parser.parse_args()

def convert_old_log_to_new_log(target_dir):
    # %%
    root_to_files = {}
    for root, dirs, files in os.walk(target_dir):
        root_to_files[root] = files
    roots = sorted(root_to_files, key=lambda x: natsort.natsort_key(x.lower()))   

    # %%
    for root in roots:
        print(root)
        files = root_to_files[root]
        for file in files:
            if "main_pid" not in file:
                continue
            
            data = ""
            # starts with 0
            batch_id = 0
            # read file line by line
            file_path = os.path.join(root, file)
            f = open(file_path, "r")
            seen = set()
            start_ts = ''
            for line in f.readlines():
                # only modify lines with "SBatchWait"
                if "SBatchWait" in line:
                    # each line has format "SBatchWait_idx,start_ts,duration\n"
                    read_batch_index = int(line.split(",")[0].split("_")[-1]) # retrieves idx
                    if (read_batch_index > batch_id):
                        # 1000 is a dummy duration of 1 us
                        line = line.split(",")[0] + "," + line.split(",")[1] + "," + "1000\n"
                        data += line
                        seen.add(read_batch_index)
                        if start_ts == '':
                            start_ts = line.split(",")[1]
                    
                    elif (read_batch_index == batch_id):
                        if start_ts == '':
                            data += line
                            batch_id += 1
                            while(batch_id in seen):
                                batch_id += 1
                        else:
                            duration = int(line.split(",")[-1])
                            duration += int(line.split(",")[1]) - int(start_ts)
                            line = line.split(",")[0] + "," + start_ts + "," + str(duration) + "\n"
                            data += line
                            start_ts = ''
                            batch_id += 1
                            while(batch_id in seen):
                                batch_id += 1                        
                        
                    else:
                        raise Exception("Read batch index is less than batch id")
                else:
                    # keep other lines as they are
                    data += line
                    pass
                
            f.close()
            open(file_path, 'w').write(data)

convert_old_log_to_new_log(args.target_dir)