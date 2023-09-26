from PIL import Image
import time
import argparse
import subprocess

Image.MAX_IMAGE_PIXELS = 1000000000


# create argparse to accept log_file path
parser = argparse.ArgumentParser()

parser.add_argument('--log-file-suffix', type=str, help='prefix path to log file. Eg: "jemalloc" if the it is the mem allocator to be used', required=True)

args = parser.parse_args()


# All measurements in seconds
files = ['/mydata/synthetic_data_1MBeach_10GBtotal_symlink/random_image1MB.jpg',\
        '/mydata/synthetic_data_10MBeach_10GBtotal_symlink/random_image10MB.jpg',\
        '/mydata/synthetic_data_100MBeach_10GBtotal_symlink/random_image100MB.jpg']

logs = {}

for i in files:
    # open only reads meta data to determine type of image file.
    open_times = []
    # Inside convert, there's load and copy
        # Inside load, there's image mem allocate, read file and decode file. read and decode happen in chunks.
        # Inside copy, there's a call to C function in _imaging library which copies the image data from one buffer to another.
    mem_allocate_times = []
    read_file_times = []
    decode_file_times = []
    copy_times = []
    e2e_times = []
    for j in range(3):
        # evict cache of files
        status = subprocess.run(['vmtouch', '-e', i]).returncode
        if status != 0:
            print('Error evicting cache for file: ', i)
            exit(1)
        start_e2e = time.time()
        
        start = start_e2e
        im = Image.open(i, logging=True) 
        end = time.time()
        open_times.append(end - start)
        
        start = time.time()
        im.convert('RGB')
        end = time.time()
        
        end_e2e = end
        
        mem_allocate_times.append(im.log_time['load_prepare'])
        read_file_times.append(im.log_time['read'])
        decode_file_times.append(im.log_time['decode'])
        copy_times.append(im.log_time['copy'])
        e2e_times.append(end_e2e - start_e2e)
        im.close()
    
    # get rid of the first 2 measurements
    open_times = open_times[2:]
    mem_allocate_times = mem_allocate_times[2:]
    read_file_times = read_file_times[2:]
    decode_file_times = decode_file_times[2:]
    copy_times = copy_times[2:]
    e2e_times = e2e_times[2:]

    logs[i] = {'open': open_times, 'mem_allocate': mem_allocate_times, 'read_file': read_file_times, 'decode_file': decode_file_times, 'copy': copy_times, 'e2e': e2e_times}

# get average and std for each file
for i in logs:

    avg_open_percentage = sum(logs[i]['open'][j] / logs[i]['e2e'][j] for j in range(len(logs[i]['open'])))/len(logs[i]['open'])
    std_open_percentage = (sum([((logs[i]['open'][j] / logs[i]['e2e'][j]) - avg_open_percentage)**2 for j in range(len(logs[i]['open']))]) / len(logs[i]['open']))**0.5

    avg_mem_allocate_percentage = sum(logs[i]['mem_allocate'][j] / logs[i]['e2e'][j] for j in range(len(logs[i]['mem_allocate'])))/len(logs[i]['mem_allocate'])
    std_mem_allocate_percentage = (sum([((logs[i]['mem_allocate'][j] / logs[i]['e2e'][j]) - avg_mem_allocate_percentage)**2 for j in range(len(logs[i]['mem_allocate']))]) / len(logs[i]['mem_allocate']))**0.5

    avg_read_file_percentage = sum(logs[i]['read_file'][j] / logs[i]['e2e'][j] for j in range(len(logs[i]['read_file'])))/len(logs[i]['read_file'])
    std_read_file_percentage = (sum([((logs[i]['read_file'][j] / logs[i]['e2e'][j]) - avg_read_file_percentage)**2 for j in range(len(logs[i]['read_file']))]) / len(logs[i]['read_file']))**0.5

    avg_decode_file_percentage = sum(logs[i]['decode_file'][j] / logs[i]['e2e'][j] for j in range(len(logs[i]['decode_file'])))/len(logs[i]['decode_file'])
    std_decode_file_percentage = (sum([((logs[i]['decode_file'][j] / logs[i]['e2e'][j]) - avg_decode_file_percentage)**2 for j in range(len(logs[i]['decode_file']))]) / len(logs[i]['decode_file']))**0.5

    avg_copy_percentage = sum(logs[i]['copy'][j] / logs[i]['e2e'][j] for j in range(len(logs[i]['copy'])))/len(logs[i]['copy'])
    std_copy_percentage = (sum([((logs[i]['copy'][j] / logs[i]['e2e'][j]) - avg_copy_percentage)**2 for j in range(len(logs[i]['copy']))]) / len(logs[i]['copy']))**0.5



    avg_e2e = sum(logs[i]['e2e']) / len(logs[i]['e2e'])
    std_e2e = (sum([(x - avg_e2e)**2 for x in logs[i]['e2e']]) / len(logs[i]['e2e']))**0.5

    # print above percentages
    print('File: ', i)
    print('Percentage of e2e time for open: ', avg_open_percentage*100, ' +/- ', std_open_percentage*100)
    print('Percentage of e2e time for mem_allocate: ', avg_mem_allocate_percentage*100, ' +/- ', std_mem_allocate_percentage*100)
    print('Percentage of e2e time for read_file: ', avg_read_file_percentage*100, ' +/- ', std_read_file_percentage*100)
    print('Percentage of e2e time for decode_file: ', avg_decode_file_percentage*100, ' +/- ', std_decode_file_percentage*100)
    print('Percentage of e2e time for copy: ', avg_copy_percentage*100, ' +/- ', std_copy_percentage*100)
    print('Average e2e time: ', avg_e2e, ' +/- ', std_e2e)

    # dump above percentages to a file
    with open('percentage_' + i.split('/')[-1].split('.')[0] + '_' + args.log_file_suffix  + '.log', 'w') as f:
        f.write('File: ' + i + '\n')
        f.write('Percentage of e2e time for open: ' + str(avg_open_percentage*100) + ' +/- ' + str(std_open_percentage*100) + '\n')
        f.write('Percentage of e2e time for mem_allocate: ' + str(avg_mem_allocate_percentage*100) + ' +/- ' + str(std_mem_allocate_percentage*100) + '\n')
        f.write('Percentage of e2e time for read_file: ' + str(avg_read_file_percentage*100) + ' +/- ' + str(std_read_file_percentage*100) + '\n')
        f.write('Percentage of e2e time for decode_file: ' + str(avg_decode_file_percentage*100) + ' +/- ' + str(std_decode_file_percentage*100) + '\n')
        f.write('Percentage of e2e time for copy: ' + str(avg_copy_percentage*100) + ' +/- ' + str(std_copy_percentage*100) + '\n')
        f.write('Average e2e time: ' + str(avg_e2e) + ' +/- ' + str(std_e2e) + '\n')