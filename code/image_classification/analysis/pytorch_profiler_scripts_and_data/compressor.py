# compress files recursively given a root directory using `xz` equivalent compression
# below compression is lossless and can be decompressed using `unxz` command
# LSMA is just a library that implements LZMA compression algorithm
import os,lzma,argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, required=True,
                    help='path to pytorch profiler data directory which has json files to be compressed/decompressed')

parser.add_argument('--decompress', action='store_true',\
                     help='decompress option will decompress all the files in the directory that end with .xz extension')

def compress_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.json'):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'rb') as f_in:
                    with lzma.open(filepath + '.xz', 'wb') as f_out:
                        f_out.write(f_in.read())
                os.remove(filepath)

def uncompress_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if filepath.endswith('.xz'):
                with lzma.open(filepath, 'rb') as f_in:
                    with open(filepath.split('.xz')[0], 'wb') as f_out:
                        f_out.write(f_in.read())
                os.remove(filepath)

args = parser.parse_args()

if args.decompress:
    uncompressed_files(args.data_dir)
else:
    compress_files(args.data_dir)