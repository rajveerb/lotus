# %%
# open a file as csv without header

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,natsort
import argparse

plt.rc('axes', labelsize=45)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=35)    # fontsize of the tick labels
plt.rc('ytick', labelsize=35)    # fontsize of the tick labels
plt.rc('legend', fontsize=30)    # legend fontsize

# take below arguments using argparse
# add argument to pass pytorch_profiler_data_file
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    default='/users/rajveerb/pytorch_custom_log_one_epoch_imagenet_dataset/', help='Root directory with custom_log for different configs')

parser.add_argument('--sort_criteria', type=str,
                    default='batch_id', help='sort by `batch_id` or `duration`')

parser.add_argument('--fig_dir', type=str,
                    default='./figs',
                    help='Path to store the figures')
                    

args = parser.parse_args()

# %%
# sort_by = 'batch_id' or 'duration'
def plotter_preprocessing_time(target_dir,sort_by='batch_id',fig_size=(50,12),remove_outliers=True,fig_prefix='',fig_dir=''):
    
    root_to_files = {}
    for root, dirs, files in os.walk(target_dir):
        root_to_files[root] = files
    roots = sorted(root_to_files, key=lambda x: natsort.natsort_key(x.lower()))

    prev_batch = None
    for root in roots:
        print(root)
        files = root_to_files[root]
        plot_df = pd.DataFrame()
        
        for file in files:
            if "worker_pid" not in file:
                continue

            df = pd.read_csv(os.path.join(root, file)
                            , header=None)

            # add header
            df.columns = ['name','start_ts','duration']

            # names that start with 'SBatchPreprocessed'
            df = df[df['name'].str.startswith('SBatchPreprocessed')]
            # map 'SBatchPreprocessed_' such that 'SBatchPreprocessed_idx' becomes 'idx' where idx is an integer
            df['batch_id'] = df['name'].map(lambda x: int(x.replace('SBatchPreprocessed_','')))


            # divide by 1000000 to convert from nanoseconds to milliseconds
            df['duration'] = df['duration']/1000000

            # concatentate all dataframes
            plot_df = pd.concat([plot_df, df])
        
        if plot_df.empty:
            continue
        def remove_wild_outliers(plot_df):
            # mean and std before removing outliers
            mean = np.mean(plot_df["duration"])
            std = np.std(plot_df["duration"])
            print (f'Before removing outliers:')
            print(f'avg = {mean:.2f} ms, std = {std:.2f} ({100*std/mean:.2f}% of avg) ms , min = {plot_df["duration"].min():.2f} ms, max = {plot_df["duration"].max():.2f} ms')
            # remove rows with duration's > mean - 2*std and < mean + 2*std
            plot_df = plot_df[plot_df['duration'] < plot_df['duration'].mean() + 2*plot_df['duration'].std()]
            plot_df = plot_df[plot_df['duration'] > plot_df['duration'].mean() - 2*plot_df['duration'].std()] 
            # mean and std after removing outliers
            print (f'After removing outliers:')
            return plot_df

        batch = root.split('/')[-1].split('b')[1].split('_')[0] # <long_dir_path> -> 128_gpu4 -> 128

        label = root.split('/')[-1] # retrieves b128_gpu4 kind of label
        print(f'{label}:')
        # remove outliers
        if remove_outliers:
            plot_df = remove_wild_outliers(plot_df)

        if prev_batch is None:
            prev_batch = batch 
        elif batch != prev_batch:

            # plt.figure(figsize=fig_size)
            plt.gcf().set_size_inches(fig_size[0], fig_size[1])
            # label x axis
            plt.xlabel(f'Batch ids preprocessed (sorted by {sort_by}) (batch size = {prev_batch})')
            # label y axis
            plt.ylabel('Preprocessing time in ms')
            plt.legend()
            fig_path = os.path.join(fig_dir, f'{fig_prefix}{prev_batch}_batch_preprocessing_time.png') 
            plt.savefig(fig_path)
            plt.clf()
            prev_batch = batch            
        
        mean = np.mean(plot_df["duration"])
        std = np.std(plot_df["duration"])
        print (f'avg = {mean:.2f} ms, std = {std:.2f} ({100*std/mean:.2f}% of avg) ms, min = {plot_df["duration"].min():.2f} ms, max = {plot_df["duration"].max():.2f} ms')
        plot_df = plot_df.sort_values(by=[sort_by])
        plt.plot(plot_df['batch_id'], plot_df['duration'], label=label, marker='s')
        # plot on log scale
        plt.yscale('log')

    # plt.figure(figsize=fig_size)
    plt.gcf().set_size_inches(fig_size[0], fig_size[1])
    # label x axis
    plt.xlabel(f'Batch ids preprocessed (sorted by {sort_by})')
    # label y axis
    plt.ylabel('Preprocessing time in ms')
    plt.legend()    
    fig_path = os.path.join(fig_dir, f'{fig_prefix}{prev_batch}_batch_preprocessing_time.png') 
    plt.savefig(fig_path)
    plt.clf()

# %%
# sort_by = 'name' or 'duration'
def plotter_preprocessing_wait_time(target_dir,sort_by='batch_id',fig_size=(50,12),fig_prefix='',fig_dir=''):
    plt.figure(figsize=fig_size)
    root_to_files = {}
    for root, dirs, files in os.walk(target_dir):
        root_to_files[root] = files
    roots = sorted(root_to_files, key=lambda x: natsort.natsort_key(x.lower()))
    
    prev_batch = None
    
    for root in roots:
        print (root)
        files = root_to_files[root]
        plot_df = pd.DataFrame()
        for file in files:
            if "main_pid" not in file:
                continue

            df = pd.read_csv(os.path.join(root, file)
                            , header=None)

            # add header
            df.columns = ['name','start_ts','duration']

            # names that start with 'SBatchWait'
            df = df[df['name'].str.startswith('SBatchWait')]

            # map 'SBatchWait' such that 'SBatchWait_idx' becomes 'idx' where idx is an integer
            df['batch_id'] = df['name'].map(lambda x: int(x.replace('SBatchWait_','')))

            # divide by 1000000 to convert from nanoseconds to milliseconds
            df['duration'] = df['duration']/1000000

            # concatentate all dataframes
            plot_df = pd.concat([plot_df, df])
        
        if plot_df.empty:
            continue

        label = root.split('/')[-1] # retrieves b128_gpu4 kind of label
        batch = root.split('/')[-1].split('b')[1].split('_')[0] # <long_dir_path> -> 128_gpu4 -> 128

        print(f'{label}:')

        if prev_batch is None:
            prev_batch = batch 
        elif batch != prev_batch:

            # plt.figure(figsize=fig_size)
            plt.gcf().set_size_inches(fig_size[0], fig_size[1])
            # label x axis
            plt.xlabel(f'Batch ids preprocessed (sorted by {sort_by}) (batch size = {prev_batch})')
            # label y axis
            plt.ylabel('Wait time in ms')
            plt.legend()
            fig_path = os.path.join(fig_dir,f'{fig_prefix}{prev_batch}_batch_preprocessing_wait_time.png')
            plt.savefig(fig_path)
            plt.clf()
            prev_batch = batch 

        mean = np.mean(plot_df["duration"])
        std = np.std(plot_df["duration"])
        minimum = plot_df["duration"].min()
        maximum = plot_df["duration"].max() 
        print (f'avg = {mean:.2f} ms, std = {std:.2f} ({100*std/mean:.2f}% of avg) ms, min = {minimum:.2f} ms, max = {maximum:.2f} ms')
        # print % of batches which had high wait time > 100 ms pretty print
        print (f'{100*len(plot_df[plot_df["duration"] > 100])/len(plot_df):.2f}% of batches had wait time > 100 ms')
        
        # sort by sort_by
        plot_df = plot_df.sort_values(by=[sort_by])
        plt.scatter(plot_df['batch_id'], plot_df['duration'], label=label)
        # plot on log scale
        plt.yscale('log')

    plt.gcf().set_size_inches(fig_size[0], fig_size[1])
    # label x axis
    plt.xlabel(f'Wait time for batch ids preprocessed (sorted by {sort_by})')
    # label y axis
    plt.ylabel('Wait time in ms')
    plt.legend()
    fig_path = os.path.join(fig_dir,f'{fig_prefix}{prev_batch}_batch_preprocessing_wait_time.png')
    plt.savefig(fig_path)
    plt.clf()

# # %%
print("preprocessing time")
plotter_preprocessing_time(args.data_dir,sort_by=args.sort_criteria,remove_outliers=True,fig_dir=args.fig_dir)

# %%
print("wait time")
plotter_preprocessing_wait_time(args.data_dir,sort_by=args.sort_criteria,fig_dir=args.fig_dir)