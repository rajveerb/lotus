import pandas as pd
import numpy as np
import os, natsort
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    default="lotustrace_result/b512_gpu4",
    help="Root directory with P3Torch_log for different configs",
)

parser.add_argument(
    "--remove_outliers",
    action="store_true",
    help="Remove outliers less than q1 - iqr, where iqr = (q3 - q1) * 2.\
                     This was done to account for the last batch which might batch_size < the selected batch_size ",
)

parser.add_argument(
    "--output_file",
    default="lotustrace_result/iqr_and_stddev_preprocessing_time_stats.log",
    help="Output file to save the stats",
)

args = parser.parse_args()

logging.basicConfig(
    filename=args.output_file, level=logging.INFO, filemode="w", format=""
)

def preprocessing_time_summary(
    target_dir,
    remove_outliers,
):

    root_to_files = {}
    for root, dirs, files in os.walk(target_dir):
        root_to_files[root] = files
    roots = sorted(root_to_files, key=lambda x: natsort.natsort_key(x.lower()))

    batch_to_summary = {"std": {}, "IQR": {}, 'median': {}, '25th': {}, '75th': {}}

    for root in roots:
        if "e2e" in root:
            continue
        print(root)
        files = root_to_files[root]
        plot_df = pd.DataFrame()

        for file in files:
            if "worker_pid" not in file:
                continue

            df = pd.read_csv(os.path.join(root, file), header=None)

            # add header
            df.columns = ["name", "start_ts", "duration"]

            # names that start with 'SBatchPreprocessed'
            df = df[df["name"].str.startswith("SBatchPreprocessed")]
            # map 'SBatchPreprocessed_' such that 'SBatchPreprocessed_idx' becomes 'idx' where idx is an integer
            df["batch_id"] = df["name"].map(
                lambda x: int(x.replace("SBatchPreprocessed_", ""))
            )

            # divide by 1000000 to convert from nanoseconds to milliseconds
            df["duration"] = df["duration"] / 1000000

            # concatentate all dataframes
            plot_df = pd.concat([plot_df, df])

        if plot_df.empty:
            continue

        def remove_wild_outliers(plot_df):
            q1 = plot_df["duration"].quantile(0.25)
            q3 = plot_df["duration"].quantile(0.75)
            iqr = (q3 - q1) * 2

            # remove outliers less than q1 - iqr only (these are numbers from last batch which has
            #  elements less than batch size because elements in a dataset may not be a multiple of batch size)
            print("remove outliers less than q1 - iqr, where iqr = (q3 - q1) * 2")
            plot_df = plot_df[~(plot_df["duration"] < (q1 - iqr))]
            return plot_df

        label = root.split("/")[-1]  # retrieves b128_gpu4 kind of label
        print(f"{label}:")
        # remove outliers
        if remove_outliers:
            plot_df = remove_wild_outliers(plot_df)

        std = np.std(plot_df["duration"])
        quartile_diff = np.percentile(plot_df["duration"], 75) - np.percentile(
            plot_df["duration"], 25
        )

        batch_to_summary["std"][label] = std
        batch_to_summary["IQR"][label] = quartile_diff
        batch_to_summary['median'][label] = np.median(plot_df["duration"])
        batch_to_summary['25th'][label] = np.percentile(plot_df["duration"], 25)
        batch_to_summary['75th'][label] = np.percentile(plot_df["duration"], 75)

    return batch_to_summary


batch_to_summary = preprocessing_time_summary(
    args.data_dir, remove_outliers=args.remove_outliers
)


quartile_diff = batch_to_summary["IQR"]
std = batch_to_summary["std"]
IQR = pd.DataFrame.from_dict(quartile_diff, orient="index", columns=["IQR=75th-25th"])
std = pd.DataFrame.from_dict(std, orient="index", columns=["std"])
# normalize quartile_diff wrt to smallest quartile_diff
IQR["IQR_normalized"] = IQR["IQR=75th-25th"] / IQR["IQR=75th-25th"].min()
std["std_normalized"] = std["std"] / std["std"].min()
max_iqr = IQR["IQR_normalized"].max()
max_std = std["std_normalized"].max()
logging.info(
    f"Maximum normalized IQR for preprocessing time across configs:\n\t{max_iqr:.2f}X"
)
logging.info(
    f"Maximum normalized standard deviation for preprocessing time across configs:\n\t{max_std:.2f}X"
)

# print 25th and 75th percentile
logging.info("\nPreprocessing time stats...")
logging.info("25th percentile:")
for k,v in batch_to_summary['25th'].items():
    logging.info(f'\t{k}: {v}')
logging.info("50th percentile:")
for k,v in batch_to_summary['median'].items():
    logging.info(f'\t{k}: {v}')
logging.info("75th percentile:")
for k,v in batch_to_summary['75th'].items():
    logging.info(f'\t{k}: {v}')