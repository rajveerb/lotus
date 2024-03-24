import pandas as pd
import numpy as np
import os, natsort
import argparse
import logging

# take below arguments using argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    default="/mydata/P3Tracer/P3Torch_imagenet_vary_batch_and_gpu",
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
    default="code/image_classification/analysis/P3Torch_imagenet_vary_batch_and_gpu/log_stats/preprocessing_time_stats.log",
    help="Output file to save the stats",
)

args = parser.parse_args()

logging.basicConfig(filename=args.output_file, level=logging.INFO, filemode="w", format="")


def log_stats_to_file(plot_df):
    mean = np.mean(plot_df["duration"])
    std = np.std(plot_df["duration"])
    std_percentage_of_mean = 100 * std / mean
    minimum = plot_df["duration"].min()
    maximum = plot_df["duration"].max()
    total_preprocessing_time = np.sum(plot_df["duration"])
    median = np.median(plot_df["duration"])
    percentile_25 = np.percentile(plot_df["duration"], 25)
    percentile_75 = np.percentile(plot_df["duration"], 75)

    logging.info(
        f"\tsum = {total_preprocessing_time:.2f} ms,\n\tavg = {mean:.2f} ms,\n\tstd = {std:.2f} ({std_percentage_of_mean:.2f}% of avg) ms,\n\tmin = {minimum:.2f} ms,\n\tmax = {maximum:.2f} ms"
    )
    logging.info(
        f"\tmedian = {median:.2f} ms,\n\t25th percentile = {percentile_25:.2f} ms,\n\t75th percentile = {percentile_75:.2f} ms"
    )
    logging.info(f"\t75th - 25th = {percentile_75 - percentile_25:.2f} ms")


def plotter_preprocessing_time(target_dir, remove_outliers):

    root_to_files = {}
    for root, dirs, files in os.walk(target_dir):
        root_to_files[root] = files
    roots = sorted(root_to_files, key=lambda x: natsort.natsort_key(x.lower()))

    for root in roots:
        if "e2e" in root:
            continue
        logging.info("\n\n")
        logging.info(root)
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
            # mean and std before removing outliers
            logging.info(f"\tBefore removing outliers:")
            log_stats_to_file(plot_df)

            q1 = plot_df["duration"].quantile(0.25)
            q3 = plot_df["duration"].quantile(0.75)
            iqr = (q3 - q1) * 2

            # remove outliers less than q1 - iqr only (these are numbers from last batch which has
            #  elements less than batch size because elements in a dataset may not be a multiple of batch size)
            logging.info("\tRemove outliers less than q1 - iqr, where iqr = (q3 - q1) * 2")
            plot_df = plot_df[~(plot_df["duration"] < (q1 - iqr))]
            # mean and std after removing outliers
            logging.info(f"\tAfter removing outliers:")
            return plot_df

        label = root.split("/")[-1]  # retrieves b128_gpu4 kind of label
        logging.info(f"{label}:")
        # remove outliers
        if remove_outliers:
            plot_df = remove_wild_outliers(plot_df)

        log_stats_to_file(plot_df)


logging.info("Preprocessing time stats:")
plotter_preprocessing_time(args.data_dir, remove_outliers=args.remove_outliers)
logging.info("----------------------------------------------\n")
