import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, natsort
import argparse

plt.rc("axes", labelsize=80)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=70)  # fontsize of the tick labels
plt.rc("ytick", labelsize=70)  # fontsize of the tick labels
plt.rc("legend", fontsize=60)  # legend fontsize

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
    default="lotustrace_result/box_plot_preprocessing_time.png",
    help="Output file to save the stats",
)

args = parser.parse_args()


def plotter_preprocessing_time(target_dir, fig_path, remove_outliers, fig_size=(50, 25)):

    root_to_files = {}
    for root, dirs, files in os.walk(target_dir):
        root_to_files[root] = files
    roots = sorted(root_to_files, key=lambda x: natsort.natsort_key(x.lower()))
    plot_df = pd.DataFrame()
    plot_dfs = []
    configs = []

    for root in roots:
        if "e2e" in root:
            continue
        print(root)
        files = root_to_files[root]
        config_df = pd.DataFrame()

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

            batch_config = root.split("/")[-1]

            # add batch_config to configs
            if batch_config not in configs:
                configs.append(batch_config)
            config_df = pd.concat([config_df, df])

        # concatentate all dataframes
        if config_df.empty:
            continue

        # rename 'duration' to 'preprocessing_time'
        config_df = config_df.rename(columns={"duration": batch_config})

        # drop columns 'name' and 'start_ts'
        config_df = config_df.drop(columns=["name", "batch_id", "start_ts"])
        # reset index
        config_df = config_df.reset_index(drop=True)
        
        if remove_outliers:
            # get first quartile of config_df
            q1 = config_df.quantile(0.25)
            q3 = config_df.quantile(0.75)
            iqr = (q3 - q1) * 2

            # remove outliers less than q1 - iqr only (these are numbers from last batch which has
            #  elements less than batch size because elements in a dataset may not be a multiple of batch size)

            config_df = config_df[~(config_df < (q1 - iqr))]

        plot_dfs.append(config_df)

    plot_df = pd.concat(plot_dfs, axis=1, ignore_index=True)

    # reset index of plot_df
    plot_df = plot_df.reset_index(drop=True)

    # set column names to configs
    plot_df.columns = configs

    plot_df.boxplot(
        figsize=fig_size,
        medianprops=dict(linestyle="-", linewidth=5),
        boxprops=dict(linestyle="-", linewidth=5),
        whiskerprops=dict(linestyle="-", linewidth=7),
        capprops=dict(linestyle="-", linewidth=7),
        flierprops=dict(
            marker="o",
            markersize=20,  # fill it with color no empty circles
            markerfacecolor="r",
            linestyle="none",
            markeredgecolor="g",
        ),
    )

    plt.gcf().set_size_inches(fig_size[0], fig_size[1])
    # label x axis
    plt.xlabel(f"Configurations", labelpad=40)
    # label y axis
    plt.ylabel("Preprocessing time in ms", labelpad=40)
    plt.tick_params(axis="y", which="major", pad=40)
    plt.tick_params(axis="x", which="major", pad=40)
    # rotate x ticks
    plt.xticks(rotation=90)
    # add legend to bottom right and increase size of legend markers
    plt.legend(loc="lower right", markerscale=4)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.clf()


plotter_preprocessing_time(args.data_dir, args.output_file, args.remove_outliers)
