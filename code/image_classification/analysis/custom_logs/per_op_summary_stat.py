# %%
import os,argparse
import pandas as pd

# %%

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str,\
                     default='/mydata/pytorch_custom_log_one_epoch_imagenet_dataset/',\
                        help="path to the directory containing the log files")
# for percentile
parser.add_argument("--percentile", type=float, default=0.9,\
                        help="percentile to be calculated (between 0 and 1)")
args = parser.parse_args()


# percentile function
def percentile_func(x):
    return x.quantile(args.percentile)

def generate_summary_stats_per_op(data_dir):
    # recursively find all the log files
    for root, dirs, files in os.walk(data_dir):
        print(root)
        combine_df = pd.DataFrame()
        for file in files:
            if 'worker_pid' in file:
                fp = os.path.join(root, file)
                #  read the log file as csv with no header
                df = pd.read_csv(fp, header=None)
                df.columns = ["name", "start", "dur"]

                # drop columns with name "start"
                df = df.drop(columns=["start"])

                # divide "dur" column by 1000,000 to get time in ms
                df["dur"] = df["dur"] / 1000000
                # discard the rows with name starting with SBatch
                df = df[~df["name"].str.startswith("SBatch")]

                # concatenate all the dataframes
                combine_df = pd.concat([combine_df, df])

        # check if the dataframe is empty
        if combine_df.empty:
            continue

        # print mean, std and P99 for each operation
        print(f"Percentile: {args.percentile}")
        print(combine_df.groupby("name").agg(["mean", "median", percentile_func, "std"]))
        print("\n\n")

        # %%
        print("% of each operation with duration < 10 ms")
        print(pd.DataFrame(combine_df[combine_df["dur"] < 10].groupby("name").count() * 100 / combine_df.groupby("name").count()).fillna(0))
        print("\n\n")

        # %%
        print("% of each operation with duration < 100 us")
        print(pd.DataFrame(combine_df[combine_df["dur"] < 0.1].groupby("name").count() * 100 / combine_df.groupby("name").count()).fillna(0))
        print("--------------------------------------------------\n\n")

generate_summary_stats_per_op(args.data_dir)