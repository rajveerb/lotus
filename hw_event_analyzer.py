# %%
import pandas as pd, json, re, os, natsort
import matplotlib.pyplot as plt
# print Source Function / Function / Call Stack column without truncation
pd.set_option('display.max_colwidth', None)
# display all columns
pd.set_option('display.max_columns', None)

# %%
mapping_file = '/mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/code/image_classification/analysis/hw_event_analysis/mapping_funcs.json'
uarch_dir ='/mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/code/image_classification/analysis/hw_event_analysis/vtune_uarch_csvs'

# %%
# load a json file
with open(mapping_file) as f:
    data = json.load(f)

cpp_funcs = set()

for py_func in data['op_to_func']:
    for cpp_func in data['op_to_func'][py_func]:
        cpp_funcs.add(cpp_func.split('|')[0])
interested_functions = list(cpp_funcs)

# previous interested_functions =

# ["__memmove_avx_unaligned_erms",\
# "_int_free",\
# "ImagingResampleHorizontal_8bpc",\
# "ImagingResampleVertical_8bpc",\
# "ImagingFlipLeftRight",\
# "ImagingPackRGB",\
# "munmap",\
# "copy_kernel",\
# "div_true_kernel",\
# "direct_copy_kernel",\
# "add_kernel",\
# "decompress_onepass",\
# "jpeg_idct_islow",\
# "jpeg_idct_16x16",\
# "ycc_rgb_convert",\
# "decode_mcu",\
# "ImagingUnpackRGB",\
# "__memset_avx2_unaligned_erms",\
# "__libc_calloc",\
#         ]
uarch_files = []
# loop through uarch_dir and find csv files
for file in os.listdir(uarch_dir):
    if not file.endswith(".csv"):
        print("Files other than csv exist ", uarch_dir)
        exit(1)
    uarch_files.append(os.path.join(uarch_dir, file))

# natsort uarch_files
uarch_files = natsort.natsorted(uarch_files)
print(uarch_files)

combined_df = pd.DataFrame()
# loop through uarch_dir and find csv files
for uarch_file in uarch_files:
    if not file.endswith(".csv"):
        print("Files other than csv exist ", uarch_dir)
        exit(1)

    # %%
    #  read csv separated by tab
    df = pd.read_csv(uarch_file, sep='\t')

    # %%
    # remove trailing "s" in 'CPU Time' column and cast the column to float
    df['CPU Time'] = df['CPU Time'].str.rstrip('s').astype(float)
    # create a new column called "CPU Time %" from "CPU Time" column
    df['CPU Time %'] = df['CPU Time'] / df['CPU Time'].sum() * 100

    # %%
    # sort by column 'CPU Time' and reset index
    df = df.sort_values(by=['CPU Time'], ascending=False).reset_index(drop=True)

    # %%
    #  find index of interested functions in the dataframe in "Source Function / Function / Call Stack" column
    indices = {}
    empty_indices = []
    print('uarch_file: ', uarch_file)
    for func in interested_functions:
        # escape special characters
        func_ = re.escape(func)
        indices_for_func = df[df["Source Function / Function / Call Stack"].str.contains(func_)].index.values
        # if empty, add to empty_indices
        if len(indices_for_func) == 0:
            empty_indices.append(func)
        else:
            indices[func] = indices_for_func
    print("C/C++ functions not found in dataframe:")
    print(empty_indices)
    print("C/C++ functions (indices) found in dataframe:")    
    for func in indices:
        print("Index:",indices[func],"Function: ", func)
    print('\n\n')

    # %%
    # find above functions in the dataframe and map the function name to the one in interested_functions

    # for function in interested_functions:
    #     df.loc[df['Source Function / Function / Call Stack'].str.contains(function), 'Source Function / Function / Call Stack'] = function


    # %%
    # combine all the interested functions' row into a dataframe
    df2 = pd.DataFrame()
    for func in indices:
        df2 = pd.concat([df2,df.iloc[indices[func]]])

    # %%
    # sort by 'CPU Time' column and reset index
    df2 = df2.sort_values(by=['CPU Time %'], ascending=False).reset_index(drop=True)
    # rename 'CPU Time' column to 'CPU Time (s)'
    df2 = df2.rename(columns={"CPU Time": "CPU Time (s)"})
    remove_cols = [
        "Source File",
        "Start Address",
        "Module",
        "Average CPU Frequency",
        "Clockticks",
        "Instructions Retired",
        "CPI Rate",
        "Function (Full)"
    ] 
    # remove columns
    df2 = df2.drop(columns=remove_cols)

    percentage_symbol_cols = [
        'Retiring',
        'Front-End Bound', 
        'Bad Speculation', 
        'L1 Bound', 
        'L2 Bound',
        'L3 Bound', 
        'Memory Bandwidth', 
        'Local DRAM', 
        'Remote DRAM',
        'Remote Cache', 
        'Store Bound', 
        'Core Bound',
        ]
    
    # remove the % symbol and cast the column to float
    for col in percentage_symbol_cols:
        df2[col] = df2[col].str.rstrip('%').astype(float)


    # %%
    # print sum of 'CPU Time (s)' column
    # print("Total CPU Time (s): ", df['CPU Time'].sum())

    # %%
    # set index to 'Source Function / Function / Call Stack' column
    df2 = df2.set_index('Source Function / Function / Call Stack')

    def flatten(S):
        if len(S)==1:
            return [S[0]]
        if isinstance(S[0], list):
            return flatten(S[0]) + flatten(S[1:])
        return S[:1] + flatten(S[1:])

    combined_df = pd.concat([combined_df, df2])
    combined_df = combined_df.groupby(level=0).agg(flatten)

# reset index
# combined_df = combined_df.reset_index()
# print(combined_df.columns)
# print(combined_df.head(len(combined_df)))
# remove CPU time column
combined_df = combined_df.drop(columns=['CPU Time (s)'])
#  %%
plt.figure(figsize=(20, 10))

# fix batch size vary gpus 
for index, row in combined_df.iterrows():
    print(index)
    # print mean, min, max of cols
    for col in combined_df.columns:
        print(f"\t{col}")
        # print row[col]
        print(f"\t\t-> {row[col]}")
        # convert row[col] to series
        series = pd.Series(row[col])
        # set plot fig size

        # plot uarch_file as x axis and row[col] as y axis
        plt.plot(uarch_files, series)
    plt.xticks(rotation=90)
    if len(index) > 50:
        index = index[:50]
        # save fig on path
    plt.legend(combined_df.columns,loc='upper right')
    plt.savefig(f"/mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/tempo/{index}_{col}.png")

        # print(f"\t\tMean: {sum(row[col])/len(row[col]):.2f}")
        # print(f"\t\tMin: {min(row[col])}")
        # print(f"\t\tMax: {max(row[col])}")
# %%
