# %%
import pandas as pd, json, re, os, natsort
import matplotlib.pyplot as plt, argparse
# print Source Function / Function / Call Stack column without truncation
pd.set_option('display.max_colwidth', None)
# display all columns
pd.set_option('display.max_columns', None)

# create an argument parser for mapping_file and uarch_dir
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mapping_file",
    type=str,
    default="code/image_classification/LotusMap/Intel/mapping_funcs.json",
    help="Path to the mapping file",
)
parser.add_argument(
    "--uarch_dir",
    type=str,
    default="code/image_classification/analysis/combine_lotus/lotustrace_uarch",
    help="Path to the directory containing uarch csv files",
)

parser.add_argument(
    "--combined_hw_events",
    type=str,
    default="code/image_classification/analysis/combine_lotus/combined_lotustrace_uarch.csv",
    help="Path to save combined uarch csv files for all configurations",
)

parser.add_argument(
    "--cpp_hw_events_plot_dir",
    type=str,
    default="code/image_classification/analysis/combine_lotus/cpp_hw_events_figs",
    help="Path to save C++ functions hardware events plots",
)

args = parser.parse_args()

mapping_file = args.mapping_file
uarch_dir = args.uarch_dir

os.makedirs(args.cpp_hw_events_plot_dir, exist_ok=True)

with open(mapping_file) as f:
    data = json.load(f)

cpp_funcs = set()

for py_func in data['op_to_func']:
    # if py_func == 'RandomResizedCrop' or py_func == 'Loader':
    for cpp_func in data['op_to_func'][py_func]:
        cpp_funcs.add(cpp_func.split('|')[0])
interested_functions = list(cpp_funcs)


uarch_files = []
# loop through uarch_dir and find csv files
for file in os.listdir(uarch_dir):
    if not file.endswith(".csv"):
        print("Files other than csv exist ", uarch_dir)
        exit(1)

    if 'dataloader' not in file:
        print('File name should be like b1024_gpu4_dataloader8.csv, where 1024 is the batch size, 4 is the number of GPUs and 8 is the number of dataloaders')
    uarch_files.append(file)

# natsort uarch_files
uarch_files = natsort.natsorted(uarch_files)
print(uarch_files)


def plot_stacked_bar_chart():
    combined_df = pd.DataFrame()
    # loop through uarch_dir and find csv files
    for uarch_file_ in uarch_files:
        if not file.endswith(".csv"):
            print("Files other than csv exist ", uarch_dir)
            exit(1)

        uarch_file = os.path.join(uarch_dir, uarch_file_)

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
            # "Clockticks",
            "Instructions Retired",
            "CPI Rate",
            "Function (Full)"
        ] 
        # remove columns
        df2 = df2.drop(columns=remove_cols)

        # add column 'uarch_file' to df2 with value uarch_file
        tmp = uarch_file_.split('.csv')[0]

        df2['config'] = tmp


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

        # replace 'Clockticks' ',' with '' and cast the column to int
        df2['Clockticks'] = df2['Clockticks'].str.replace(',', '').astype(int)
        
        # multiply each column with 'CPU Time (s)' column in percentage_symbol_cols
        for col in percentage_symbol_cols:
            # option 1
            # df2[col] = df2[col] * df2['CPU Time (s)'] / 100
            # option 2 - more realistic because of https://github.com/intel/perfmon/blob/main/BDX/metrics/broadwellx_metrics.json
            df2[col] = df2[col] * df2['Clockticks']

        # remove 'Clockticks' column
        df2 = df2.drop(columns=['Clockticks'])


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
    # %%

    # reset index
    combined_df = combined_df.reset_index()

    combined_df = combined_df.rename(columns={"Source Function / Function / Call Stack": "Function"})
    # %%
    func_rename = {
        'c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#3}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(float)#1}, at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#3}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>)#2}>>': 'AVX2::direct_copy_kernel(float)',
        'at::native::AVX2::vectorized_loop<at::native::(anonymous namespace)::div_true_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()(void) const::{lambda()#2}::operator()(void) const::{lambda(floatfloat)#1}&, at::native::(anonymous namespace)::div_true_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()(void) const::{lambda()#2}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>at::vec::AVX2::Vectorized<float>)#2}&>.isra.0' :'AVX2::div_true_kernel(float)',
        'c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#3}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda(unsigned char)#1}, at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#3}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<unsigned char>)#2}>>' : 'AVX2::direct_copy_kernel(unsigned char)',
        'c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::TensorIteratorBase::loop_2d_from_1d<at::native::AVX2::copy_kernel(at::TensorIterator&, bool)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda(char**long const*, long)#1}>(, signed char, at::native::AVX2::copy_kernel(at::TensorIterator&, bool)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda(char**long const*, long)#1} const&)::{lambda(char**long const*, long, long)#1}>' : 'AVX2::copy_kernel(char**long const*, long, long)',
        'c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, c10::Scalar const&)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(floatfloat)#1}, at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, c10::Scalar const&)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>at::vec::AVX2::Vectorized<float>)#2}>>' : 'add_kernel(float)',
    }

    # replace function names in 'Source Function / Function / Call Stack' column

    combined_df['Function'] = combined_df['Function'].replace(func_rename)

    # %%

    plt.rc('xtick', labelsize=60)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=60)    # fontsize of the tick labels
    plt.rc('legend', fontsize=30)    # legend fontsize
    # increase size of axis label size
    plt.rcParams['axes.labelsize'] = 60
    # increase space between axis label and axis
    plt.rcParams['axes.labelpad'] = 60
    # increase title size
    plt.rcParams['axes.titlesize'] = 60
    # increase space between title and plot
    plt.rcParams['axes.titlepad'] = 60
    # increase padding between ticks and their labels
    plt.rcParams['xtick.major.pad'] = 60
    plt.rcParams['ytick.major.pad'] = 60

    print(combined_df['Function'].head(len(combined_df)))

    print(combined_df.index)
    print(combined_df.columns)
    # if combined_df config contains dataloader    
    if 'dataloader' in combined_df['config'][0]:
        # config has values in fiormat b1024_gpu4_dataloader8, keep only 8 as int
        combined_df['config'] = combined_df['config'].str.split('dataloader').str[1].astype(int)

    # merge duplicate rows in percentage_symbol_cols
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
    
    for col in percentage_symbol_cols:
        combined_df[col] = combined_df.groupby(['config'])[col].transform('sum')
        combined_df[col] = combined_df[col] / combined_df['CPU Time (s)'] * 100
    
    # remove duplicate rows across all columns
    combined_df = combined_df.drop_duplicates()
        

    print('*************************')


    combined_df.to_csv(args.combined_hw_events)

    # escape udnerscore escape characters in 'Function' column
    combined_df['Function'] = combined_df['Function'].str.replace('_', '\_')
    

    for col in combined_df.columns:
        if col == 'config' or col == 'Function':
            continue
        print(col)
        # pivot table
        df_pivot = combined_df.pivot(index='config', columns='Function', values=col)
        # sort the index using natsort
        df_pivot = df_pivot.reindex(natsort.natsorted(df_pivot.index))
        

        ax = df_pivot.plot(kind='bar', stacked=True, figsize=(75,30), colormap='jet')
        # for c in ax.containers:
        #     ax.bar_label(c, label_type='center')
        # add x axis label
        plt.xlabel('Dataloaders')
        # rotate x axis labels
        plt.xticks(rotation=0)
        # add y axis label
        plt.ylabel(col)
        # add title to legend
        plt.title(f"Hardware metric breakdown by preprocessing operations")    
        # reverse legend order
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1],loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=2)
        
        # remove legend
        # plt.legend().remove()
        # tight layout
        plt.tight_layout()
        plt.subplots_adjust(right=0.60)

        plt.savefig(os.path.join(args.cpp_hw_events_plot_dir, col+'.png'))
        plt.close() 

plot_stacked_bar_chart()