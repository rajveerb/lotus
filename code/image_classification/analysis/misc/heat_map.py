# %%
import pandas as pd, json, re, os, natsort
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# print Source Function / Function / Call Stack column without truncation
pd.set_option('display.max_colwidth', None)
# display all columns
pd.set_option('display.max_columns', None)


plt.rc('axes', labelsize=63)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=60)    # fontsize of the tick labels
plt.rc('ytick', labelsize=60)    # fontsize of the tick labels
plt.rc('legend', fontsize=50)    # legend fontsize
# # increase size of axis label size
plt.rcParams['axes.labelsize'] = 65
# increase title size
plt.rcParams['axes.titlesize'] = 65
# increase space between axis label and axis
plt.rcParams['axes.labelpad'] = 50
# increase space between title and plot
plt.rcParams['axes.titlepad'] = 50
plt.rcParams['font.size'] = 50

# %%
mapping_file = '/mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/code/image_classification/analysis/hw_event_analysis/mapping_funcs.json'
uarch_dir ='/mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/code/image_classification/analysis/hw_event_analysis/vtune_uarch_csvs'

# uarch_dir ='/mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/code/image_classification/analysis/hw_event_analysis/vtune_vary_dataloaders_csv'


# %%
# load a json file
with open(mapping_file) as f:
    data = json.load(f)

cpp_funcs = set()

for py_func in data['op_to_func']:
    # if py_func == 'RandomResizedCrop' or py_func == 'convertRGB':
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
    uarch_files.append(file)

# natsort uarch_files
uarch_files = natsort.natsorted(uarch_files)
print(uarch_files)


def plot_heatmap():
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
        # print("C/C++ functions not found in dataframe:")
        # print(empty_indices)
        # print("C/C++ functions (indices) found in dataframe:")    
        # for func in indices:
        #     print("Index:",indices[func],"Function: ", func)
        # print('\n\n')

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

        # add column 'uarch_file' to df2 with value uarch_file
        tmp = uarch_file_.split('.csv')[0]
        batch_size,gpus = tmp.split('_')
        batch_size = batch_size.split('b')[1]
        gpus = gpus.split('gpu')[1]
        df2['batch_size'] = int(batch_size)
        df2['gpus'] = int(gpus)

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
        # print sum of 'CPU Time %' contributed
        # print config i.e. batch_size and gpus
        print(f"{df2['CPU Time %'].sum():.2f}%")
        
        # rename 'Source Function / Function / Call Stack' column to 'Functions'
        df2 = df2.rename(columns={"Source Function / Function / Call Stack": "Functions"})
        # %%
        # set index to 'Source Function / Function / Call Stack' column
        df2 = df2.set_index('Functions')

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
    # print(combined_df.columns)
    # print(combined_df.head(len(combined_df)))
    # %%

    # %%


    # print(combined_df['Functions'].head(len(combined_df)))
    # # print index of combined_df
    # print(combined_df.index)
    # # print column names of combined_df
    # print(combined_df.columns)


    func_rename = {
        'c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#3}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(float)#1}, at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#3}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>)#2}>>': 'AVX2::direct_copy_kernel(float)',
        'at::native::AVX2::vectorized_loop<at::native::(anonymous namespace)::div_true_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()(void) const::{lambda()#2}::operator()(void) const::{lambda(floatfloat)#1}&, at::native::(anonymous namespace)::div_true_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()(void) const::{lambda()#2}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>at::vec::AVX2::Vectorized<float>)#2}&>.isra.0' :'AVX2::div_true_kernel(float)',
        'c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#3}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda(unsigned char)#1}, at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#3}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<unsigned char>)#2}>>' : 'AVX2::direct_copy_kernel(unsigned char)',
        'c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::TensorIteratorBase::loop_2d_from_1d<at::native::AVX2::copy_kernel(at::TensorIterator&, bool)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda(char**long const*, long)#1}>(, signed char, at::native::AVX2::copy_kernel(at::TensorIterator&, bool)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda(char**long const*, long)#1} const&)::{lambda(char**long const*, long, long)#1}>' : 'AVX2::copy_kernel(char**long const*, long, long)',
        'c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, c10::Scalar const&)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(floatfloat)#1}, at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, c10::Scalar const&)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>at::vec::AVX2::Vectorized<float>)#2}>>' : 'add_kernel(float)',
    }

    # replace function names in 'Source Function / Function / Call Stack' column

    combined_df['Functions'] = combined_df['Functions'].replace(func_rename)  

    # Remove functions taking less than 1% of CPU total time
    functions_to_remove = combined_df[combined_df.groupby('Functions')['CPU Time %'].transform('min') < 1]['Functions']
    
    print("Below are functions taking less than 1% of CPU total time in at least one configuration:")
    print(functions_to_remove.unique())

    # print high level functions these functions belong to
    print("Below are high level functions these functions belong to:")
    for func in functions_to_remove.unique():
        for py_func in data['op_to_func']:
            for cpp_func in data['op_to_func'][py_func]:
                if cpp_func.split('|')[0] == func:
                    print('\t' + py_func)
                    break

    # remove functions in functions_to_remove from combined_df
    combined_df = combined_df[~combined_df['Functions'].isin(functions_to_remove)]

    # plot a heatmap for each function wrt to batch_size and gpus column with values from every other column

    for func in combined_df['Functions'].unique():
        # print(func)

        tmp_df = combined_df[combined_df['Functions'] == func]
        # print(tmp_df.head(len(tmp_df)))

        fig, axes = plt.subplots(7,2,figsize=(30 * 2, 25 * 7))
        for subplot_index,col in enumerate(tmp_df.columns):
            if col == 'batch_size' or col == 'gpus' or col == 'Functions':
                continue
            # pivot table
            df_pivot = tmp_df.pivot(index='batch_size', columns='gpus', values=col)

            # reindex df_pivot with sorted index
            df_pivot = df_pivot.reindex(sorted(df_pivot.index, reverse=True), axis=0)
            # heatmap
            ax = sns.heatmap(df_pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[(subplot_index-1)%7,(subplot_index-1)//7])
            # add title to heatmap
            axes[(subplot_index-1)%7,(subplot_index-1)//7].set_title('Metric: ' + col + '\n' + 'Function: ' + func)
        # tight layout
        plt.tight_layout()
        # fix axis labels based on the actual values in the dataframe
        # create directory for func
        plt.savefig('/mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/heatmap_figs/'+func+'.png')
        plt.close()



plot_heatmap()