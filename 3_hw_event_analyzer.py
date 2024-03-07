# %%
import pandas as pd, json, re, os, natsort
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# print Source Function column without truncation
pd.set_option('display.max_colwidth', None)
# display all columns
pd.set_option('display.max_columns', None)

# %%
mapping_file = '/mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/code/image_classification/analysis/hw_event_analysis/mapping_funcs.json'
# uarch_dir ='/mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/code/image_classification/analysis/hw_event_analysis/vtune_uarch_csvs'

# uarch_dir ='/mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/code/image_classification/analysis/hw_event_analysis/vtune_vary_dataloaders_csv'

uarch_dir = '/mydata/vtune_logs/pytorch_vtune_logs/csv/vtune_mem_access_vary_dataloader'
# uarch_dir = '/mydata/vtune_logs/pytorch_vtune_logs/csv/vtune_mem_access_unpinned'

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
    uarch_files.append(file)

# natsort uarch_files
uarch_files = natsort.natsorted(uarch_files)
print(uarch_files)


def plot_stacked_bar_chart(switch):
    combined_df = pd.DataFrame()
    # loop through uarch_dir and find csv files
    for uarch_file_ in uarch_files:
        if not file.endswith(".csv"):
            print("Files other than csv exist ", uarch_dir)
            exit(1)

        uarch_file = os.path.join(uarch_dir, uarch_file_)

        # %%
        #  read csv separated by tab
        df = pd.read_csv(uarch_file, sep=',')
        df['avg_cpu_freq'] = df['Hardware Event Count:CPU_CLK_UNHALTED.THREAD'] / df['Hardware Event Count:CPU_CLK_UNHALTED.REF_TSC'] * 3200000000
        df['CPU Time'] = df['Hardware Event Count:CPU_CLK_UNHALTED.THREAD'] / df['avg_cpu_freq']
        # create a new column called "CPU Time %" from "CPU Time" column
        df['CPU Time %'] = df['CPU Time'] / df['CPU Time'].sum() * 100

        # %%
        # sort by column 'CPU Time' and reset index
        df = df.sort_values(by=['CPU Time'], ascending=False).reset_index(drop=True)

        # %%
        #  find index of interested functions in the dataframe in "Source Function" column
        indices = {}
        empty_indices = []
        print('uarch_file: ', uarch_file)
        for func in interested_functions:
            # escape special characters
            func_ = re.escape(func)
            # indices_for_func = df[df["Source Function"].str.contains(func_)].index.values
            # instead of contains, use equals
            indices_for_func = df[df["Source Function"] == func].index.values
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
        #     df.loc[df['Source Function'].str.contains(function), 'Source Function'] = function


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
        remove_cols = ['Hardware Event Count:INST_RETIRED.ANY',
       'Hardware Event Count:CPU_CLK_UNHALTED.THREAD',
       'Hardware Event Count:CPU_CLK_UNHALTED.REF_TSC',
       'Hardware Event Count:CPU_CLK_UNHALTED.ONE_THREAD_ACTIVE',
       'Hardware Event Count:MEM_UOPS_RETIRED.ALL_LOADS_PS',
       'Hardware Event Count:MEM_UOPS_RETIRED.ALL_STORES_PS',
       'Hardware Event Count:CPU_CLK_UNHALTED.THREAD_P',
       'Hardware Event Count:MEM_TRANS_RETIRED.LOAD_LATENCY_GT_4',
       'Hardware Event Count:DTLB_LOAD_MISSES.STLB_HIT',
       'Hardware Event Count:DTLB_LOAD_MISSES.WALK_COMPLETED',
       'Hardware Event Count:DTLB_LOAD_MISSES.WALK_DURATION:cmask=1',
       'Hardware Event Count:DTLB_STORE_MISSES.STLB_HIT',
       'Hardware Event Count:DTLB_STORE_MISSES.WALK_COMPLETED',
       'Hardware Event Count:DTLB_STORE_MISSES.WALK_DURATION:cmask=1',
       'Hardware Event Count:IDQ_UOPS_NOT_DELIVERED.CORE',
       'Hardware Event Count:IDQ_UOPS_NOT_DELIVERED.CYCLES_0_UOPS_DELIV.CORE',
       'Hardware Event Count:INT_MISC.RECOVERY_CYCLES',
       'Hardware Event Count:L1D_PEND_MISS.FB_FULL:cmask=1',
       'Hardware Event Count:L2_RQSTS.RFO_HIT',
       'Hardware Event Count:LD_BLOCKS.NO_SR',
       'Hardware Event Count:LD_BLOCKS.STORE_FORWARD',
       'Hardware Event Count:LD_BLOCKS_PARTIAL.ADDRESS_ALIAS',
       'Hardware Event Count:MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_HITM_PS',
       'Hardware Event Count:MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_HIT_PS',
       'Hardware Event Count:MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_MISS_PS',
       'Hardware Event Count:MEM_LOAD_UOPS_L3_MISS_RETIRED.LOCAL_DRAM_PS',
       'Hardware Event Count:MEM_LOAD_UOPS_L3_MISS_RETIRED.REMOTE_DRAM_PS',
       'Hardware Event Count:MEM_LOAD_UOPS_L3_MISS_RETIRED.REMOTE_FWD_PS',
       'Hardware Event Count:MEM_LOAD_UOPS_L3_MISS_RETIRED.REMOTE_HITM_PS',
       'Hardware Event Count:MEM_LOAD_UOPS_RETIRED.HIT_LFB_PS',
       'Hardware Event Count:MEM_LOAD_UOPS_RETIRED.L1_HIT_PS',
       'Hardware Event Count:MEM_LOAD_UOPS_RETIRED.L1_MISS',
       'Hardware Event Count:MEM_LOAD_UOPS_RETIRED.L2_HIT_PS',
       'Hardware Event Count:MEM_LOAD_UOPS_RETIRED.L3_HIT_PS',
       'Hardware Event Count:MEM_LOAD_UOPS_RETIRED.L3_MISS_PS',
       'Hardware Event Count:MEM_UOPS_RETIRED.LOCK_LOADS_PS',
       'Hardware Event Count:MEM_UOPS_RETIRED.SPLIT_LOADS_PS',
       'Hardware Event Count:MEM_UOPS_RETIRED.SPLIT_STORES_PS',
       'Hardware Event Count:MEM_UOPS_RETIRED.STLB_MISS_LOADS_PS',
       'Hardware Event Count:MEM_UOPS_RETIRED.STLB_MISS_STORES_PS',
       'Hardware Event Count:OFFCORE_REQUESTS_BUFFER.SQ_FULL',
       'Hardware Event Count:OFFCORE_REQUESTS_OUTSTANDING.ALL_DATA_RD:cmask=4',
       'Hardware Event Count:OFFCORE_REQUESTS_OUTSTANDING.CYCLES_WITH_DATA_RD',
       'Hardware Event Count:OFFCORE_REQUESTS_OUTSTANDING.CYCLES_WITH_DEMAND_RFO',
       'Hardware Event Count:RESOURCE_STALLS.SB',
       'Hardware Event Count:RS_EVENTS.EMPTY_CYCLES',
       'Hardware Event Count:RS_EVENTS.EMPTY_END',
       'Hardware Event Count:UOPS_EXECUTED.CYCLES_GE_1_UOP_EXEC',
       'Hardware Event Count:UOPS_EXECUTED.CYCLES_GE_2_UOPS_EXEC',
       'Hardware Event Count:UOPS_EXECUTED.CYCLES_GE_3_UOPS_EXEC',
       'Hardware Event Count:UOPS_ISSUED.ANY',
       'Hardware Event Count:UOPS_RETIRED.RETIRE_SLOTS',
       'Hardware Event Count:Total_Latency_MEM_TRANS_RETIRED.LOAD_LATENCY_GT_4',
       'Function (Full)', 'Source File', 'avg_cpu_freq']
        # remove columns
        df2 = df2.drop(columns=remove_cols)

        # add column 'uarch_file' to df2 with value uarch_file
        tmp = uarch_file_.split('.csv')[0]
        if switch:
            tmp = tmp.split('_')
            tmp = tmp[1] + '_' + tmp[0]
        df2['config'] = tmp

        


        # %%
        # print sum of 'CPU Time (s)' column
        # print("Total CPU Time (s): ", df['CPU Time'].sum())

        # %%
        # set index to 'Source Function' column
        df2 = df2.set_index('Source Function')

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

    # save combined_df to csv
    combined_df.to_csv('/mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/code/image_classification/analysis/hw_event_analysis/raw_combined_df_vary_dataloaders.csv')

    # %%

    plt.rc('axes', labelsize=55)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=50)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=60)    # fontsize of the tick labels
    plt.rc('legend', fontsize=50)    # legend fontsize
    # increase size of axis label size
    plt.rcParams['axes.labelsize'] = 80
    # increase title size
    plt.rcParams['axes.titlesize'] = 80
    # increase space between axis label and axis
    plt.rcParams['axes.labelpad'] = 50
    # increase space between title and plot
    plt.rcParams['axes.titlepad'] = 50
    plt.rcParams['font.size'] = 50

    print(combined_df['Source Function'].head(len(combined_df)))


    func_rename = {
        'c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#3}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(float)#1}, at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#3}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>)#2}>>': 'AVX2::direct_copy_kernel(float)',
        'at::native::AVX2::vectorized_loop<at::native::(anonymous namespace)::div_true_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()(void) const::{lambda()#2}::operator()(void) const::{lambda(floatfloat)#1}&, at::native::(anonymous namespace)::div_true_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()(void) const::{lambda()#2}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>at::vec::AVX2::Vectorized<float>)#2}&>.isra.0' :'AVX2::div_true_kernel(float)',
        'c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#3}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda(unsigned char)#1}, at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::{lambda()#3}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<unsigned char>)#2}>>' : 'AVX2::direct_copy_kernel(unsigned char)',
        'c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::TensorIteratorBase::loop_2d_from_1d<at::native::AVX2::copy_kernel(at::TensorIterator&, bool)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda(char**long const*, long)#1}>(, signed char, at::native::AVX2::copy_kernel(at::TensorIterator&, bool)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda()#1}::operator()(void) const::{lambda(char**long const*, long)#1} const&)::{lambda(char**long const*, long, long)#1}>' : 'AVX2::copy_kernel(char**long const*, long, long)',
        'c10::function_ref<void (char**, long const*, long, long)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, c10::Scalar const&)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(floatfloat)#1}, at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, c10::Scalar const&)::{lambda()#1}::operator()(void) const::{lambda()#7}::operator()(void) const::{lambda(at::vec::AVX2::Vectorized<float>at::vec::AVX2::Vectorized<float>)#2}>>' : 'add_kernel(float)',
    }

    # replace function names in 'Source Function' column

    combined_df['Source Function'] = combined_df['Source Function'].replace(func_rename)


    for col in combined_df.columns:
        if col == 'config' or col == 'Source Function':
            continue
        # pivot table
        df_pivot = combined_df.pivot(index='config', columns='Source Function', values=col)
        # replace _ with space in 'Source Function' column
        df_pivot.columns = df_pivot.columns.str.replace('_', ' ')
        # sort by 'config' column
        df_pivot = df_pivot.sort_values(by=['config'])    
        # natsort 'config' column
        df_pivot.index = natsort.natsorted(df_pivot.index)
        ax = df_pivot.plot(kind='bar', stacked=True, figsize=(100,60))
        for c in ax.containers:
            ax.bar_label(c, label_type='center')
        # add x axis label
        plt.xlabel('Configurations')
        # rotate x axis labels
        plt.xticks(rotation=75)
        # add y axis label
        plt.ylabel(col)
        # add title to legend
        plt.title("C/C++ functions")    
        # reverse legend order
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1],loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.subplots_adjust(right=0.70)
        if switch:
            swapped_config = "format_gpu_b"
        else:
            swapped_config = "format_b_gpu"
        plt.savefig('/mydata/rbachkaniwala3/code/rajveerb-ml-pipeline-benchmark/raw_combined_vary_dataloaders/'+ swapped_config +'/'+col+'.png')
        plt.close()


# plot_stacked_bar_chart(True)
plot_stacked_bar_chart(False)