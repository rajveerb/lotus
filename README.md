
# Lotus

<p align="center">
    <img src="assets/lotus.png"  width="20%" height="20%" alt="Lotus">
</p>

<p align="center"><i><b>A profiling
tool for the preprocessing stage of machine learning pipelines</b></i></p>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13245169.svg)](https://doi.org/10.5281/zenodo.13245169)

We introduce **Lotus**, a profiling tool for machine learning (ML) preprocessing pipelines defined using PyTorch's DataLoader.

**Lotus** is an easy-to-use, low overhead, and visualization-ready profiler specialized for the widely used PyTorch DataLoader preprocessing library.

## News:

- [Oct 2024] Lotus presented to **Intel Process Architecture Research (PAR)** Lab
- [Sep 2024] Lotus accepted to [**HotInfra 2024** (co-located with SOSP'24)](https://hotinfra24.github.io/) - [[PDF]](https://www.rajveerbachkaniwala.com/papers/lotus-hotinfra24.pdf)!
- [Aug 2024] Lotus won a 🏆 Best Paper Nomination in [**IISWC 2024**](https://iiswc.org/iiswc2024/)!
- [Aug 2024] Lotus artifact won Available, Reviewed, and Reproduced badges according to [IEEE Badges](https://ieeexplore.ieee.org/Xplorehelp/overview-of-ieee-xplore/about-content#reproducibility-badges)!
- [Jul 2024] Lotus accepted to [**IISWC 2024**](https://iiswc.org/iiswc2024/) - [[PDF](https://www.rajveerbachkaniwala.com/papers/lotus-iiswc24.pdf)]!

## Quick links
- [About Lotus](#about-Lotus)
- [Cite Lotus](#cite-lotus)
- [Replicate IISWC24 paper experiments](#Replicate-IISWC24-paper-experiments)
- [Get Lotus](#how-to-get-Lotus)
- [Use Lotus](#use-Lotus)
    - [How to use LotusTrace](#how-to-use-LotusTrace)
    - [Visualize Lotus' trace](#how-to-visualize-Lotus-trace)
    - [How to use LotusMap](#how-to-use-LotusMap)
- [Concrete examples](#concrete-examples)
    - [Example for LotusTrace](#example-for-LotusTrace)
    - [Example for LotusMap](#example-for-LotusMap)
- [Limitations of Lotus](#limitations-of-Lotus)
- [Acknowledgment](#acknowledgment)
- [License](#license)
- [Contact](#Contact)



## About Lotus

Lotus employs two novel approaches:

1. **LotusTrace** - An instrumentation methodology for the PyTorch library, which enables fine-grained elapsed time profiling with minimal time and storage overheads. 
2. **LotusMap** - A mapping methodology to reconstruct a mapping between Python functions and the underlying C++ functions they call, effectively linking high-level Python functions with low-level hardware counters. 

Above combination is powerful as it allows enables users to better reason about their pipeline’s performance, both at the level of preprocessing operations and their performance on hardware resource usage.

## Cite Lotus

```latex
@INPROCEEDINGS{lotus-iiswc24,
 title={{Lotus: Characterization of Machine Learning Preprocessing Pipelines via Framework and Hardware Profiling}}, 
 author={Bachkaniwala, Rajveer and Lanka, Harshith and Rong, Kexin and Gavrilovska, Ada},
 booktitle={2024 IEEE International Symposium on Workload Characterization (IISWC)},
 year={2024}
}

@INPROCEEDINGS{lotus-hotinfra24,
 title={{Lotus: Characterize Architecture Level CPU-based Preprocessing in Machine Learning Pipelines}}, 
 author={Bachkaniwala, Rajveer and Lanka, Harshith and Rong, Kexin and Gavrilovska, Ada},
 booktitle={The 2nd Workshop on Hot Topics in System Infrastructure (HotInfra’24), co-located with SOSP’24, November 3, 2024, Austin, TX, USA},
 year={2024}
}
```

## Replicate IISWC24 paper experiments

For replicating the key experiments in our paper presented at the [2024 IEEE International Symposium on Workload Characterization (IISWC'24)](https://iiswc.org/iiswc2024/), refer to the `SETUP.md` and `REPLICATE.md` files. You can also refer to the appendix of our paper.

## How to get Lotus

1. Clone this repository
2. Get submodules:

    ```git
    git submodule update --init --recursive
    ```
3. Create a conda environment

    ```bash
    conda create -n Lotus python=3.10
    conda activate Lotus
    ```
4. Install Intel VTune from [here](https://www.intel.com/content/www/us/en/docs/vtune-profiler/installation-guide/2023-1/overview.html) and activate it as Intel descsribes.

    Note: we used `Intel(R) VTune(TM) Profiler 2023.2.0 (build 626047)`
5. Install AMD uProf from [here](https://www.amd.com/en/developer/uprof/uprof-archives.html)

    Note: we used `AMDuProfCLI Version 4.0.341.0`
6. Install CUDA 11.8 from [here](https://developer.nvidia.com/cuda-11-8-0-download-archive) and CuDNN 8.7.0 from [here](https://developer.nvidia.com/rdp/cudnn-archive)
7. Follow the **LotusTrace** build instructions in `code/LotusTrace/README.md`
8. Follow the **itt-python** build instructions in `code/itt-python/README.md`
9. Follow the **amduprofile-python** build instructions in `code/amdprofilecontrol-python/README.md`
10. That's it!

## Use Lotus

### How to use LotusTrace


**LotusTrace** can be enabled by simply passing a `custom_log_file` to be used by **LotusTrace** using keywords `log_transform_elapsed_time` and `log_file` as shown below:

```python
import torchvision.transforms as transforms
import torchvision.datasets as datasets
custom_log_file = <To use our instrumentation>
train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    ], log_transform_elapsed_time=custom_log_file), 
    log_file=custom_log_file
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=(train_sampler is None) and args.shuffle,
    num_workers=args.workers,
    pin_memory=True,
    sampler=train_sampler,
)
```

But, what if you have a custom dataset?

We do support **LotusTrace** for custom datasets as well check below instance:


```python
import torchvision.transforms as transforms
log_file = <To use our instrumentation>
transforms = transforms.Compose([
  op1(), op2(), op3(), op4()], 
  log_transform_elapsed_time=log_file)
class CustomDataset:
  def __init__(self, log_file = None, transforms):
    ...
    self.log_file = log_file # If None, then no logging
    self.transforms = transforms # A Compose object
    ...
  def __getitem__(self, index):
    ...
    data,label = self.transforms(index) # Calls Compose's __call__()
    ...
    return data, label
dataset = CustomDataset(log_file = log_file, transforms = transforms)
```

You simply need to add `self.log_file` and `self.transforms` variable in `__init__` function of your custom dataset object as shown above. 
Moreover, you need to structure the code such that you use torchvision's `Compose` class' object to perform preprocessing operations as shown in `self.transforms(index)` line. That's it!

### How to visualize Lotus' trace

The trace generated by **LotusTrace** will be stored in the directory of the `log_file` as mentioned in [How to use LotusTrace](#how-to-use-LotusTrace). To generate a visualization ready trace from **LotusTrace**'s trace run the below command:

```bash
python code/visualize_LotusTrace_trace/visualization_augmenter.py \
    --LotusTrace_trace_dir <LotusTrace_trace_dir> \
    --coarse \
    --output_LotusTrace_viz_file <viz_file_path>
```

Note: `--coarse` option is great option for a quick high level view. Visualization trace will be stored in the same directory as `<LotusTrace_trace_dir>`. You can open this trace in your chrome browser with URL set to `chrome://tracing/` and simply upload the file using `Load` button.

For more options:
```bash
python code/visualize_LotusTrace_trace/visualization_augmenter.py \
    --help 
```

### How to use LotusMap

#### For Intel VTune:

Below is an example of how to write a python file called RandomResizedCrop.py such that using **LotusMap**'s method can be applied to collect the mapping: 

```python
import torchvision.transforms as t
from PIL import Image
import time,itt
# increase PIL image open size
Image.MAX_IMAGE_PIXELS = 1000000000
image_file = "<path to image>"
for i in range(5):
  # Open the image
  image = Image.open(image_file)
  # convert to RGB like torch's pil_loader
  image = image.convert('RGB') # Responisble for Loader operation
  # Define the desired crop size
  crop_size = 224  # Define this as needed
  time.sleep(1)  # sleep for 1 sec
  if i == 4: # Delay collection to prevent cold start
    itt.resume()
  image = t.RandomResizedCrop(crop_size)(image)
  if i == 4:
    itt.detach()
```

Now, run below commands to collect mapping:

```bash
vtune -collect hotspots -start-paused \
    -result-dir <your_vtune_result_dir> \
    -- python RandomResizedCrop.py
vtune -report hotspots \ 
    -result-dir <your_vtune_result_dir> \
    -format csv \
    -csv-delimiter comma \
    -report-output RandomResizedCrop.csv
```

`RandomResizedCrop.csv` contains the C/C++ functions mapped to `RandomResizedCrop` operation.

#### For AMD uProf:

Below is an example of how to write a python file called RandomResizedCrop.py such that using **LotusMap**'s method can be applied to collect the mapping: 

```python
import torchvision.transforms as t
from PIL import Image
import time, amdprofilecontrol as amd
# increase PIL image open size
Image.MAX_IMAGE_PIXELS = 1000000000
image_file = "<path to image>"
for i in range(5):
  # Open the image
  image = Image.open(image_file)
  # convert to RGB like torch's pil_loader
  image = image.convert('RGB') # Responisble for Loader operation
  # Define the desired crop size
  crop_size = 224  # Define this as needed
  time.sleep(1)  # sleep for 1 sec
  if i == 4: # Delay collection to prevent cold start
    amd.resume(1)
  image = t.RandomResizedCrop(crop_size)(image)
  if i == 4:
    amd.pause(1)
```

Now, run below commands to collect mapping:

```bash
AMDuProfCLI collect --config tbp --start-paused \
 --output-dir <your_uprof_result_dir> \
 python RandomResizedCrop.py

AMDuProfCLI report \
 --input-dir <your_uprof_generated_result_dir> \ 
 --report-output RandomResizedCrop.csv \
 --cutoff 100 -f csv #can be set to more than 100 
```

`RandomResizedCrop.csv` contains the C/C++ functions mapped to `RandomResizedCrop` operation.

*Note*: For completeness, checkout our paper to navigate how to correctly use **LotusMap** methodology.

## Concrete examples

### Example for LotusTrace

An example of how to enable **LotusTrace** facilitated logging for an image classification task has been described in `code/image_classification/code/pytorch_main.py`, we add the snippet below for the same:

```python
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ],	
        log_transform_elapsed_time=args.log_train_file,
    ),
    log_file=args.log_train_file,
)
```

Notice that the user simply has to pass the same log file to be used by **LotusTrace** using keywords `log_transform_elapsed_time` and `log_file`.

### Example for LotusMap

We provide 6 examples of how to use **LotusMap** in `code/image_classification/LotusMap` directory. Please check the code for more details.


## Limitations of Lotus

Similar to other tools in the past which do not claim to be perfect, we follow the same tradition with **Lotus**:

1. No current support for multi-node setting
2. No current support for DDP setting
3. **LotusMap** is approximate, checkout our paper for additional information

We claim issues 1 and 2 as a limitation as we simply have not tested the system in these settings yet.

## Acknowledgment

The lotus image is from "<a href="https://www.freepik.com/free-ai-image/flower-that-is-yellow-pink_40612656.htm#fromView=search&page=1&position=33&uuid=4a4a6af9-ac36-4555-8c15-7c4261fcdee6">Image by Sketchepedia on Freepik</a>"


## License
Click [here](LICENSE).

## Contact
Name: Rajveer Bachkaniwala

Email: rr [at] gatech [dot] edu

