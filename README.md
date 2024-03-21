<p align="center">
    <img src="P3Tracer.png" width="40%" alt="P3Tracer">
</p>

<p align="center"><i><b>Fine-grained Profiling for Machine Learning Data
Preprocessing in PyTorch</b></i></p>


We introduce **P3Tracer**, a specialized profiling tool for PyTorch preprocessing pipelines. 

**P3Tracer** is a easy-to-use, low overhead, visualization-ready and  profiler specialized for the widely used PyTorch framework.

## Quick links
- [About P3Tracer](#about-p3tracer)
- [Get P3Tracer](#how-to-get-p3tracer)
- [Use P3Tracer](#how-to-use-p3tracer)
    - [How to use P3Torch](#how-to-use-p3torch)
    - [How to use P3Map](#how-to-use-p3map)
- [Concrete examples](#concrete-examples)
    - [Example for P3Torch](#example-for-p3torch)
    - [Example for P3Map](#example-for-p3map)
- [Cite P3Tracer](#citation)
- [License](#license)



## About P3Tracer

P3Tracer employs two novel approaches:

1. **P3Torch** - An instrumentation methodology for the PyTorch library, which enables fine-grained elapsed time profiling with minimal time and storage overheads. 
2. **P3Map** - A mapping methodology to reconstruct a mapping between Python functions and the underlying C++ functions they call, effectively linking high-level Python functions with low-level hardware counters. 

Above combination is powerful as it allows enables users to better reason about their pipeline’s performance, both at the level of preprocessing operations and their performance on hardware resource usage.

## How to get P3Tracer

1. Clone this repository
2. Get submodules:

    ```git
    git submodule update --init --recursive
    ```
3. Create a conda environment

    ```bash
    conda create -n P3Tracer python=3.10
    conda activate P3Tracer
    ```
4. Install Intel VTune from here.

    Note: we used `Intel(R) VTune(TM) Profiler 2024.0.1 (build 627177)`
5. Install CUDA 11.8 from [here](https://developer.nvidia.com/cuda-11-8-0-download-archive) and CuDNN 8.7.0 from [here](https://developer.nvidia.com/rdp/cudnn-archive)
6. Follow the **P3Torch** build instructions in `code/P3Torch/README.md`
7. Follow the **itt-python** build instructions in `code/itt-python/README.md`
8. That's it!

## How to use P3Tracer

### How to use P3Torch


**P3Torch** can be enabled by simply passing a `custom_log_file` to be used by **P3Torch** using keywords `log_transform_elapsed_time` and `log_file` as shown below:

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

We do support **P3Torch** for custom datasets as well check below instance:


```python
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

### How to use P3Map

Below is an example of how to write a program such that using **P3Map** will enable collection of mapping: 

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

## Concrete examples

### Example for P3Torch

An example of how to enable **P3Torch** facilitated logging for an image classification task has been described in `code/image_classification/code/pytorch_main.py`, we add the snippet below for the same:

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

Notice that the user simply has to pass the same log file to be used by **P3Torch** using keywords `log_transform_elapsed_time` and `log_file`.

### Example for P3Map

We provide 6 examples of how to use **P3Map** in `code/image_classification/P3Map` directory. Please check the code for more details.

## Cite P3Tracer

TODO

## License
Copyright (c) [Georgia Tech Database Group](http://db.cc.gatech.edu/).
Licensed under an [Apache License](LICENSE.txt).