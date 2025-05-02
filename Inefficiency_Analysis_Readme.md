# Lotus GPU Idle Time Analysis

A research codebase for analyzing GPU idle time during deep learning workloads, with custom worker assignment and memory pinning logic. This repository contains scripts, modified PyTorch dataloader logic, and analysis tools for reproducibility and extensibility.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Setup Instructions](#setup-instructions)
- [Running Experiments](#running-experiments)
- [Code Structure](#code-structure)
- [Analysis Workflow](#analysis-workflow)
- [License](#license)

---

## Getting Started

Clone the repository and check out the relevant branch:

```bash
git clone https://github.com/rajveerb/lotus.git
cd lotus
git checkout origin/41-gpu-idle-time-analysis
git submodule sync
git submodule update --init --recursive --depth 1
```


---

## Setup Instructions

- **CloudLab Users:**
Follow the instructions in [`setup.MD`](SETUP.MD) for environment setup.
- **Other Environments:**
Manually install CUDA and cuDNN as described in [`setup.MD`](SETUP.MD).

---

## Running Experiments

1. **Preparation:**
Complete steps 4, 5, and 6 in [`replicate.MD`](REPLICATE.MD).
2. **Run Experiment:**
Use the provided script to launch experiments with varying configurations:

```bash
bash scripts/cloudlab/LotusTrace_imagenet.sh
```

    - Check the script for configuration options being run.

---

## Code Structure

| Path | Description |
| :-- | :-- |
| `code/LotusTrace/torch/utils/data/dataloader.py` | Custom worker assignment logic (see also `_utils/worker.py`) |
| `code/LotusTrace/torch/utils/data/_utils/worker.py` | Additional worker assignment logic |
| `code/LotusTrace/torch/utils/data/_utils/pin_memory.py` | Modified logic for memory pinning |
| `code/LotusTrace/torch/utils/data/_utils/pin_memory_ideal.py` | Ideal pinning: provides batches to the GPU out of order |


---

## Analysis Workflow

1. **Log Parsing:**
Pass the path to your logs folder to the appropriate directory variable in the analysis Python notebooks.
2. **Data Extraction:**
Use the `Get_everything` function to parse all logs and generate a consolidated dataframe.
3. **Analysis:**
Perform your analysis using the generated dataframe as needed.

---

## License

Distributed under the repository's license. See [`LICENSE`](LICENSE) for more information.

---

For further details, refer to the in-repo documentation and scripts.
