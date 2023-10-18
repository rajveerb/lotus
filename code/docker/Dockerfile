# Use an official PyTorch base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Switch to utils dir of PyTorch
# This path may be specific to the PyTorch Image Version
WORKDIR /opt/conda/lib/python3.10/site-packages/torch/utils/data/_utils/

# Copy Custom Dataloader script
COPY ./customDataloaderWorker.py .
