# Use an official Python runtime as a parent image
# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04 as base
ENV WANDB_API_KEY=3e2af65a360f4257c91aad8a7affec7b85d70dd5

# Set environment name
ARG ENV_NAME=afford
ENV DEBIAN_FRONTEND noninteractive

# Set the working directory in the container
WORKDIR /affordance

# Copy the environment.yml file into the container at /app
COPY environment.yml .

# Install system dependencies if needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


# Install Python 3.8 and pip, and make python and pip aliases to python3.8 and pip3
RUN apt-get update && \
    apt-get install -y python3.8 python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

RUN pip install fvcore iopath
# RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu116_pyt1131/download.html

# Install dependencies using pip
RUN pip install --no-cache-dir -r environment.yml
RUN pip install --upgrade pip

COPY affordance .
# WORKDIR "/affordance"
# RUN pip install -e .
RUN pip install -e ./src

CMD [ "python", "new_scripts/preprocess_csv.py"]




