# Use NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set up basic utilities
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -all && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH

# Install additional dependencies for video processing
# RUN apt-get update && apt-get install -y \
#     ffmpeg \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid 1000 -m vscode
USER vscode
WORKDIR /home/vscode

# Accept ToS
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment
RUN conda create -n uninavid python=3.10 -y
COPY pyproject.toml /home/vscode/pyproject.toml
RUN conda init && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate uninavid && \
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && \
    pip install --upgrade pip && \
    cd /home/vscode && \
    pip install .
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate uninavid && \
    pip install flash-attn==2.5.9.post1 --no-build-isolation

# Initialize conda for the user
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
echo "conda activate uninavid" >> ~/.bashrc
