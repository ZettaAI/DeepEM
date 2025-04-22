FROM mambaorg/micromamba AS intermediate
# Multi-stage build to pull in zettasets (private repo)

USER root
ARG GIT_ACCESS_TOKEN

RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y --no-install-recommends build-essential git && \
    git clone https://${GIT_ACCESS_TOKEN}@github.com/ZettaAI/zettasets.git /tmp/zettasets && \
    rm -rf /var/lib/apt/lists/*

# Use an optimized PyTorch base image
# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

COPY --from=intermediate /tmp/zettasets /tmp/zettasets

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git libboost-all-dev \
        curl apt-transport-https ca-certificates gnupg \
        ffmpeg libsm6 libxext6 && \
    # Install Google Cloud CLI
    echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies using Conda (now with faster solving)
RUN conda install -y h5py cython matplotlib scikit-image scikit-learn && \
    conda clean -a

# Explicitly pin NumPy to <2 to avoid ABI issues
RUN pip install --no-cache-dir --upgrade \
        "numpy<2" cloud-volume task-queue imgaug wandb \
        cffi brotli fastremap imagecodecs onnx

RUN pip install --no-cache-dir \
        git+https://github.com/seung-lab/DataTools && \
    pip install --no-cache-dir \
        git+https://github.com/ZettaAI/DataProvider3 && \
    pip install --no-cache-dir \
        git+https://github.com/ZettaAI/Augmentor && \
    pip install --no-cache-dir \
        git+https://github.com/seung-lab/pytorch-emvision && \
    pip install --no-cache-dir \
        git+https://github.com/ZettaAI/samwise && \
    pip install --no-cache-dir \
        git+https://github.com/ZettaAI/cirrus-volume.git@main

WORKDIR /tmp/zettasets
RUN pip install --no-cache-dir -e .

# Set up working directory and environment
RUN mkdir -p /workspace
ENV PYTHONPATH=/DeepEM:${PYTHONPATH}

COPY . /DeepEM/
WORKDIR /workspace

ENTRYPOINT ["python", "/DeepEM/deepem/train/run.py"]
