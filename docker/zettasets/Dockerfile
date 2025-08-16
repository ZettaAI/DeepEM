FROM mambaorg/micromamba AS intermediate
# multi-stage build to pull in zettasets (private repo)

USER root
ARG GIT_ACCESS_TOKEN
RUN apt-get --allow-releaseinfo-change update && \
    apt-get install -y build-essential git && \
    git clone https://${GIT_ACCESS_TOKEN}@github.com/ZettaAI/zettasets.git && \
    rm -rf /var/lib/apt/lists/*

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

COPY --from=intermediate /tmp/zettasets /tmp/zettasets


RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential git \
        # gcloud cli (for samwise)
        curl apt-transport-https ca-certificates gnupg \
        # boost libraries for datatools compilation
        libboost-all-dev \
    && rm -rf /var/lib/apt/lists/* \
    # gcloud cli (for samwise)
    && echo "deb https://packages.cloud.google.com/apt cloud-sdk main" \
        | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | apt-key add - \
    && apt-get update \
    && apt-get install google-cloud-cli -y \
    # handle imgaug issue
    && apt-get install ffmpeg libsm6 libxext6 -y \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    # python requirements
    && conda install h5py cython matplotlib \
        scikit-image scikit-learn \
    && conda clean -t -p \
    # pypi packages
    && pip install git+https://github.com/ZettaAI/cirrus-volume \
    && pip install --no-cache-dir --upgrade \
        numpy cloud-volume task-queue tensorboardX imgaug wandb simplejpeg==1.7.6 imagecodecs \
    
    # github packages
    && pip install --no-cache-dir \
        git+https://github.com/seung-lab/DataTools \
        git+https://github.com/ZettaAI/DataProvider3 \
        git+https://github.com/ZettaAI/Augmentor \
        git+https://github.com/seung-lab/pytorch-emvision \
        git+https://github.com/ZettaAI/samwise

RUN cd /tmp/zettasets && \
    pip install -e .

RUN mkdir -p /workspace
ENV PYTHONPATH=/DeepEM:${PYTHONPATH}

COPY . /DeepEM/
WORKDIR /workspace
ENTRYPOINT ["python", "-u", "/DeepEM/deepem/train/run.py"]
