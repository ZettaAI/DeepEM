FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ARG GIT_ACCESS_TOKEN

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential git \
        #gcloud cli (for samwise)
        curl apt-transport-https ca-certificates gnupg \
    && rm -rf /var/lib/apt/lists/* \
    # gcloud cli (for samwise)
    && echo "deb https://packages.cloud.google.com/apt cloud-sdk main" \
        | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | apt-key add - \
    && apt-get update && apt-get install google-cloud-cli \
    # python requirements
    && conda install h5py cython matplotlib \
        scikit-image scikit-learn \
    && conda clean -t -p \
    # pypi packages
    && pip install --no-cache-dir --upgrade \
        cloud-volume task-queue tensorboardX \
    # github packages
    && pip install --no-cache-dir \
        git+https://github.com/seung-lab/DataProvider3 \
        git+https://github.com/ZettaAI/Augmentor \
        git+https://github.com/seung-lab/pytorch-emvision \
        git+https://github.com/ZettaAI/samwise@fix_setup \
        git+https://${GIT_ACCESS_TOKEN}@github.com/ZettaAI/zettasets

RUN mkdir -p /workspace
ENV PYTHONPATH /DeepEM:${PYTHONPATH}

COPY . /DeepEM/
WORKDIR /workspace
ENTRYPOINT ["python", "/DeepEM/deepem/train/run.py"]
