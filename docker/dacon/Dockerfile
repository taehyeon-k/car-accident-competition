FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# -------------------------------------------------
# System packages corresponding to DACON environment
# -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    ca-certificates \
    build-essential \
    ffmpeg \
    sudo \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libffi-dev \
    libblas3 \
    liblapack3 \
    libomp-dev \
    libatlas-base-dev \
    gfortran \
    cmake \
    pkg-config \
    ninja-build \
    tzdata \
    zip \
    unzip \
    p7zip-full \
    file \
    procps \
    time \
    jq \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

# -------------------------------------------------
# Python tooling
# -------------------------------------------------
RUN python -m pip install --upgrade \
    pip \
    setuptools \
    wheel

# -------------------------------------------------
# DACON package environment
# -------------------------------------------------
COPY requirements-dacon.txt /tmp/requirements-dacon.txt

RUN python -m pip install \
    -r /tmp/requirements-dacon.txt

# -------------------------------------------------
# Intentional deviations from DACON:
#
# DACON: transformers 4.57.6
# Ours:  transformers 5.16.1
#
# RF-DETR 1.10.1 requires Transformers 5.x
# -------------------------------------------------
RUN python -m pip install \
    "transformers==5.16.1" \
    "rfdetr==1.10.1"

# Catch dependency conflicts at build time
RUN python -m pip check

WORKDIR /workspace/car-accident

CMD ["/bin/bash"]
