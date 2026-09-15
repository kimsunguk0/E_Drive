# RTX 4090 measurement image for the MotionDrive V2 candidate.
# Versions are pinned to the environment the B200 numbers were produced with, so
# the only intended difference between the two measurements is the device.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3-pip libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --no-cache-dir --upgrade pip

RUN python3.10 -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.7.1 torchvision==0.22.1

RUN python3.10 -m pip install --no-cache-dir \
      numpy==1.26.4 \
      opencv-python==4.8.1.78 \
      pillow==12.2.0 \
      pyarrow==24.0.0 \
      scipy==1.15.3

WORKDIR /work
ENV PYTHONPATH=/work/code:/work/code/scripts
