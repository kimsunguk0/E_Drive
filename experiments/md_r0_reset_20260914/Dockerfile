# Draft image for reproducing the MotionDrive V2 candidate.
# NOT built or tested here; it records the environment the run actually used.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.10 python3.10-venv python3-pip \
      libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# torch 2.7.1+cu128 is the build the reported numbers were produced with.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.7.1 torchvision
COPY environment_pip_freeze.txt /tmp/environment_requirements.txt
RUN pip install --no-cache-dir -r /tmp/environment_requirements.txt

WORKDIR /workspace
# Mount the repository, the checkpoint and the official test tars, then:
#   python experiments/md_r0_reset_20260914/submit_official_test.py \
#     --init <checkpoint> --test-root <test tars> --out submission.json \
#     --label review --gpu 0
