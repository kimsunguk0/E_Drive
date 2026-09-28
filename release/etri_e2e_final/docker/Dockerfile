# ETRI E2E Driving — final model (EXT-FULL-v7) runtime for inference and training.
# The release is mounted at /release; tools/assemble.py rebuilds the original
# repository layout (/NHNHOME/data/sukim/adcl) at start-up so source-hash pins hold.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.1.3 scipy==1.15.3 pillow==11.2.1 \
    opencv-python-headless==4.11.0.86 pyarrow==20.0.0 pandas==2.2.3 torchvision==0.22.1
RUN git config --global --add safe.directory '*'
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    EXT_K=5 EXT_LAT=3 EXT_SPREAD=0.06 EXT_LAT_SPREAD=0.04 EXT_ALL_W=0 EXT_EXT_W=0.3
WORKDIR /NHNHOME/data/sukim/adcl
