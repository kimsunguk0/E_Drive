# A2 student only. Build validation is recorded separately from native B200 runs.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.15.3 \
    pillow==11.2.1 opencv-python-headless==4.11.0.86 pyarrow==20.0.0 torchvision==0.22.1
WORKDIR /workspace/code
ENV ADCL_CODE_ROOT=/workspace/code
ENTRYPOINT ["python", "experiments/a2_visual_teacher_20260919/infer_a2.py"]
