# FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
FROM ultralytics/ultralytics:latest

RUN apt-get update && apt-get install -y \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
