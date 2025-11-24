FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# 1. Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace

# 2. Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Alias python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# 4. Install core python build tools AND flash-attn build dependencies
# Added: packaging, psutil, ninja (Required for building flash-attn without isolation)
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel packaging psutil ninja

# 5. Copy requirements first to cache pip installs
COPY requirements.txt .

# 6. Install PyTorch stack FIRST
RUN pip3 install --no-cache-dir \
    numpy==1.26.4 \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1

# 7. Install flash-attn with --no-build-isolation
# Now that psutil/ninja/torch are present, this should succeed
RUN pip3 install --no-cache-dir --no-build-isolation flash-attn==2.7.1.post4

# 8. Install remaining requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# 9. Copy source and install gr00t
COPY . .
RUN pip3 install -e .

# 10. Permissions fix
RUN chmod -R 777 /workspace
