# Dockerfile - A robust and efficient blueprint for the GR00T environment.

# 1. Start from an official NVIDIA CUDA **devel** image.
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# 2. Set environment variables.
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# 3. Install necessary system-level packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# 4. Copy ONLY the requirements file.
COPY requirements.txt .

# 5. Install the most critical, heavy, and build-time-dependent libraries first.
RUN pip3 install --no-cache-dir \
    packaging \
    wheel \
    setuptools \
    numpy==1.26.4 \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1

# 6. Now, install the rest of the packages from the requirements file.
RUN pip3 install --no-cache-dir -r requirements.txt

# 7. Copy the entire project source code into the container.
COPY . .

# 8. Install the gr00t project itself.
RUN pip3 install .

# 9. Set the default command. The cluster batch job will override this.
CMD ["/bin/bash"]
