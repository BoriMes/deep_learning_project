# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- system deps (pandas/matplotlib) ---
RUN echo ">>> [1/6] Installing system deps" && \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
    && rm -rf /var/lib/apt/lists/*

# --- python deps ---
COPY requirements.txt /app/requirements.txt

RUN echo ">>> [2/6] Installing python deps (requirements.txt)" && \
    pip install -v --no-cache-dir -r /app/requirements.txt

# CPU-only PyTorch from official CPU wheel index
RUN echo ">>> [3/6] Installing PyTorch (CPU wheels)" && \
    pip install -v --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0

# --- copy sources and entrypoint ---
RUN echo ">>> [4/6] Copying source code"
COPY src/ /app/src/
COPY run.sh /app/run.sh

# Fix CRLF from Windows, make executable
RUN echo ">>> [5/6] Preparing run.sh" && \
    sed -i 's/\r$//' /app/run.sh && \
    chmod +x /app/run.sh

# expected mount points
RUN echo ">>> [6/6] Creating mount points" && \
    mkdir -p /app/data /app/output

ENTRYPOINT ["/app/run.sh"]
