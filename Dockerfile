FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Speed up and harden pip installs
ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_NO_CACHE_DIR=1

# Upgrade pip and install Python deps + DVC S3
RUN python -m pip install --upgrade pip \
 && pip install --retries 10 --timeout 120 -r requirements.txt \
 && pip install --retries 10 --timeout 120 "dvc[s3]"

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Pull DVC artifacts at startup, then run Streamlit
ENTRYPOINT ["/bin/sh","-c","dvc pull || echo 'No DVC artifacts pulled'; exec streamlit run src/main.py --server.port=8501 --server.address=0.0.0.0"]
