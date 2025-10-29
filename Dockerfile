FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt \
    && pip install "dvc"

# Copy the rest of the code
COPY . .

# Environment variables
ENV RAPIDAPI_KEY=${RAPIDAPI_KEY}
ENV PYTHONUNBUFFERED=1

# Expose API port (if using Flask)
EXPOSE 8080

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Pull DVC artifacts at startup, then run Streamlit
ENTRYPOINT ["/bin/sh","-c","dvc pull || echo 'No DVC artifacts pulled'; exec streamlit run src/main.py --server.port=8501 --server.address=0.0.0.0"]
