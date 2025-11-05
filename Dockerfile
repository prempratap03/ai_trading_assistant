# ---- Stage 1: Builder ----
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies needed for Prophet and other libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpython3-dev \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirement files first (to leverage caching)
COPY requirements.txt .

# Install only core dependencies with no cache
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir cmdstanpy==1.2.0 prophet==1.1.5 \
 && pip install --no-cache-dir -r requirements.txt \
 && rm -rf /root/.cache /tmp/*

# ---- Stage 2: Runtime ----
FROM python:3.10-slim

WORKDIR /app

# Copy installed Python packages from builder image
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10

# Copy your project code
COPY . .

# Streamlit uses port 8501
EXPOSE 8501

# Optional: clean environment vars
ENV PYTHONUNBUFFERED=1

# Default command to run your Streamlit app
CMD ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]