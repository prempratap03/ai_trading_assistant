# =============================
# STAGE 1: Builder
# =============================
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Install build tools only for this stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for caching
COPY requirements.txt .

# Install only necessary dependencies in build layer
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# =============================
# STAGE 2: Final lightweight image
# =============================
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder (smaller final image)
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Environment configs
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false
ENV RAPIDAPI_KEY=""

# Run Streamlit app (make sure path is correct)
CMD ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
