FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first
COPY requirements.txt .

# Install dependencies (ignore errors from DVC for now)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt || true

# Copy the full project
COPY . .

# Default environment
ENV PYTHONUNBUFFERED=1

# Expose port (for Streamlit or Flask)
EXPOSE 8501

# Run Streamlit app by default
CMD ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
