# Use slim Python base to save several GB
FROM python:3.10-slim AS builder

WORKDIR /app

# Pre-install build tools just for Prophet/CMDStanPy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make curl \
 && pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir cmdstanpy==1.2.0 prophet==1.1.5 \
 && apt-get clean && rm -rf /var/lib/apt/lists/* /root/.cache

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Final smaller runtime image
FROM python:3.10-slim

WORKDIR /app

# Copy only installed Python libs from builder
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY . .

# Ensure Streamlit is accessible
RUN pip install --no-cache-dir streamlit

EXPOSE 8501
CMD ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
