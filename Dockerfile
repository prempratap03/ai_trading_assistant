# Use slim Python base to save several GB
FROM python:3.10-slim AS builder
WORKDIR /app

# Pre-install build tools just for Prophet/CMDStanPy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make curl \
 && pip install --no-cache-dir --upgrade pip setuptools wheel \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and remove prophet if it's there (we'll install it separately)
COPY requirements.txt .
RUN grep -v "prophet\|cmdstanpy" requirements.txt > requirements_filtered.txt || cp requirements.txt requirements_filtered.txt

# Install Prophet first (heavy dependency)
RUN pip install --no-cache-dir cmdstanpy==1.2.0 prophet==1.1.5

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements_filtered.txt

# Clean up build artifacts
RUN apt-get purge -y gcc g++ make curl \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

# Final smaller runtime image
FROM python:3.10-slim
WORKDIR /app

# Copy only installed Python libs from builder
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]