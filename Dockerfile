# Use a lightweight python slim base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies:
# - build-essential: needed to compile native extensions (websockets, greenlet)
# - libgomp1: OpenMP runtime required by LightGBM for multi-threaded inference
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source (exclude training data, notebooks, scratch files)
COPY agents/ agents/
COPY config/ config/
COPY core/ core/
COPY models/ models/
COPY utils/ utils/
COPY main.py .

# Ensure models and logs directories exist at runtime
RUN mkdir -p /app/data/models /app/logs

# Stream logs in real-time (no Python output buffering)
ENV PYTHONUNBUFFERED=1

# Run the live trading bot
CMD ["python", "main.py"]
