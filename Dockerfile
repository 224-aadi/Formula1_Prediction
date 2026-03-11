# Use a lightweight official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables to prevent Python from writing .pyc files
# and to ensure output is logged in real-time
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Install system dependencies required for LightGBM and FastF1
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements to cache them in docker layer
COPY pyproject.toml ./
COPY src/ ./src/

# Install pip dependencies (fastapi, uvicorn, lightgbm, fastf1, pandas)
RUN pip install --no-cache-dir .

# Copy the application artifacts

COPY data/processed/ ./data/processed/
COPY artifacts/final/ ./artifacts/final/

# Create raw data cache directories for FastF1/Jolpica if they don't exist
RUN mkdir -p data/raw/fastf1_cache data/raw/jolpica_cache

# Expose port (Cloud Run sets PORT env var automatically, default 8000)
ENV PORT=8000
EXPOSE $PORT

# Start Uvicorn pointing to the API app
CMD ["sh", "-c", "uvicorn f1outcome.api.app:app --host 0.0.0.0 --port $PORT"]
