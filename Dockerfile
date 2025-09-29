FROM python:3.12-slim

WORKDIR /app

# Only install minimal build tools (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Start the FastAPI app with 1 worker to reduce memory use
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
