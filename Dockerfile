FROM python:3.12-slim

# Set workdir early
WORKDIR /app

# Install system dependencies (split into smaller layers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    portaudio19-dev \
    espeak \
    espeak-ng-data \
    libespeak-ng1 \
    libespeak1 \
    ffmpeg \
    git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Pre-copy requirements to leverage caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy only app code (after deps to save cache)
COPY . .

# Set environment and port
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
