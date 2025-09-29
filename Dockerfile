# Stage 1: Build
FROM python:3.12-slim AS builder

WORKDIR /install

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev git \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip freeze > requirements.lock

# Stage 2: Final image
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
