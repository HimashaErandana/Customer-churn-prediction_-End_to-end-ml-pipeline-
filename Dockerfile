FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api

# Copy only the approved deployment model artifact. .env is intentionally
# NOT copied — configuration is injected via environment variables at runtime.
COPY deployment/model ./deployment/model

EXPOSE 8000

CMD ["python", "api/main.py"]
