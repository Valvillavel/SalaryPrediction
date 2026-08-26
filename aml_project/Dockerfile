# Build stage: install dependencies
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY requirements-docker.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements-docker.txt

# Runtime stage: minimal production image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and trained models
COPY src ./src
COPY models ./models

# Create non-root user for security
RUN useradd --create-home --no-log-init appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Health check with Python requests
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
