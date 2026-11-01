FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends fonts-wqy-zenhei libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY deploy/requirements-cloud.txt /tmp/requirements-cloud.txt
RUN pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /tmp/requirements-cloud.txt
COPY src /app/src
RUN useradd --uid 10001 --create-home appuser && mkdir -p /app/results /app/weights \
    && chown -R appuser:appuser /app
USER appuser
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=3)"
CMD ["uvicorn", "web.server:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
