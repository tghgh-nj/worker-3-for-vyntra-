FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates fontconfig fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY worker.py ai_runtime.py espcn.py ai_image_models.py temporal_models.py nanovsr_model.py mobile_rrn.py NANOVSR_LICENSE.txt requirements-ai-image.txt requirements-vrt.txt requirements-temporal.txt /app/
RUN useradd --create-home --uid 10001 worker && chown -R worker:worker /app
USER worker
CMD ["python", "-u", "/app/worker.py"]
