# Single image, two possible entrypoints (API or UI) selected via CMD
# override in docker-compose.yml - keeps the Dockerfile itself simple
# since both processes share the same dependencies.
FROM python:3.11-slim

WORKDIR /app

# PyMuPDF/chromadb/sentence-transformers wheels are prebuilt for this
# platform, so no extra system build toolchain is required beyond curl
# (used by the healthcheck).
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY ui ./ui
COPY scripts ./scripts
COPY data ./data

RUN mkdir -p /app/chroma_data

EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
