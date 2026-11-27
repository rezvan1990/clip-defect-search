FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app \
    QC_DATA_DIR=/var/qc \
    QC_CACHE=/var/cache/features \
    HF_HOME=/var/cache/huggingface \
    PORT=7860

EXPOSE 8000 7860 8501

CMD ["uvicorn", "qcapi.main:app", "--host", "0.0.0.0", "--port", "8000"]
