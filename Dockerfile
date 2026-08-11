# Bank Document Classification & Information Extraction API
FROM python:3.11-slim

# System deps: tesseract for OCR, libgl for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Expect outputs/models/*.pt to be mounted or baked in before running
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
