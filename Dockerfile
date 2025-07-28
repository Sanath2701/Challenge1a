FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        fonts-dejavu \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Uncomment the next lines if you need French or German OCR support:
# RUN apt-get update && apt-get install -y tesseract-ocr-fra tesseract-ocr-deu

COPY Challenge1a.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Make sure input/output folders exist by default:
RUN mkdir -p /app/Input /app/Output

CMD ["python", "Challenge1a.py"]
