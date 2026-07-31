FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download fastembed ONNX model into image so runtime is instant and low-memory
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding(model_name='BAAI/bge-small-en-v1.5').embed(['test']))"

COPY . .

# Create required directories
RUN mkdir -p uploads data chroma_store

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
# CMD python scripts/seed_index.py && uvicorn main:app --host 0.0.0.0 --port $PORT