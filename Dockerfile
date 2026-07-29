FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model into image so runtime doesn't depend on HF network
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

# Create required directories
RUN mkdir -p uploads data chroma_store

EXPOSE 8000

CMD python scripts/seed_index.py && uvicorn main:app --host 0.0.0.0 --port $PORT
