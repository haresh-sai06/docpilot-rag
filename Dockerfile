# --- 1. build the React UI ---
FROM node:20-slim AS ui
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- 2. API + UI runtime ---
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src \
    DOCPILOT_DATA_DIR=/app/data DOCPILOT_QDRANT_PATH=/app/data/qdrant FASTEMBED_CACHE_PATH=/app/models PORT=8080 \
    DOCPILOT_LLM_PROVIDER=openai
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY scripts/warmup.py scripts/
# bake corpus, embedding/reranker models and the vector index into the image -> fast cold starts
RUN python scripts/warmup.py
COPY --from=ui /ui/dist frontend/dist
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8080
CMD ["sh", "-c", "uvicorn docpilot.api:app --host 0.0.0.0 --port ${PORT}"]
