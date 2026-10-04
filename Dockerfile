# Stage 1: Build virtual environment and pre-download models (offline execution)
FROM python:3.11-slim AS builder
WORKDIR /app
ENV HF_HOME=/app/models/huggingface
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
RUN uv run python -m spacy download en_core_web_sm
RUN uv run python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2')"

# Stage 2: Runtime image
FROM python:3.11-slim
WORKDIR /app
ENV HF_HOME=/app/models/huggingface
ENV PATH="/app/.venv/bin:$PATH"
RUN pip install --no-cache-dir uv
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/models /app/models
COPY . .
EXPOSE 8000
CMD ["uvicorn", "oasis.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
