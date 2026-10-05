# Stage 1: Build virtual environment and pre-download models (offline execution)
FROM python:3.11-slim AS builder
WORKDIR /app
ENV HF_HOME=/app/models/huggingface
RUN pip install --no-cache-dir "uv>=0.11.7,<0.12.0"
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
RUN uv run python -m spacy download en_core_web_sm
RUN uv run python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2')"

# Stage 2: Runtime image
FROM python:3.11-slim
WORKDIR /app
ENV HF_HOME=/app/models/huggingface
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"
RUN pip install --no-cache-dir "uv>=0.11.7,<0.12.0"
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/models /app/models
COPY . .
RUN uv sync --frozen --no-dev
EXPOSE 8000
CMD ["uvicorn", "oasis.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

