FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev
RUN uv run python -m spacy download en_core_web_sm
RUN uv run python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2')"

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app /app
COPY . .
CMD ["uv", "run", "uvicorn", "oasis.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
