OASIS — Deployment and Reproducibility
OASIS has two deployment targets with different requirements: a local development and demonstration setup, and a reproducible artifact package for the paper. There is no production deployment, and the system is not hardened for one.

Reference environment
Any modern x86-64 machine with four cores, 16 GB RAM and 10 GB free disk. No GPU: spaCy en_core_web_sm and all-MiniLM-L6-v2 run comfortably on CPU, and all agent reasoning is API-based. Python 3.10 or 3.11, Node 20 for the dashboard build, Docker 24 with Compose v2.

The absence of a GPU requirement is not an accommodation, it is a design consequence — and it is why GPU and RAM were removed as budget dimensions. A layer that cannot observe GPU allocation cannot honestly claim to enforce it.

Configuration
All configuration is environment variables plus version-controlled YAML. Secrets never enter YAML or the repository.

```
# Providers — pinned versions only, aliases prohibited (NFR-5)
OASIS_LLM_PROVIDER=openai
OPENAI_API_KEY=<secret>
OASIS_MODEL_DEFAULT=gpt-4o-mini-2024-07-18
OASIS_MODEL_CHEAP=gpt-4o-mini-2024-07-18
OASIS_MODEL_STRONG=gpt-4o-2024-11-20
OASIS_MODEL_JUDGE=gpt-4o-mini-2024-07-18

# Global cost guardrail (NFR-3)
OASIS_SPEND_CEILING_USD=250
OASIS_SPEND_KILL_MULTIPLIER=1.2
OASIS_PER_RUN_MAX_COST_USD=2.00

# Execution
OASIS_MODE=live                  # live | replay
OASIS_CONCURRENCY=8
OASIS_CACHE_ENABLED=true
OASIS_CACHE_DIR=./artifacts/llm_calls
OASIS_DB_PATH=./data/oasis.db
OASIS_API_KEY=<dev header key>
```

config/complexity.yaml holds TCE weights and thresholds, config/budget_profiles.yaml the tight/moderate/generous profiles, config/templates.yaml the Agent Template Library, config/rubric_r3.yaml the judge rubric. Every file's SHA is recorded in each run's manifest, so a results file always identifies the exact configuration that produced it.

Local setup
```
git clone <repo> && cd oasis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock
python -m spacy download en_core_web_sm
python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2')"

cp .env.example .env            # add key
python -m oasis.db migrate
python -m oasis.bench ingest --all
uvicorn oasis.api:app --reload  # :8000
cd frontend && npm ci && npm run dev   # :5173
```

Smoke check before any real spend: `python -m oasis.bench smoke --tasks 5 --arm full --mode live --max-cost 1.00`.

Docker
docker compose up brings up two services — api (FastAPI plus all components, models baked into the image so first start is fast) and dashboard (nginx serving the built React bundle). Volumes mount ./data for SQLite and ./artifacts for the JSONL cache and results. There is no Redis service and no database service; both were removed with the single-process decision. The image builds in two stages, with the model download in the builder layer so it is cached.

Cost controls
Three independent guards, because a runaway agent loop is the documented failure mode this project exists partly to address and it would be embarrassing to reproduce it.

A per-call admission check via the RBE pre-call hook enforces the run's declared budget. A per-run hard ceiling (OASIS_PER_RUN_MAX_COST_USD) terminates a run regardless of its declared budget, catching bugs where the budget itself is wrong. A global cumulative spend counter persisted in SQLite blocks all run-creating endpoints once the ceiling is reached and hard-exits the process at the kill multiplier. Every tranche of the main experiment reports spend before the next tranche launches, and the Friday note to the guide includes cumulative spend.

Cost reduction comes mainly from the prompt-hash cache, which makes re-analysis free, plus using the cheap pinned model for judges and for all baseline arms where the comparison is structural rather than about model strength.

Reproducibility artifact
The artifact is the deliverable that makes the paper credible, and it is built to be runnable by a reviewer with no API key.

It contains the source at a tagged commit, requirements.lock, the Dockerfile, all config YAMLs, the benchmark task definitions with the authored domains included in full and the anchored domains referenced by dataset ID, the gzipped JSONL replay cache with keys scrubbed, the SQLite database snapshot, the results tables and figure-generating scripts, the human-labelling data for judge validation, and a manifest listing model pins, seeds, library versions, config hashes and commit.

Reproduction is one command: `docker compose run api python -m oasis.bench reproduce --job ablation-v3 --mode replay`, which reruns the full pipeline from cache and regenerates every table and figure in the paper with zero network calls. The replay-purity CI test (FR-30) guards this claim continuously rather than at submission time.

Live reproduction is documented separately with an explicit warning that model drift means numbers will differ, and with the expected spend stated. Being clear that live and replay reproduction are different things — and that only replay is exact — is more honest than the usual claim that an LLM experiment is reproducible.

Operational notes and known limits
SQLite in WAL mode with a single writer task handles concurrency of eight; raising concurrency much higher requires moving writes behind a queue or switching to Postgres, which is out of scope. Nightly .backup snapshots go to artifacts/db_snapshots/ and the JSONL cache is the true source of record for model traffic, so losing the database costs analysis time but not data.

The static API key is development-grade and the service must not be exposed publicly. Raw logs contain full prompts and responses and are scrubbed only for keys, not for task content, so any authored task containing sensitive material must not be used. Provider deprecation of a pinned model is the most likely mid-project break; the cache makes existing results survive it, and the nightly live smoke job detects it within a day.
