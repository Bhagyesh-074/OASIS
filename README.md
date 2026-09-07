# OASIS

Framework-agnostic supervisory layer over LangGraph / AutoGen: complexity-based
team sizing, runtime budget enforcement, mid-run agent replacement — studying
whether the three interfere under a shared budget. Group A-9.

## Setup (uv)

```bash
uv sync                                   # installs from uv.lock
uv run python -m spacy download en_core_web_sm
uv run python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2')"

cp .env.example .env                      # add your API key
uv run python -m oasis.db.migrate
uv run python -m oasis.bench.ingest --all
uv run uvicorn oasis.api.app:app --reload --port 8000

cd frontend && npm ci && npm run dev      # :5173
```

Smoke check before any real spend:
```bash
uv run python -m oasis.bench.smoke --tasks 5 --arm full --mode live --max-cost 1.00
```

## Dev

```bash
uv run ruff check .
uv run mypy src/oasis
uv run pytest tests/unit tests/contract tests/integration tests/invariant
```

## Docs

`docs/` holds the nine spec documents (PRD, ARCHITECTURE, REQUIREMENTS,
API_SPEC, DATABASE, DEVELOPMENT_PLAN, TESTING, DEPLOYMENT, DECISIONS) —
these are the source of truth, not this README.

## Antigravity

`.agents/skills/oasis-context/SKILL.md` is a project-scope Agent Skill.
Antigravity loads it automatically for anyone who clones the repo — it
orients the agent to the architecture, points it at the right doc for a
given task, and lists invariants (budget-bypass, score-collapsing,
justification-completeness, etc.) that must hold regardless of which
module is being edited.

## Ownership (vertical slices, see DEVELOPMENT_PLAN.md)

| Area | Owner |
|---|---|
| `gateway/`, `rbe/`, `adapters/` | Vedha |
| `tce/`, `bench/` (task ingestion + verifiers), `eval/oracle_sweep.py` | Netra |
| `rtpm/`, `replacement/`, `cm/`, `xai/`, `eval/injection/` | Bhagyesh |
| `api/`, `eval/runner.py`, `eval/stats.py`, `db/`, `frontend/` | Vidish |

Add-a-framework rule (NFR-7): a new adapter must require zero changes to
`tce/`, `rbe/`, `rtpm/`, `replacement/`, or `cm/`.
