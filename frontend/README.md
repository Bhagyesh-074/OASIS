React dashboard (Vidish). SSE client against /v1/runs/{id}/events.
Scaffold last on purpose — demonstration artifact, not a research contribution (ADR-016).

The dashboard uses the API's development key (`dev-local-key`) by default. To use a
different local API key, set `VITE_OASIS_API_KEY` in the frontend build environment.
Run IDs and benchmark summaries are loaded from the API; no demo results are substituted
when the API is unavailable.
