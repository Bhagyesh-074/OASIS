---
name: oasis-context
description: Use whenever implementing, reviewing, extending, or reasoning about any part of the OASIS project — TCE, RBE, RTPM, Replacement Engine, Configuration Memory, adapters, the eval/ablation harness, the API, or the DB schema. Also use when writing tests, config YAML, or paper/report sections that describe system behaviour. Points to the authoritative docs in docs/ and lists invariants that must never be violated regardless of which module is being touched.
---

# OASIS project context

OASIS is a framework-agnostic supervisory layer over LangGraph/AutoGen. It does
NOT own the agent execution loop — it gains control at exactly three seams:
pre-execution config, the LiteLLM model-call gateway, and the output boundary.
Keeping those seams narrow is the core design constraint; do not add new
integration points without checking ARCHITECTURE.md first.

Research question this whole project exists to answer: do complexity-based
team sizing (TCE), runtime budget enforcement (RBE), and mid-run agent
replacement (RTPM+RE) compose additively, or interfere with each other under
one shared budget? Code changes that make an interference measurement
impossible to compute (e.g. silently merging RE and RBE into one decision, or
excluding supervision tokens from accounting) are wrong even if they "work."

## Where the authoritative detail lives — read before implementing

Don't guess at behaviour this project already specifies. Load the relevant
file(s) before writing code:

- `docs/PRD.md` — product behaviour, scope, the six hypotheses H1-H6
- `docs/ARCHITECTURE.md` — component map, the three seams, data flow, determinism strategy
- `docs/REQUIREMENTS.md` — FR-1..FR-30 and NFR-1..NFR-7, each with its owning component and its verifying test
- `docs/API_SPEC.md` — every endpoint, request/response shape, error envelope
- `docs/DATABASE.md` — SQLite schema (source of truth for all table/column names), derived-quantity formulas
- `docs/DECISIONS.md` — ADR-001..ADR-017, each a reversal from a v1.0 mistake with the reasoning; re-implementing a v1.0 approach without reading the matching ADR is a regression, not a fresh idea
- `docs/TESTING.md` — test layers, and which invariant test guards which claim
- `docs/DEVELOPMENT_PLAN.md` — phasing, ownership by vertical slice, critical path
- `docs/DEPLOYMENT.md` — env vars, cost controls, reproducibility artifact

When a task references an FR/NFR/ADR id (e.g. "implement FR-14"), open
REQUIREMENTS.md or DECISIONS.md for that id's exact text rather than inferring
it from the code alone.

## Ownership (don't implement outside your lane without flagging it)

| Area | Owner | Modules |
|---|---|---|
| Gateway, budget enforcement, framework adapters | Vedha | `gateway/`, `rbe/`, `adapters/` |
| Complexity estimation, benchmark tasks, oracle sweep | Netra | `tce/`, `bench/`, `eval/oracle_sweep.py` |
| Performance monitoring, replacement, memory, XAI, injection | Bhagyesh | `rtpm/`, `replacement/`, `cm/`, `xai/`, `eval/injection/` |
| API, runner, stats, DB, dashboard | Vidish | `api/`, `eval/runner.py`, `eval/stats.py`, `db/`, `frontend/` |

Adding a new framework adapter must require zero changes to `tce/`, `rbe/`,
`rtpm/`, `replacement/`, or `cm/` (NFR-7) — if a change touches one of those
to support an adapter, stop and reconsider the adapter contract instead.

## Non-negotiable invariants

These hold no matter which module is being edited. Each backs a specific
paper claim, and violating one silently corrupts results rather than causing
a visible bug:

- **No call bypasses budget enforcement.** Every outbound model call passes
  the RBE admission hook (FR-7). Gateway call count must equal the sum of
  budget events — never call a provider directly from a component.
- **Supervision cost is counted, not excluded.** L2 judge and RE calls are
  tagged `purpose=supervision` and count against the same budget as
  productive work (FR-9, ADR-006). Never carve these out of accounting to
  make numbers look better.
- **Score layers are never collapsed before storage.** L1/L2/L3 write to
  separate `score` rows keyed by `(output_id, layer)` — never overwrite one
  layer with another or store only a composite (FR-15).
- **Every automated decision gets a non-empty justification string** written
  to `decision_log` (FR-24) — sizing, reduction, escalation, flag, swap,
  denial, halt. A decision with no justification fails CI (see
  `tests/invariant/`).
- **Replacement must negotiate budget, never override it.** RE requests
  handoff budget from RBE; a refusal is logged as `replacement_denied_budget`,
  not silently retried or bypassed (FR-17, ADR-007). Denials are the primary
  interference measurement — don't suppress or hide them.
- **Model identifiers are pinned, never aliased** (NFR-5) — no `latest`,
  no bare model family name. Config YAML changes to `config/*.yaml` are
  illustrative placeholders per DEPLOYMENT.md until Phase 0 produces real
  numbers; don't treat the current values as calibrated.
- **Tests never depend on live model behaviour.** Unit/contract/integration
  tests run against `gateway/fake_llm.py`, keyed by prompt hash — no network,
  no API key, per TESTING.md's CI principle.
- **Replay mode issues zero network calls.** Anything that would break
  `mode=replay` (e.g. a component calling a provider directly instead of
  through the gateway) breaks the reproducibility artifact (FR-30).
- **Detection recall is reported per failure-type and severity, never as a
  single aggregate** (see TESTING.md "Detection-recall validity") — don't
  add code paths that only surface an aggregate number.

## When writing tests

Check TESTING.md for which layer a new test belongs in (unit / contract /
integration / invariant / performance) before adding it — invariant tests in
particular exist because they guard a paper claim, not because they check a
function's return value. A new component isn't "done" until its FR-mapped
tests pass, it participates in a green end-to-end run with FakeLLM, and its
decisions show up in the decision log (see TESTING.md "Definition of done").
