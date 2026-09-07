OASIS — API Specification
Base URL http://localhost:8000 · Version prefix /v1 · Auth X-API-Key static header, development only · Content type application/json

Conventions
All identifiers are ULIDs. All timestamps are RFC 3339 UTC. Errors follow a single envelope:

```
{ "error": { "code": "budget_infeasible", "message": "No team of size >=1 fits max_cost_usd=0.001", "details": {"binding": "cost"} } }
```

Codes in use: validation_error, budget_infeasible, unknown_domain, adapter_unsupported, run_not_found, cache_miss_in_replay, global_spend_limit.

Shared objects
```
// Budget — exactly four dimensions
{ "max_tokens": 200000, "max_cost_usd": 0.75, "max_wall_seconds": 300, "max_calls": 60 }

// TeamConfig
{
  "config_id": "01J...",
  "team_size": 3,
  "origin": "estimator",              // estimator | memory | baseline | oracle
  "topology": "hub_and_spoke",        // linear | hub_and_spoke | parallel
  "roles": [
    { "role": "researcher", "template_id": "accurate_slow",
      "model": "gpt-4o-mini-2024-07-18", "temperature": 0.2,
      "est_tokens": 42000, "est_cost_usd": 0.11 }
  ],
  "justification": "MVTS=3 from subtasks=4, skill_clusters=3, dep_density=0.42"
}
```

Estimation and planning
POST /v1/estimate — complexity only, no side effects.

```
// request
{ "statement": "Summarise three papers on X and produce a cited comparison table.",
  "domain": "research_qa", "estimator": "heuristic" }
// 200
{ "mvts": 3,
  "subscores": { "subtask_count": 4, "skill_clusters": 3, "dep_density": 0.42 },
  "weights": { "subtask_count": 0.4, "skill_clusters": 0.4, "dep_density": 0.2 },
  "subtasks": ["retrieve papers", "extract claims", "compare", "format table"],
  "justification": "…", "latency_ms": 63 }
```

POST /v1/plan — memory lookup, estimation and pre-execution enforcement; returns the configuration that would be spawned.

```
// request
{ "statement": "…", "domain": "code_generation", "budget": { … },
  "use_memory": true, "framework": "langgraph", "pareto": false }
// 200
{ "config": { …TeamConfig… },
  "memory_hit": { "mem_id": "01J…", "similarity": 0.91, "prior_quality": 0.87 },
  "enforcement": {
    "feasible": true,
    "reductions": [ { "from_size": 4, "to_size": 3, "binding": "cost",
                      "merged": ["reviewer","formatter"],
                      "justification": "projected 0.91 USD > budget 0.75 USD" } ],
    "projection": { "tokens": 138000, "cost_usd": 0.62, "wall_seconds": 210, "calls": 41 }
  } }
```

With "pareto": true the response adds budget_frontier, an array of three points each holding a budget, a projected configuration and an expected-quality estimate. This endpoint is the C-priority FR-10 and may be absent.

Runs
POST /v1/runs — start a run. Returns immediately with 202.

```
// request
{ "task_id": "mbpp_0421",            // or inline "statement" + "domain"
  "budget": { … },
  "arm": "full",                      // see ablation arms below
  "framework": "langgraph",
  "seed": 1,
  "mode": "live",                     // live | replay
  "injection": { "enabled": true, "rate": 0.15, "types": ["numeric_perturb"] } }
// 202
{ "run_id": "01J…", "status": "queued", "events_url": "/v1/runs/01J…/events" }
```

Valid arm values are the eight ablation configurations: vanilla_fixed, single_agent, captainagent, tce_only, rbe_only, monitor_only, tce_rbe, full.

GET /v1/runs/{run_id}

```
{ "run_id": "01J…", "status": "completed",
  "status_detail": null,               // "halted_budget" | "adapter_error" | …
  "arm": "full", "seed": 1, "framework": "langgraph",
  "config": { …TeamConfig… },
  "totals": { "tokens": 141230, "cost_usd": 0.58, "wall_seconds": 187, "calls": 39,
              "supervision_tokens": 16880, "supervision_share": 0.12 },
  "compliance": { "tokens": true, "cost": true, "wall": true, "calls": true, "all": true },
  "quality": { "composite": 0.86, "l3_verifier": 0.90, "l2_judge": 0.84,
               "judge_model": "gpt-4o-mini-2024-07-18", "rubric_version": "r3" },
  "replacements": 1, "replacements_denied": 2,
  "final_output_ref": "artifacts/01J….md" }
```

GET /v1/runs/{run_id}/events — text/event-stream. Event types: run_started, config_selected, agent_output, score_computed, budget_event, agent_flagged, replacement, replacement_denied, run_completed.

```
event: score_computed
data: {"run_id":"01J…","agent_id":"01J…","step":7,"l1":0.41,"l2":0.55,
       "l3":null,"composite":0.48,"window_mean":0.52,"flagged":false}

event: replacement_denied
data: {"run_id":"01J…","agent_id":"01J…","reason":"budget",
       "handoff_tokens_required":8200,"tokens_remaining":3100,
       "justification":"handoff would exceed token budget by 5100"}
```

GET /v1/runs/{run_id}/decisions — the full XAI trail.

```
{ "decisions": [
  { "decision_id":"01J…","ts":"2026-10-04T09:12:03Z","component":"RBE",
    "decision":"reduce_team","inputs":{"from":4,"to":3,"binding":"cost"},
    "justification":"projected cost 0.91 USD exceeded declared 0.75 USD; merged reviewer+formatter (cosine 0.79)" },
  { "decision_id":"01J…","component":"RE","decision":"replace_agent",
    "inputs":{"dimension":"factuality","value":0.44,"threshold":0.60,
              "window":3,"template":"accurate_slow"},
    "justification":"factuality below threshold for 3 consecutive outputs; substituted accurate_slow" }
] }
```

POST /v1/runs/{run_id}/cancel — 200 with final status.

Memory and templates
GET /v1/memory/search?statement=…&k=3&exclude_run=01J… returns neighbours with similarity, prior_quality, prior_cost_usd and the stored config. DELETE /v1/memory resets the store, used between ablation arms. GET /v1/templates lists the Agent Template Library: fast_cheap, accurate_slow, and the per-domain specialists, each with model, temperature, system-prompt hash and mean historical cost.

Benchmark orchestration
POST /v1/benchmark/jobs launches a matrix job.

```
{ "name": "ablation-v3",
  "task_split": "eval",
  "arms": ["vanilla_fixed","tce_only","rbe_only","monitor_only","tce_rbe","full"],
  "seeds": [1,2,3],
  "budget_profile": "moderate",
  "concurrency": 8,
  "mode": "live" }
```

GET /v1/benchmark/jobs/{job_id} reports progress, spend to date and failures. GET /v1/metrics/summary?job_id=… returns the aggregate table: per-arm median quality, agent count, cost, tokens, compliance rate, detection recall per failure type, replacement counts and denials, supervision share, with bootstrap CIs and Wilcoxon p-values against the named reference arm.

Operational
GET /healthz returns component readiness including whether spaCy and SBERT models are loaded. GET /v1/version returns git commit, config hashes and pinned model identifiers. GET /v1/spend returns cumulative API spend against the global ceiling; when the ceiling is hit every run-creating endpoint returns global_spend_limit.
