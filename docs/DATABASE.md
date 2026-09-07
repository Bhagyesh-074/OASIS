OASIS — Data Model
Storage is SQLite (WAL mode, foreign_keys=ON) for structured records, plus append-only JSONL on disk for raw model traffic. FAISS is not used: exact cosine over a few hundred 384-dimensional vectors is sub-millisecond, and an approximate index at this scale would be unjustifiable in the paper.

Embeddings are stored as 384-float little-endian BLOBs from all-MiniLM-L6-v2. Free-form structures are TEXT holding JSON, validated at the application layer.

Schema
```sql
-- ============ Benchmark definition ============
CREATE TABLE task (
  task_id           TEXT PRIMARY KEY,
  domain            TEXT NOT NULL CHECK (domain IN
                      ('code_generation','research_qa','quant_analysis',
                       'support_triage','content_generation')),
  source            TEXT NOT NULL,      -- 'mbpp' | 'humaneval' | 'hotpotqa' | 'gsm8k' | 'authored'
  source_ref        TEXT,
  statement         TEXT NOT NULL,
  reference_answer  TEXT,
  verifier_type     TEXT,               -- 'pytest' | 'citation_resolve' | 'numeric_consistency' | NULL
  verifier_spec     TEXT,               -- JSON: test code, expected values, tolerances
  complexity_label  TEXT CHECK (complexity_label IN ('low','medium','high')),
  split             TEXT NOT NULL CHECK (split IN ('calibration','eval')),
  created_at        TEXT NOT NULL
);
CREATE INDEX idx_task_domain_split ON task(domain, split);

-- ============ Runs ============
CREATE TABLE run (
  run_id            TEXT PRIMARY KEY,
  task_id           TEXT NOT NULL REFERENCES task(task_id),
  job_id            TEXT,
  arm               TEXT NOT NULL,      -- vanilla_fixed | single_agent | captainagent
                                        -- | tce_only | rbe_only | monitor_only
                                        -- | tce_rbe | full | oracle
  seed              INTEGER NOT NULL,
  framework         TEXT NOT NULL,      -- langgraph | autogen
  mode              TEXT NOT NULL CHECK (mode IN ('live','replay')),
  budget_json       TEXT NOT NULL,      -- {max_tokens,max_cost_usd,max_wall_seconds,max_calls}
  status            TEXT NOT NULL,      -- queued|running|completed|halted_budget|failed|cancelled
  status_detail     TEXT,
  started_at        TEXT, ended_at TEXT,
  total_tokens      INTEGER DEFAULT 0,
  total_cost_usd    REAL    DEFAULT 0,
  total_calls       INTEGER DEFAULT 0,
  wall_ms           INTEGER,
  supervision_tokens INTEGER DEFAULT 0,
  supervision_cost_usd REAL DEFAULT 0,
  orchestration_ms  INTEGER,            -- NFR-1: compute only, excludes judge network time
  quality_composite REAL,
  compliant_all     INTEGER,            -- 0/1, derived
  manifest_json     TEXT NOT NULL,      -- git commit, model pins, lib versions, config hashes
  UNIQUE (task_id, arm, seed, framework)
);
CREATE INDEX idx_run_job ON run(job_id, arm);

-- ============ Configuration ============
CREATE TABLE team_config (
  config_id    TEXT PRIMARY KEY,
  run_id       TEXT NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
  team_size    INTEGER NOT NULL,
  origin       TEXT NOT NULL,           -- estimator | memory | baseline | oracle
  topology     TEXT NOT NULL,
  roles_json   TEXT NOT NULL,
  mvts_proposed INTEGER,                -- pre-enforcement
  subscores_json TEXT,                  -- subtask_count, skill_clusters, dep_density
  memory_mem_id TEXT REFERENCES config_memory(mem_id),
  memory_similarity REAL
);

CREATE TABLE agent_instance (
  agent_id     TEXT PRIMARY KEY,
  run_id       TEXT NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
  role         TEXT NOT NULL,
  template_id  TEXT NOT NULL,
  model        TEXT NOT NULL,           -- pinned version string
  temperature  REAL NOT NULL,
  active_from_step INTEGER NOT NULL,
  active_to_step   INTEGER,             -- NULL = active at run end
  replaces_agent_id TEXT REFERENCES agent_instance(agent_id)
);
CREATE INDEX idx_agent_run ON agent_instance(run_id);

-- ============ Outputs and scores ============
CREATE TABLE agent_output (
  output_id    TEXT PRIMARY KEY,
  run_id       TEXT NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
  agent_id     TEXT NOT NULL REFERENCES agent_instance(agent_id),
  step         INTEGER NOT NULL,
  content      TEXT NOT NULL,
  content_sha  TEXT NOT NULL,
  prompt_sha   TEXT NOT NULL,           -- join key into raw JSONL log
  purpose      TEXT NOT NULL CHECK (purpose IN ('productive','supervision')),
  tokens_in    INTEGER, tokens_out INTEGER,
  cost_usd     REAL, latency_ms INTEGER,
  UNIQUE (run_id, agent_id, step)
);
CREATE INDEX idx_output_run_step ON agent_output(run_id, step);

-- Layers stored separately (FR-15): never collapse before storage
CREATE TABLE score (
  score_id     TEXT PRIMARY KEY,
  output_id    TEXT NOT NULL REFERENCES agent_output(output_id) ON DELETE CASCADE,
  layer        TEXT NOT NULL CHECK (layer IN ('l1_embed','l2_judge','l3_verifier')),
  relevance    REAL, factuality REAL, adherence REAL,
  completeness REAL, efficiency REAL, composite REAL,
  scorer_model TEXT, rubric_version TEXT,
  cost_usd     REAL DEFAULT 0,
  computed_at  TEXT NOT NULL,
  UNIQUE (output_id, layer)
);

-- ============ Interference events ============
CREATE TABLE budget_event (
  event_id     TEXT PRIMARY KEY,
  run_id       TEXT NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
  step         INTEGER,
  dimension    TEXT NOT NULL CHECK (dimension IN ('tokens','cost','wall','calls')),
  action       TEXT NOT NULL CHECK (action IN ('allow','downgrade','truncate','halt')),
  projected    REAL, remaining REAL,
  model_before TEXT, model_after TEXT,
  justification TEXT NOT NULL,
  created_at   TEXT NOT NULL
);
CREATE INDEX idx_budget_run_action ON budget_event(run_id, action);

CREATE TABLE replacement_event (
  event_id     TEXT PRIMARY KEY,
  run_id       TEXT NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
  step         INTEGER NOT NULL,
  out_agent_id TEXT NOT NULL REFERENCES agent_instance(agent_id),
  in_agent_id  TEXT REFERENCES agent_instance(agent_id),   -- NULL when denied
  outcome      TEXT NOT NULL CHECK (outcome IN
                 ('executed','denied_budget','denied_cap','unsupported_adapter')),
  trigger_dimension TEXT NOT NULL,
  trigger_value REAL, threshold REAL, window_size INTEGER,
  handoff_tokens INTEGER, handoff_cost_usd REAL,
  quality_before REAL, quality_after REAL,   -- windowed means, for H4
  justification TEXT NOT NULL,
  created_at   TEXT NOT NULL
);

-- ============ Memory ============
CREATE TABLE config_memory (
  mem_id       TEXT PRIMARY KEY,
  source_run_id TEXT NOT NULL REFERENCES run(run_id),
  domain       TEXT NOT NULL,
  statement    TEXT NOT NULL,
  embedding    BLOB NOT NULL,           -- 384 float32
  embed_model  TEXT NOT NULL,
  complexity_json TEXT NOT NULL,
  budget_json  TEXT NOT NULL,
  config_json  TEXT NOT NULL,
  quality      REAL NOT NULL,
  cost_usd     REAL NOT NULL,
  created_at   TEXT NOT NULL
);
CREATE INDEX idx_memory_domain ON config_memory(domain);

-- ============ Explainability ============
CREATE TABLE decision_log (
  decision_id  TEXT PRIMARY KEY,
  run_id       TEXT REFERENCES run(run_id) ON DELETE CASCADE,
  component    TEXT NOT NULL CHECK (component IN ('CM','TCE','RBE','ASL','RTPM','RE')),
  decision     TEXT NOT NULL,
  inputs_json  TEXT NOT NULL,
  justification TEXT NOT NULL,          -- FR-24: never NULL, never empty
  created_at   TEXT NOT NULL
);
CREATE INDEX idx_decision_run ON decision_log(run_id, created_at);

-- ============ Evaluation ground truth ============
CREATE TABLE injected_failure (
  inj_id       TEXT PRIMARY KEY,
  run_id       TEXT NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
  output_id    TEXT REFERENCES agent_output(output_id),
  agent_id     TEXT NOT NULL, step INTEGER NOT NULL,
  failure_type TEXT NOT NULL CHECK (failure_type IN
                 ('numeric_perturb','fabricated_citation','silent_truncation',
                  'subtask_substitution','topical_drift','confident_contradiction')),
  severity     TEXT NOT NULL CHECK (severity IN ('subtle','moderate','gross')),
  detected     INTEGER NOT NULL DEFAULT 0,
  detected_at_step INTEGER,
  detected_by_layer TEXT
);
CREATE INDEX idx_inj_type ON injected_failure(failure_type, detected);

CREATE TABLE oracle_sweep (
  task_id      TEXT NOT NULL REFERENCES task(task_id),
  k            INTEGER NOT NULL CHECK (k BETWEEN 1 AND 5),
  seed         INTEGER NOT NULL,
  quality      REAL NOT NULL,
  cost_usd     REAL NOT NULL, tokens INTEGER, wall_ms INTEGER,
  PRIMARY KEY (task_id, k, seed)
);

CREATE TABLE judge_validation (
  jv_id        TEXT PRIMARY KEY,
  output_id    TEXT NOT NULL REFERENCES agent_output(output_id),
  domain       TEXT NOT NULL,
  human_a      REAL NOT NULL, human_b REAL NOT NULL,
  human_a_id   TEXT NOT NULL, human_b_id TEXT NOT NULL,
  judge_score  REAL NOT NULL,
  judge_model  TEXT NOT NULL, rubric_version TEXT NOT NULL,
  labelled_at  TEXT NOT NULL
);
```

Derived quantities
oracle_k is the smallest k whose median quality lies within a documented tolerance (default 0.03) of the best median quality for that task; H1 and H5 are computed as regret against it. compliant_all is written at run finalisation by comparing totals against budget_json. Detection recall per failure type is COUNT(detected=1) / COUNT(*) grouped by failure_type and severity — reported disaggregated, never as a single aggregate, because a single number would be an artifact of the injection mix. replacement_suppression_rate is denied events over total flags, and is the primary interference statistic for the research question.

Raw log
artifacts/llm_calls/{job_id}.jsonl, one object per call: {prompt_sha, run_id, agent_id, purpose, model, temperature, messages, response, tokens_in, tokens_out, cost_usd, latency_ms, ts}. This file is the replay cache and the reproducibility artifact; prompt_sha joins it to agent_output. Keys are scrubbed before release. Expected volume is roughly 40k–60k calls, a few hundred MB, gzipped for the artifact.

Operational notes
SQLite handles the write volume comfortably at eight concurrent runs, but only with WAL enabled and a single writer connection; worker processes queue writes through an asyncio writer task rather than opening independent write connections. Migrations are plain numbered SQL files applied in order, since Alembic would be overhead for a project of this size. Nightly .backup copies go to artifacts/db_snapshots/.
