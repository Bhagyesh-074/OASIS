OASIS — System Architecture

1. Architectural position
OASIS is a single-process meta-orchestrator. It does not own the agent execution loop; the underlying framework does. OASIS gains control at exactly three seams, and the design rests on keeping those seams narrow.

The first seam is pre-execution configuration, where OASIS decides team size and composition and hands a configuration to the framework's normal construction path. The second is the model call, intercepted through LiteLLM, which is where budget enforcement and cost accounting happen — and because every framework ultimately makes HTTP calls to a provider, this seam is framework-independent by construction. The third is the output boundary, where OASIS reads each agent's produced message, scores it, and may route to a replacement.

Choosing the LLM gateway as the enforcement point rather than the framework API is the single most consequential decision in this architecture. It converts what would have been three framework-specific hacks into one component, and it is what makes the "framework-agnostic" claim in the paper defensible rather than aspirational.

2. Component map
```
┌──────────────────────────────────────────────────────────────────┐
│  Client:  React dashboard  │  Benchmark runner (CLI)             │
└───────────────────┬──────────────────────────────────────────────┘
                    │  REST + SSE
┌───────────────────▼──────────────────────────────────────────────┐
│  FastAPI service                                                 │
│                                                                  │
│  ┌────────────────┐   ┌──────────────────┐   ┌────────────────┐  │
│  │ Configuration  │──▶│ Task Complexity  │──▶│ Resource       │  │
│  │ Memory (CM)    │   │ Estimator (TCE)  │   │ Budget         │  │
│  │ SQLite + SBERT │   │ spaCy/SBERT/nx   │   │ Enforcer (RBE) │  │
│  └────────▲───────┘   └──────────────────┘   └───────┬────────┘  │
│           │                                          │           │
│           │            ┌─────────────────────────────▼────────┐  │
│           │            │ Agent Spawning Layer (ASL)           │  │
│           │            │  adapter: LangGraph | AutoGen        │  │
│           │            └─────────────────┬────────────────────┘  │
│           │                              │                       │
│           │   ┌──────────────────────────▼───────────────────┐   │
│           │   │ Supervisor loop                              │   │
│           │   │  Perf Monitor (RTPM) ──▶ Replacement (RE)    │   │
│           │   │   L1 cosine · L2 judge · L3 verifier          │   │
│           │   └──────────────────────────┬───────────────────┘   │
│           │                              │                       │
│           └──────────── write-back ──────┘                       │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Decision Log (XAI)  ·  Metrics collector                   │  │
│  └────────────────────────────────────────────────────────────┘  │
└───────────────────┬──────────────────────────────────────────────┘
                    │  ALL model traffic
┌───────────────────▼──────────────────────────────────────────────┐
│  LiteLLM gateway                                                 │
│   pre-call hook: RBE admission control                           │
│   post-call hook: token/cost accounting → metrics                │
│   cache: prompt-hash replay  ·  sink: raw JSONL log               │
└───────────────────┬──────────────────────────────────────────────┘
                    ▼
          OpenAI / Gemini / Anthropic / local stub
```

Persistence is SQLite for structured records and append-only JSONL on disk for raw request/response bodies, linked by SHA-256 prompt hash. There is no message broker; the dashboard receives live updates over server-sent events from the same process that runs the supervisor loop.

3. Data flow for a single run
The runner submits a task and budget. CM embeds the statement with all-MiniLM-L6-v2 and runs exact cosine k-NN (k=3) over stored task embeddings; a hit above the similarity floor with a stored quality above threshold produces a seed configuration. On a miss, TCE tokenises with spaCy, extracts candidate subtasks from clause and imperative structure, embeds each subtask and clusters to obtain a skill-diversity count, builds a dependency graph in NetworkX to obtain density, and combines the three into a proposed team size through a calibrated weighted function.

RBE prices the proposal using per-role token estimates from historical means, compares against all four budget dimensions, and iteratively merges the two most semantically similar roles until the projection fits — recording the binding constraint at each step. ASL translates the final configuration into framework-native constructs: a compiled StateGraph with one node per agent plus a supervisor node for LangGraph, or a GroupChat with a custom speaker selector for AutoGen.

Execution proceeds under the framework's own control. Each model call passes the LiteLLM pre-call hook, which projects the call's cost against remaining budget and returns one of four verdicts. Each completed agent message enters the supervisor, which computes L1 cosine drift against the task and role context, escalates to an L2 judge call if L1 is below the escalation floor or if the output falls in a random audit sample, and applies an L3 verifier where the domain supports one. Scores enter a rolling window of size three; sustained breach triggers RE, which selects a template, requests a handoff budget from RBE, and — if granted — rewrites the graph edge or the speaker roster to route future turns to the replacement.

On completion the metrics collector finalises totals, CM stores the configuration if the run's quality cleared the write threshold, and the decision log is sealed.

4. Interference points, made explicit
Because the research question is about interaction, the architecture deliberately surfaces the three couplings rather than hiding them behind defaults.

RE must request budget from RBE before executing a swap, and RBE may refuse. The refusal is logged as a distinct event type so the paper can count how often replacement was wanted but unaffordable — this is the primary interference measurement.

RTPM's judge calls consume the same budget as productive work and are tagged purpose=supervision in the accounting, so supervision overhead can be reported as a share of total tokens rather than quietly excluded.

TCE's team size changes per-agent budget share, so a smaller team gives each agent more room. Whether this compensates for lost parallelism or compounds into worse quality is measurable only with both mechanisms varying independently, which the ablation matrix provides.

5. Framework adapter contract
Every adapter implements the same five operations: build(config) returns a runnable, run(runnable, task) yields output events, swap(runnable, out_agent, in_agent, handoff) returns success or a not-supported signal, capabilities() declares whether mid-run swap is available, and teardown(). LangGraph reports full mid-run swap. AutoGen reports task-boundary swap only. The paper reports results separately by capability rather than averaging over frameworks with different powers, and the CrewAI adapter exists as a stub returning not-supported, documenting the limitation honestly.

6. Determinism strategy
LLMs are not deterministic, so reproducibility is achieved by recording rather than by re-generating. Every request is hashed over model identifier, full message list, temperature and tool schema; the response is written to JSONL and indexed. Re-running the pipeline in replay mode serves from the cache and performs zero network calls, which means the entire analysis is reproducible by a reviewer with the artifact and no API key. Temperature is fixed at 0 for all supervisory and judge calls; agent calls use a pinned per-domain temperature recorded in the run manifest.
