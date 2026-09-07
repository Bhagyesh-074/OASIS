OASIS — Architecture Decision Record
Each entry records the decision, the alternatives, and the consequence. Entries marked reversal overturn a version 1.0 decision, with the reason stated plainly; these are the ones worth defending in the viva, since they are where the project's judgement is visible.

ADR-001 — Reposition the contribution around mechanism interaction. (reversal)

Version 1.0 claimed that complexity-based sizing, budget enforcement and mid-run replacement were each unaddressed in the literature, based on five reviewed papers. A wider search falsified this. DyLAN performs team optimisation with an unsupervised Agent Importance Score and deactivates low-contribution agents during inference. AgentVerse recruits experts per task and adjusts group composition each round from evaluator feedback. CaptainAgent, shipped inside AG2, decomposes tasks and retrieves specialists from an agent library with nested-chat reflection. MaAS samples query-dependent architectures with per-query resource allocation. BAMAS solves an ILP for LLM selection under a hard cost budget and selects interaction topology by reinforcement learning. GPTSwarm and AgentPrune cover topology optimisation and cost-driven pruning.

Absence of evidence in five papers was mistaken for evidence of absence. The contribution is therefore restated as the first study of the interaction among the three mechanisms inside a shared budget, delivered as a supervisory layer over unmodified frameworks rather than a new framework, with an open benchmark. Consequence: related work must cover twelve-plus systems honestly, and the ablation becomes the paper's core rather than a supplement. Interference, if found, is a result rather than a failure.

ADR-002 — Remove GPU and RAM as budget dimensions. (reversal)

The synopsis simultaneously claimed GPU and RAM budget enforcement and stated that no GPU is required because inference is API-based. Both cannot be true; an orchestrator calling a hosted API observes neither quantity. Budget is now exactly tokens, dollars, wall-clock seconds and call count — all directly measurable at the gateway. Consequence: the claim becomes verifiable, and the API rejects unknown dimensions rather than accepting them and ignoring them.

ADR-003 — Enforce budget at the LLM gateway, not the framework API.

Alternatives were per-framework wrappers, which would need three implementations and would break on framework upgrades, or accounting-only observation with no enforcement. Routing all traffic through LiteLLM gives one pre-call hook that works for any framework, plus exact cost accounting and a prompt-hash cache in the same place. Consequence: this is what makes framework-agnosticism real; it also makes the gateway a single point of failure and a mandatory Phase 0 deliverable.

ADR-004 — Budget enforcement is runtime admission control, not pre-execution guarantee. (reversal)

Version 1.0 promised compliance "before the first LLM call is made." Token consumption is stochastic and cannot be known in advance, so this was unachievable as stated. Enforcement is now two-phase: pre-execution estimate-and-configure, plus a runtime check on every call returning allow, downgrade, truncate or halt. Consequence: the 90% compliance hypothesis becomes testable, and halted_budget becomes a legitimate terminal state that returns best-so-far output rather than an error.

ADR-005 — Three-layer output scoring instead of cosine similarity alone. (reversal)

Embedding cosine similarity measures topical relevance and drift. It cannot detect a fluent, on-topic, confidently wrong answer, which is exactly the hallucination case the project promises to catch — it will not notice 4.2 billion where 2.4 billion belongs. Scoring is now L1 cheap cosine as a first-pass filter, L2 an LLM-judge rubric on factuality, adherence and completeness for flagged and randomly audited outputs, and L3 domain verifiers where available. Layers are stored separately and never collapsed. Consequence: the monitor costs tokens, which ADR-006 forces the project to account for honestly, and per-layer detection results become a reportable finding.

ADR-006 — Count supervision cost against the same budget.

Excluding judge calls from the budget would flatter every result. Supervision calls are tagged purpose=supervision and counted normally, and supervision share of total tokens is a reported metric. Consequence: measured savings shrink, and the honest accounting of what supervision costs becomes a distinguishing contribution rather than a weakness.

ADR-007 — Replacement must negotiate with the enforcer, and refusals are logged.

Context handoff for a swapped-in agent consumes tokens. Rather than exempting replacement or letting it silently overrun, RE requests handoff budget and RBE may refuse, writing replacement_denied_budget. Consequence: this coupling is the primary interference measurement for the research question, and suppression rate becomes a headline statistic.

ADR-008 — Deep LangGraph, shallow AutoGen, CrewAI as a documented stub. (reversal)

Three deep integrations do not fit the schedule, and CrewAI in particular binds agents to tasks at crew construction, making mid-run swapping impractical without forking. LangGraph's explicit state and conditional edges make genuine mid-run replacement natural; AutoGen supports boundary-level swaps through a custom speaker selector. Adapters declare capabilities truthfully and results are reported by capability rather than averaged. Consequence: the mid-run replacement claim is scoped to where it is real, and a contract test prevents an adapter from over-claiming.

ADR-009 — Drop Redis; single process with in-process state and SQLite. (reversal)

Redis was specified for runtime state and replacement coordination, but OASIS runs one orchestration loop at a time per run; there is nothing to coordinate across processes. Live dashboard updates use FastAPI server-sent events from the same process. Consequence: one fewer service to deploy, document and defend, at the cost of no horizontal scaling — which this project does not need.

ADR-010 — Drop FAISS; exact cosine k-NN. (reversal)

Configuration Memory will hold at most a few hundred 384-dimensional vectors. An approximate index at that scale is theatre and would invite a fair reviewer question about why. Exact search over the whole store is sub-millisecond. Consequence: FAISS moves to future scope with a stated crossover threshold rather than appearing as an implemented algorithm.

ADR-011 — Establish ground truth by oracle team-size sweep.

"Minimum Viable Team Size" has no natural ground truth, and self-assigned complexity labels would make the estimator's evaluation circular. Each eval task is run at k = 1…5 across three seeds; oracle_k is the smallest k within a documented tolerance of best observed quality. The estimator is then evaluated as a cheap predictor of oracle_k, reported as correlation and regret. Consequence: the single most compute-expensive experiment in the project, scheduled early in Phase 3 because everything downstream depends on it.

ADR-012 — Include an adaptive baseline, not only vanilla.

Comparing against fixed-size AutoGen alone is a straw man and would be identified as such. CaptainAgent is included because it ships inside AG2 and costs days rather than weeks to stand up; single-agent and vanilla-fixed remain as floor and reference. Consequence: the comparison is defensible, and the risk that OASIS shows no advantage over CaptainAgent is accepted — that would itself be a finding worth reporting.

ADR-013 — Validate the judge against human labels, with interpretation fixed in advance.

Every quality number depends on the judge, so the judge is validated before the main experiment: 150 stratified outputs, two independent labellers, Cohen's κ for human–human and judge–human. Thresholds and their consequences are written down before the number is known. The two labellers are the members who do not own the judge implementation. Consequence: an October gate that can force a pivot to verifier-led metrics while there is still time to act.

ADR-014 — Design failures as a six-type taxonomy with severity levels.

Crude injected failures would be caught trivially by L1, making any recall figure an artifact of the injection method rather than a property of the detector. The taxonomy covers numeric perturbation, fabricated citation, silent truncation, subtask substitution, topical drift and confident contradiction, each at subtle, moderate and gross severity, with recall reported per cell against drift-only and random-flag reference detectors. Consequence: aggregate recall will be lower than the original 80% target, and the disaggregated table will be more informative and more credible.

ADR-015 — Reproducibility by recording, not by re-generation.

LLMs are nondeterministic, so exact re-execution is impossible. Every call is hashed and its response written to JSONL; replay mode serves entirely from cache with zero network calls, guarded by a CI test. Model identifiers are pinned to explicit versions and aliases are prohibited. Consequence: a reviewer can reproduce every table without an API key, provider deprecation cannot destroy existing results, and the distinction between exact replay reproduction and approximate live reproduction is documented rather than glossed over.

ADR-016 — Hard feature freeze on 30 September; dashboard scheduled last.

Version 1.0 planned six components, three integrations, a dashboard, CI/CD, a 100-task benchmark and a paper for four students in six months — roughly double a realistic load. After the freeze only calibration, evaluation, analysis and writing proceed. The dashboard, being a demonstration artifact rather than a research contribution, is deliberately last and is the designated cut if time runs short. Consequence: features that miss the freeze, including the optional Pareto budget sweep and any topology derivation, are abandoned rather than deferred into the evaluation window.

ADR-017 — Plan for a spring venue; treat November as opportunistic. (reversal)

AIRC 2026 and CAIN 2026 were held in April 2026 and are unreachable. CAIN 2027 closes around 30 October 2026 and AIRC 2027 around 20 November 2026, both before this project's January completion. The plan therefore targets an arXiv preprint in December and a spring-deadline venue (COMPSAC, EASE, or the no-deadline route of IEEE Access and SoftwareX, the latter suiting the benchmark artifact), with AIRC 2027 as a November fallback only if genuinely ahead. All dates require independent re-verification, as conference schedules move. Consequence: writing runs concurrently with Phase 5 rather than after it, and the artifact is treated as a potential second publication rather than a byproduct.

---

A few things worth flagging as you put these into the repo. The numbers used for the spend ceiling, concurrency, tolerances and thresholds are reasonable defaults, not measured values — replace them with your own once Phase 0 gives you real per-call costs. The model identifiers are illustrative pins and you should substitute whatever you actually have access to, keeping the no-aliases rule. And do verify the CAIN 2027 and AIRC 2027 dates yourself before committing ADR-017 to the report, since the source information stops in May 2026 and those pages change.
