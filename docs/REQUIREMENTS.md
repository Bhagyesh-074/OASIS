OASIS — Requirements Specification
Each requirement carries an ID, a priority (M must, S should, C could), the component that owns it, and the test that verifies it. IDs are referenced from TESTING.md.

Functional — Task Complexity Estimator
FR-1 (M, TCE) Given a problem statement, the system shall produce an integer Minimum Viable Team Size in the range 1–8. Verified by unit tests on 20 fixture statements.

FR-2 (M, TCE) The estimate shall be accompanied by three named sub-scores — subtask count, skill-diversity cluster count, dependency-graph density — and the weighted contribution of each. Verified by schema assertion on the estimate response.

FR-3 (M, TCE) Weights and thresholds shall be loaded from a version-controlled configuration file, not hard-coded, and calibrated only on the held-out calibration split. Verified by config-load test and a leakage check asserting no calibration task appears in the evaluation split.

FR-4 (S, TCE) The system shall support an alternative LLM-planner estimator behind the same interface, for use as a baseline arm. Verified by integration test running both estimators on the same task.

Functional — Resource Budget Enforcer
FR-5 (M, RBE) The system shall accept a budget over exactly four dimensions: max_tokens, max_cost_usd, max_wall_seconds, max_calls. GPU and RAM are not budget dimensions. Verified by API contract test rejecting unknown dimensions.

FR-6 (M, RBE) Pre-execution, the system shall project team cost and reduce the team by merging or pruning roles until the projection satisfies all four dimensions, logging the binding constraint for each reduction. Verified by unit tests with budgets forcing 1, 2 and 3 reductions.

FR-7 (M, RBE) Every outbound model call shall pass an admission check returning one of allow, downgrade, truncate, halt. No call may bypass the check. Verified by an integration test asserting call count at the gateway equals call count in the budget event log.

FR-8 (M, RBE) On halt, the run shall terminate with status halted_budget and return the best output produced so far rather than raising an error. Verified by integration test with a deliberately tiny budget.

FR-9 (M, RBE) Supervision calls shall be tagged purpose=supervision and counted against the same budget as productive calls. Verified by accounting reconciliation test.

FR-10 (C, RBE) The system shall optionally compute a three-point budget sweep (tight, moderate, generous) exposing the quality–cost tradeoff before execution. Verified by integration test; descoped if behind schedule.

Functional — Performance Monitor
FR-11 (M, RTPM) Each agent output shall receive an L1 embedding-drift score in [0,1] computed as cosine similarity against task and role context. Verified by unit test with known-similar and known-dissimilar pairs.

FR-12 (M, RTPM) Outputs below the L1 escalation floor, plus a configurable random audit fraction, shall receive an L2 judge score on factuality, task adherence and completeness against a versioned rubric. Verified by integration test asserting escalation policy.

FR-13 (M, RTPM) Where the domain declares a verifier, an L3 verifier score shall be computed — pytest execution for code, DOI/URL resolution for research, arithmetic consistency for quantitative tasks. Verified by per-verifier unit tests including deliberate-failure fixtures.

FR-14 (M, RTPM) An agent shall be flagged when its composite score falls below threshold across a rolling window of configurable size. Single-output dips shall not trigger. Verified by unit test over synthetic score sequences.

FR-15 (M, RTPM) L1, L2 and L3 scores shall be stored separately and never collapsed before storage, so per-layer detection performance is recoverable. Verified by database schema test.

Functional — Replacement Engine
FR-16 (M, RE) On a flag, the engine shall select a template from the Agent Template Library using the breached score dimension, and record the dimension, value, threshold and chosen template. Verified by unit tests for each dimension.

FR-17 (M, RE) The engine shall request handoff budget from RBE and shall not proceed if refused; refusals shall be logged as replacement_denied_budget. Verified by integration test with a budget insufficient for handoff.

FR-18 (M, RE) On LangGraph, replacement shall occur without restarting the run, preserving accumulated state. Verified by integration test asserting pre-swap state survives.

FR-19 (S, RE) On AutoGen, replacement shall occur at the next task boundary; the adapter shall declare this capability truthfully. Verified by capability contract test.

FR-20 (M, RE) At most max_replacements_per_run swaps shall occur, and no agent shall be replaced more than once by the same template. Verified by unit test on an adversarial always-failing agent.

Functional — Configuration Memory
FR-21 (M, CM) Completed runs whose quality clears the write threshold shall be stored with task embedding, complexity profile, budget, final configuration, quality and cost. Verified by integration test.

FR-22 (M, CM) Retrieval shall use exact cosine k-NN with k=3 over stored embeddings and shall return a seed configuration only above a similarity floor. Verified by unit test with paraphrased and unrelated queries.

FR-23 (M, CM) Retrieval shall be excludable by run, to support cold-start ablation arms and leave-one-out evaluation. Verified by ablation harness test.

Functional — Explainability and audit
FR-24 (M, XAI) Every automated decision — sizing, reduction, escalation, flag, swap, denial, halt — shall write a decision-log record with component, decision, inputs, and a human-readable justification string. Verified by a test asserting zero decisions without a justification.

FR-25 (M, API) The dashboard shall receive live run events over SSE and display agent scores, budget consumption per dimension, team composition and replacement events with justifications. Verified by frontend integration test against a replayed run.

Functional — Evaluation harness
FR-26 (M, EVAL) The harness shall execute the eight-configuration ablation matrix across all benchmark tasks at three seeds, with bounded asyncio concurrency. Verified by smoke run on a 5-task subset.

FR-27 (M, EVAL) The harness shall support an oracle sweep running each task at k = 1…5 to establish ground-truth minimum viable team size. Verified by smoke run.

FR-28 (M, EVAL) The injection harness shall corrupt agent outputs according to the six-type failure taxonomy at a configurable rate, recording ground truth for recall computation. Verified by per-type injection unit tests.

FR-29 (M, EVAL) The harness shall emit per-run and aggregate metrics with medians, bootstrap 95% confidence intervals, Wilcoxon signed-rank p-values and effect sizes. Verified by statistics unit tests against known distributions.

FR-30 (M, EVAL) A replay mode shall execute the full pipeline from cache with zero network calls. Verified by CI test asserting network calls equal zero.

Non-functional
NFR-1 (M) Supervisory compute overhead per task, excluding judge network latency, shall be under 500 ms measured on the reference hardware; judge-inclusive latency shall be reported separately and is not bounded.

NFR-2 (M) The system shall run on 4 CPU cores and 16 GB RAM with no GPU.

NFR-3 (M) Total API spend for the full evaluation shall stay under a declared ceiling, with a hard global kill-switch at 120% of ceiling.

NFR-4 (M) Every result file shall be accompanied by a manifest recording model versions, seeds, library versions, config file hashes and git commit.

NFR-5 (M) Model identifiers shall be pinned to explicit versions; aliases such as latest are prohibited.

NFR-6 (S) Raw request/response logs shall be scrubbed of API keys before artifact release.

NFR-7 (M) Adding a new framework adapter shall require no changes to TCE, RBE, RTPM, RE or CM.
