OASIS — Development Plan
Period August 2026 – January 2027 · Team four members · Hard feature freeze 30 September 2026

Reality check on venue timing
The original synopsis targeted AIRC 2026 and CAIN 2026; both were held in April 2026 and are unreachable. The next editions are also problematic: CAIN 2027's research track closes 30 October 2026 and AIRC 2027 around 20 November 2026, both before this project's January completion. All dates must be re-verified on the conference sites, as they shift.

The plan therefore adopts a two-track publication strategy. The primary track finishes the work properly by mid-January, posts an arXiv preprint in December, and submits to a spring-deadline venue — IEEE COMPSAC or EASE typically close in January or February, and IEEE Access and SoftwareX have no deadlines at all, with SoftwareX being a natural home for the benchmark artifact. The fallback track, used only if the team is genuinely ahead by late October, compresses evaluation into three weeks and submits to AIRC 2027 in November. Planning for the primary track and treating November as opportunistic is the honest sequencing; the reverse produces a rushed paper.

Phasing
Phase 0 — Foundations (19–31 August). Repository, CI skeleton, pinned dependency lockfile, LiteLLM gateway with the pre-call hook stub and JSONL sink, SQLite schema and migration runner, FakeLLM stub provider so integration tests need no API key. The gateway lands first because every other component depends on it; building it late would force retrofitting.

Phase 1 — Benchmark and baselines (1–20 September). Task ingestion from MBPP/HumanEval, HotpotQA, GSM8K; authoring of the support-triage and content-generation sets; the three L3 verifiers; the calibration/eval split. In parallel, the vanilla_fixed, single_agent and captainagent baseline arms — CaptainAgent first, since it ships inside AG2 and is the adaptive baseline that keeps the comparison from being a straw man. Gate: all three baselines run end-to-end on 20 tasks with cost accounting.

Phase 2 — Core components (21 September – 30 September, freeze). TCE with the heuristic and LLM-planner variants; RBE with pre-execution reduction and runtime admission control; LangGraph adapter with mid-run swap; AutoGen adapter with boundary swap; RTPM's three layers; RE with budget negotiation; CM. This is compressed deliberately — the components are individually small, and the schedule's real weight is in evaluation, not features. Gate on 30 September: the full arm completes a run end-to-end with a replacement and a budget halt both demonstrated. After this date no new features are added; the descoped FR-10 budget sweep and any topology work are abandoned rather than deferred.

Phase 3 — Calibration and oracle sweep (1–20 October). Threshold calibration on the calibration split only. The oracle sweep at k=1…5 across all eval tasks at three seeds — the most compute-hungry single experiment and the ground truth for H1 and H5, which is why it runs early. The judge-validation exercise: 150 outputs stratified across domains, two members labelling independently, Cohen's κ for human–human and judge–human. Gate: κ reported; if below 0.6, the results plan pivots to verifier-led metrics before the main run rather than after.

Phase 4 — Main experiment (21 October – 20 November). The eight-arm × 100-task × 3-seed matrix, run in tranches with spend monitored against the ceiling after every tranche. Injection runs for detection recall. Memory sessions with shuffled orderings for H5. Failures are re-run, not silently dropped, with re-runs recorded. Gate: complete results table with CIs and p-values.

Phase 5 — Analysis, dashboard, writing (21 November – 20 December). Statistical analysis, figures, the React dashboard against replayed runs, and the paper draft written concurrently rather than afterwards. arXiv preprint by 20 December. The dashboard is scheduled last on purpose: it is a viva and demonstration artifact, not a research contribution, and it is the correct thing to cut if time runs short.

Phase 6 — Artifact and submission (January). Docker packaging, key-scrubbed replay cache, artifact README with one-command reproduction, final report, viva rehearsal, venue submission.

Work distribution
Ownership is by vertical slice so each member has something demonstrable, with the deliberate constraint that no member owns both a mechanism and its evaluation.

Netra Alle (02) owns TCE in both variants, the complexity configuration and calibration, the benchmark suite construction including the five domains and the three L3 verifiers, and the oracle sweep. Writes the estimation and benchmark sections of the paper.

Vedha Bhalerao (11) owns the LiteLLM gateway, RBE in both phases, cost and token accounting, the LangGraph and AutoGen adapters, and the capability contract. Writes the enforcement and integration sections.

Bhagyesh Chaudhari (19) owns RTPM's three layers, the judge rubric and its validation exercise, RE and the budget negotiation, CM, the injection harness and failure taxonomy, and the decision log. Writes the monitoring and replacement sections.

Vidish Bajpai (70) owns the FastAPI service, the async benchmark runner and matrix harness, the statistics module, the replay-mode guarantee, Docker and CI, the React dashboard, related-work consolidation across the twelve-system comparison, and paper assembly.

Cross-cutting: every member reviews at least one PR outside their slice weekly; Bhagyesh and Netra are the two independent human labellers, since neither owns the judge implementation.

Critical path
Gateway → baselines → RBE/RTPM/RE → calibration → oracle sweep → main matrix → analysis → paper. The oracle sweep is the longest-lead compute item and the hardest to compress; the judge validation is the highest-risk gate because a bad κ invalidates downstream quality claims. Both are scheduled in October precisely so a bad result is still recoverable.

Compute and cost planning
The matrix is roughly 100 tasks × 8 arms × 3 seeds = 2,400 runs, plus 100 × 5 × 3 = 1,500 oracle runs, plus injection and memory sessions, for approximately 4,000–4,500 runs and 40,000–60,000 model calls. At small-model pricing with aggressive prompt-hash caching this lands in the low hundreds of dollars; the declared ceiling is set in DEPLOYMENT.md with a hard kill-switch at 120%.

Free-tier keys are not viable — rate limits will stall tranches for hours. A paid key is a project prerequisite. Serial execution would take hundreds of hours, so bounded asyncio concurrency of eight is the default, tuned against provider rate limits rather than local CPU.

Weekly cadence
Monday planning against the current gate, Thursday integration where all four slices must run together on the 5-task smoke set, Friday a written progress note to the guide covering completed items, blockers and spend to date. Weekly spend reporting is deliberate: cost overruns in this project surface as silence, not as errors.
