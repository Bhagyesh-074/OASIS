OASIS — Product Requirements Document
Version 2.0 (post-review revision) · Group A-9 · Status Approved for implementation

1. Summary
OASIS is a framework-agnostic supervisory layer that sits above an existing multi-agent LLM framework and adds three coordinated mechanisms: complexity-derived team sizing before execution, runtime budget admission control on every model call, and mid-execution detection and replacement of underperforming agents. It does not replace LangGraph or AutoGen; it wraps them.

Version 1.0 of this project was scoped as a six-feature system claiming three unprecedented capabilities. That claim did not survive a literature check. DyLAN, AgentVerse, CaptainAgent, MaAS, BAMAS, GPTSwarm and AgentPrune each cover part of the ground, and BAMAS in particular covers budget-aware construction plus topology selection. Version 2.0 therefore repositions OASIS around what remains genuinely unaddressed: all prior systems are self-contained frameworks that own their execution loop, and none study what happens when the three mechanisms operate together inside one budget.

2. Research question
Do complexity-based team sizing, runtime budget enforcement, and mid-run agent replacement compose additively, or do they interfere?

The interference hypothesis is concrete and testable. Replacement consumes budget through context handoff, so an active enforcer can forbid a swap the monitor has requested. The monitor itself spends tokens on judge calls, so supervision erodes the savings that smaller teams produce. Both effects are plausible, neither is documented, and measuring them requires exactly the factorial ablation this project will run. A clean negative result — "these mechanisms fight each other under tight budgets, and here is the mechanism" — is a publishable finding and an acceptable outcome.

3. Goals and non-goals
In scope: a working supervisory layer with deep LangGraph integration and shallow AutoGen/AG2 integration; a 100-task benchmark across five domains with three domains anchored in existing verifiable datasets; a factorial ablation across eight configurations; an oracle team-size sweep establishing ground truth for "minimum viable team"; LLM-judge validation against human labels; a replay-cached, reproducible artifact; an audit dashboard; and a paper draft.

Explicitly out of scope: CrewAI as a primary integration, because agents bind to tasks at crew construction and mid-run swapping needs a fork; GPU and RAM as budget dimensions, because API-based inference exposes neither; local model hosting; learned or RL-based policies of any kind, since every OASIS policy is a documented heuristic; production hardening, multi-tenancy, and authentication beyond a static key.

4. Users
The primary user is the researcher-developer running benchmark experiments and reading ablation results — this is who the system is actually built for. The secondary user is a developer with an existing LangGraph or AutoGen pipeline who wants budget ceilings and quality monitoring without migrating frameworks; the API and dashboard are shaped for this persona to keep the design honest. The tertiary user is the examiner at the viva, for whom the dashboard's explainability panel exists.

5. Product behaviour
A run begins with a problem statement and a declared budget over four dimensions: tokens, US dollars, wall-clock seconds, and model-call count. Configuration Memory is queried first; if a semantically similar past task returns a validated configuration, that configuration seeds the run and the estimator is skipped. Otherwise the Task Complexity Estimator segments the statement, scores subtask count, skill diversity and dependency-graph density, and proposes a Minimum Viable Team Size with the three sub-scores attached as justification.

The Resource Budget Enforcer then runs in two phases. Pre-execution it prices the proposed team against the declared budget and merges or prunes roles until the projection fits, logging which dimension forced each reduction. During execution it intercepts every outbound model call and, based on remaining budget, permits, downgrades to a cheaper model, truncates context, or halts the run — this interceptor is what actually delivers budget compliance, since token consumption cannot be known in advance.

As agents produce output, the Performance Monitor scores each one through three layers: embedding cosine drift as a cheap first-pass filter, an LLM-judge rubric for factuality and task adherence on outputs the filter flags or samples, and domain verifiers where they exist. A rolling-window check flags sustained underperformance, and the Replacement Engine draws a substitute from the Agent Template Library and splices it into the running graph with a context handoff. On completion the validated configuration and its quality score are written to Configuration Memory, and every decision is available on the audit dashboard with its justification.

6. Success criteria
These are hypotheses to be tested, not outcomes to be asserted. Each is measured against a defensible baseline and reported with confidence intervals.

ID	Hypothesis	Measured against
H1	OASIS reaches within 5% of best-observed quality using fewer agents than the framework's documented default team size	Vanilla fixed-size arm, oracle sweep
H2	≥90% of runs finish inside all four declared budget dimensions	Vanilla arm compliance rate
H3	Injected-failure detection recall ≥0.70 aggregate, reported per failure type	Random-flag and drift-only detectors
H4	Post-replacement quality delta is positive and larger than the cost delta is negative	No-replacement ablation arm
H5	Memory-seeded configurations show lower regret vs. oracle than cold estimates	Cold-start arm, order-shuffled
H6	Supervision overhead is <500 ms compute and <15% of run tokens	Measured and reported separately
H1's 40% figure from the original synopsis is withdrawn; the reduction depends entirely on how wasteful the chosen baseline is, and a number pre-declared before the oracle sweep would be either luck or reverse-engineering.

7. Risks
The dominant risk is scope. Version 1.0 planned six components, three framework integrations, a dashboard, CI/CD, a 100-task benchmark and a paper for four students in six months — roughly double what fits. Mitigation is a hard feature freeze on 30 September, after which only evaluation, analysis and writing proceed.

Second is judge validity: every quality number depends on an LLM judge, and if judge–human Cohen's κ falls below 0.6 the results section weakens. Mitigation is the 150-output human labelling exercise scheduled deliberately early, in October, plus verifier-based metrics that need no judge at all.

Third is benchmark circularity. If injected failures are crude, cosine drift detects them trivially and the recall number is an artifact of the injection method. Mitigation is the six-type failure taxonomy with subtle numeric and citation corruptions, and per-type recall reporting.

Fourth is provider drift over a six-month project. Mitigation is pinned model versions and the JSONL replay cache, which makes analysis re-runnable even after a model is deprecated.
