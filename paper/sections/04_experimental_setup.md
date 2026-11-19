# 4. Experimental Setup

### 4.1 Benchmark Design & Anchored Domains
The OASIS benchmark comprises 100 diverse multi-step tasks stratified across five domains:
1. **Code Generation:** Anchored in HumanEval/MBPP with unit test verifiers (L3).
2. **Research QA:** Anchored in PubMedQA with citation verification.
3. **Quantitative Analysis:** Anchored in FinQA with numerical exact-match verifiers.
4. **Support Triage:** Authored realistic multi-step customer escalation workflows.
5. **Content Generation:** Authored multi-section long-form report generation tasks.

Tasks are split into an 80-task evaluation set and a 20-task calibration set. Invariant tests assert zero leakage between calibration and evaluation sets.

### 4.2 Factorial Ablation Matrix
We evaluate eight distinct system configurations to quantify individual and interaction effects:
1. **Unconstrained Baseline:** Full team running on raw LangGraph without budget limits or monitoring.
2. **Vanilla Fixed:** Fixed 3-agent team with passive budgeting.
3. **Single Agent Baseline:** Single generalist LLM prompt.
4. **CaptainAgent Baseline:** Nested sub-agent generation.
5. **TCE Only:** Dynamic pre-execution team sizing without runtime budget enforcement or monitoring.
6. **RBE Only:** Strict 4D budget admission control without RTPM monitoring or replacements.
7. **RTPM Only:** Tiered monitoring and in-flight replacement without pre-execution sizing.
8. **Monitor Only:** Passive RTPM scoring without in-flight remediation.
9. **TCE + RBE:** Pre-sizing combined with budget enforcement.
10. **OASIS Full (Ours):** TCE + RBE + RTPM + Replacement Engine + Configuration Memory.

Each configuration is evaluated across all benchmark tasks under three distinct random seeds ($N = 240$ runs per arm).

### 4.3 Synthetic Failure Injection Taxonomy (FR-28)
To evaluate detector recall without artifacts, we author a controlled synthetic failure injection harness implementing a 6-type failure taxonomy:
1. **Topical Drift:** Semantic drift away from task objectives.
2. **Numeric Perturbation:** Altering numerical figures, percentages, or statistical values.
3. **Fabricated Citation:** Generating plausible but non-existent bibliographic references.
4. **Silent Truncation:** Abruptly cutting off outputs mid-generation.
5. **Subtask Substitution:** Answering an easier adjacent question rather than the requested task.
6. **Confident Contradiction:** Asserting direct factual contradictions with high confidence.

Each failure type is injected at three calibrated severity levels: Subtle, Moderate, and Gross.
