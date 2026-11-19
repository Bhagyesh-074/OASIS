# OASIS Paper & Publication Artifact Package

This directory contains the assembled manuscript sections, camera-ready LaTeX tables, and reproducible artifact instructions for the paper:

> **OASIS: Dynamic Supervisory Layer for Multi-Agent LLMs with Budget Admission Control and In-Flight Substitution**
> *Vidish Bajpai, Vedha, Netra, Bhagyesh*

## Package Structure

- `sections/`: Markdown manuscript sections drafted and aligned with experimental results:
  - `01_introduction.md`: Motivation, failure modes in multi-agent systems, supervisory gap, and core contributions.
  - `02_related_work.md`: Critical analysis of 12 multi-agent systems published 2023–2025.
  - `03_system_architecture.md`: Formal specification of TCE, RBE (4D admission control), RTPM (L1–L3 tiered scoring), Replacement Engine, and Configuration Memory.
  - `04_experimental_setup.md`: 100-task benchmark, 5 domains (3 anchored), 8 ablation arms, and 6-type synthetic failure injection taxonomy.
  - `05_evaluation_results.md`: Empirical results, 10,000-bootstrap 95% CIs, paired Wilcoxon signed-rank tests, rank-biserial effect sizes, Holm-Bonferroni correction, and disaggregated recall breakdown.
  - `06_discussion_conclusion.md`: Analysis of supervisory overhead, threats to validity, and ethical considerations.
- `tables/`: LaTeX tables ready for submission:
  - `table1_related_work.tex`: Twelve-system literature comparison matrix.
  - `table2_ablation_matrix.tex`: 8-configuration ablation results with bootstrap 95% CIs and significance values.
  - `table3_failure_recall.tex`: Disaggregated failure recall across the 6-type injection taxonomy and 3 severity levels.

## Exact Reproduction Command (FR-30)

Reviewers can verify and reproduce every table and metric in this paper from the included replay cache without requiring any API keys:

```bash
docker compose run api python -m oasis.eval reproduce --job-id ablation-v3 --mode replay
```
All runs execute entirely from hash-keyed local JSONL records with zero network calls, guaranteed by CI test `tests/invariant/eval/test_replay_purity.py`.
