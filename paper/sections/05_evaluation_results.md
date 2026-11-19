# 5. Evaluation Results

### 5.1 Ablation Study & Quality-Cost Frontier
Table 2 reports the central empirical findings of the 8-configuration factorial ablation. All metrics represent medians with 10,000-resample bootstrap 95% confidence intervals. Statistical significance is computed against the unconstrained baseline using paired Wilcoxon signed-rank tests with Holm-Bonferroni family-wise error rate correction.

1. **Quality Improvement:** **OASIS Full** achieves a median composite quality of **0.882 [0.865, 0.898]**, outperforming the unconstrained baseline ($0.715$, $p_{adj} < 0.001$, effect size $r_{rb} = +0.74$) and the fixed 3-agent baseline ($0.730$). In-flight replacement of failing agents accounts for a $+0.101$ quality gain over RBE-only ($0.781$).
2. **Cost & Token Reduction:** Despite achieving higher quality, OASIS Full cuts median cost from **$0.195** to **$0.082** (a 57.9% reduction) and reduces token consumption from **34,500** to **14,200** (a 58.8% reduction). This efficiency stems from TCE sizing simple tasks to $k=1\dots2$ agents rather than spinning up full 5-agent teams unnecessarily.
3. **Zero Budget Violations:** The unconstrained baseline and CaptainAgent suffer from budget breach rates of **52.0%** and **61.0%** respectively. In contrast, all configurations incorporating RBE (RBE Only, TCE+RBE, OASIS Full) achieve an absolute **0.0% violation rate**, verifying the correctness of the 4D admission controller.

### 5.2 Failure Detection Recall (Taxonomy Disaggregation)
Table 3 presents disaggregated detection recall across the six failure types and three severity levels.

- **Topical Drift** is detected almost perfectly by L1 embedding cosine drift alone (81.2% subtle, 99.1% gross), justifying L1 as a high-throughput, low-cost first filter.
- **Subtle Factual Errors:** In contrast, subtle Numeric Perturbations (14.1% on L1) and Fabricated Citations (9.4% on L1) largely bypass embedding cosine similarity. The tiered invocation of L2 judge rubrics and L3 deterministic verifiers elevates recall to **64.2%** and **58.1%** respectively on subtle errors, and **94.8%** and **91.2%** on gross errors.
- **Micro-Average Recall:** Overall micro-average recall across all 18 cells reaches **79.7%** for OASIS Tiered Monitoring, compared to 45.6% for Drift-Only and 5.0% for Random Flagging.

### 5.3 Supervisory Overhead & Replay Purity
- **Compute Latency (NFR-1):** Median orchestration compute overhead is measured at **3.4 ms** on CPU, well within the 50 ms budget. Embedding inference via MiniLM adds 1.8 ms per output.
- **Replay Verification (FR-30):** The full 100-task matrix execution in replay mode executes with exactly zero socket connections, as formally verified by `tests/invariant/eval/test_replay_purity.py`.
