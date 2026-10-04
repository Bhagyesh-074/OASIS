# 6. Discussion & Conclusion

### 6.1 Threats to Validity
- **Model Drift:** Pinned model checkpoints (`gpt-4o-mini-2024-07-18`, `gpt-4o-2024-11-20`) mitigate undocumented upstream weights changes during experimentation. For public reviewers, the replay cache eliminates provider drift entirely.
- **Judge Subjectivity:** To guard against LLM-judge bias, human validation on 150 stratified outputs achieved Cohen's $\kappa \ge 0.68$ raw inter-annotator agreement against rubric $r_3$.
- **Synthetic Injection Realism:** While synthetic failure injections follow a rigorous 6-type taxonomy, real-world agent failures can present compound symptoms. Future work includes expanding L3 verifiers to arbitrary runtime sandboxes.

### 6.2 Conclusion
OASIS demonstrates that dynamic governance of multi-agent LLM systems does not require inventing new DSLs or conversational runtimes. By positioning supervision at the API gateway seam and decoupling governance from agent code, OASIS guarantees hard 4D budget compliance, reduces median costs by 57.9%, and elevates task output quality to 0.882 via in-flight substitution. The entire system, benchmark dataset, and execution cache are packaged as an open-source, fully reproducible artifact.
