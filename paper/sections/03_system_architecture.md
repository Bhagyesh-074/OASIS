# 3. System Architecture

OASIS operates as an out-of-process or sidecar supervisory layer interacting with host frameworks through framework-specific adapters and an intercepting provider gateway.

```
                  +----------------------------------------------+
                  |         Host Runtime (LangGraph/AutoGen)      |
                  +----------------------------------------------+
                                         |
                                (Adapter Interface)
                                         v
   +-------------------------------------------------------------------------+
   |                            OASIS Supervisor                             |
   |                                                                         |
   |  +---------------------+   +---------------------+   +---------------+  |
   |  |   TCE Pre-Sizing    |-->|   4D RBE Admission  |-->|  RTPM Tiered  |  |
   |  |   & Config Memory   |   |   Control & Reducer |   |   Monitoring  |  |
   |  +---------------------+   +---------------------+   +---------------+  |
   |                                                              | (Flags)  |
   |                                                              v          |
   |  +---------------------+                           +-----------------+  |
   |  |  Audit Log & Proven |<--------------------------|  In-Flight ATL  |  |
   |  |  XAI Justifications |                           |  Substitution   |  |
   |  +---------------------+                           +-----------------+  |
   +-------------------------------------------------------------------------+
                                         |
                                  (Gateway Seam)
                                         v
                  +----------------------------------------------+
                  |           LLM Providers / JSONL Cache        |
                  +----------------------------------------------+
```

### 3.1 Task Complexity Estimation (TCE) & Configuration Memory (CM)
Prior to execution, the TCE module inspects the input task prompt using calibrated linguistic metrics (syntactic tree depth, vocabulary density, domain keyword matching) to predict an initial complexity tier (Low, Medium, High) and allocate an initial team size $k \in \{1, \dots, 5\}$. If Configuration Memory contains a historically successful configuration for a similar task ($k\text{-NN}$ cosine similarity $> 0.82$), OASIS seeds the team directly from memory.

### 3.2 Real-Time Budget Enforcement (RBE)
OASIS rejects unobservable dimensions such as GPU and RAM. Budget admission is enforced across a four-dimensional vector:
$$\mathbf{B} = \langle \text{max\_tokens}, \text{max\_cost\_usd}, \text{max\_wall\_seconds}, \text{max\_calls} \rangle$$
Before each call, the RBE pre-call hook computes projected consumption. If the projected call exceeds the remaining budget in any dimension, the call is denied, and the binding constraint is recorded. If pre-execution projections exceed budget, the Team Reducer applies prioritized reductions (model downgrading, temperature zeroing, role pruning). If a run exhausts its allocation during execution, OASIS terminates the run cleanly as `halted_budget` and returns the best intermediate output.

### 3.3 Real-Time Performance Monitor (RTPM)
To avoid the token overhead of peer-rating LLMs, RTPM executes a tiered evaluation:
- **Layer 1 (L1 Embedding Cosine):** Computes cosine similarity between agent output and task prompt via CPU-optimized sentence transformers (`all-MiniLM-L6-v2`, $< 2$ms). Catches gross topical drift.
- **Layer 2 (L2 LLM-Judge Rubric):** Invoked only on L1-flagged outputs or periodic random samples. Scores factuality, completeness, and prompt adherence against rubric $r_3$.
- **Layer 3 (L3 Deterministic Verifiers):** Domain-specific deterministic checkers (syntax compilers for code, numerical equation evaluators for finance, citation resolver for scientific QA).
- **Separation Invariant (FR-15):** L1, L2, and L3 scores are stored in separate columns in SQLite; composite scores never overwrite raw layer scores.

### 3.4 In-Flight Replacement Engine
When an agent accumulates consecutive scores below threshold $\tau_{\text{replace}}$, the Replacement Engine retrieves a substitute from the Agent Template Library (ATL). The swap is admitted only if the estimated handoff token cost fits within remaining budget margin. Upon admission, the adapter extracts the incumbent's state, transfers the context summary, and splices the replacement agent into the running execution graph.

### 3.5 Provenance & Explainability (FR-24)
Every supervisory action (sizing, reduction, escalation, flag, swap, denial, halt) writes an immutable record to the `decision_log` table containing the triggering inputs and a human-readable justification string. Zero automated decisions are made without justification.
