"""Invariant tests for oasis.rbe — guards claims made in the paper.

Key invariants (TESTING.md):
  - Call-count reconciliation: gateway calls == budget event log (FR-7).
  - Accounting reconciliation: run totals == Σ per-output costs (FR-9).
  - Supervision tokens included in budget, not excluded (ADR-006).
"""
