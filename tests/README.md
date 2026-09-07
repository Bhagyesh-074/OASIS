unit/         pure logic, fixtures, no I/O
contract/     adapter conformance + API schema validation
integration/  whole pipelines against FakeLLM
invariant/    call-count reconciliation, accounting, justification-completeness, leakage, replay-purity
performance/  NFR-1 orchestration overhead, median over 50 runs
