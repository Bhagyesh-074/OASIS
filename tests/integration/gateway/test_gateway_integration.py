from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oasis.gateway.fake_llm import FakeLLM
from oasis.gateway.jsonl_sink import JSONLSink
from oasis.gateway.litellm_hooks import Gateway


class TestGatewayIntegration:
    """Full pre-call → call → post-call pipeline with FakeLLM."""

    def test_full_call_cycle_with_fake_llm(
        self, fake_llm: FakeLLM, tmp_path: Path
    ) -> None:
        """pre-hook → FakeLLM.complete → post-hook → budget updated."""
        class MockBudgetState:
            def __init__(self) -> None:
                self.records: list[dict[str, Any]] = []

            def record(
                self,
                *,
                tokens: int = 0,
                cost_usd: float = 0.0,
                wall_seconds: float = 0.0,
                calls: int = 0,
            ) -> None:
                self.records.append({
                    "tokens": tokens,
                    "cost_usd": cost_usd,
                    "wall_seconds": wall_seconds,
                    "calls": calls,
                })

        model = "gpt-4o-2024-08-06"
        messages = [{"role": "user", "content": "Integration test prompt"}]
        fake_llm.script(
            model=model,
            messages=messages,
            response={
                "content": "Integration test response",
                "usage": {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13},
                "cost_usd": 0.00013,
                "latency_ms": 20.0,
            },
        )

        sink = JSONLSink(tmp_path / "integration_calls.jsonl")
        budget_state = MockBudgetState()

        gw = Gateway(
            sink=sink,
            llm=fake_llm,
            budget_state=budget_state,
            run_id="run-int-1",
            agent_id="agent-int-1",
            purpose="productive",
        )

        assert gw.call_count == 0

        # Execute full cycle through Gateway.call
        resp = gw.call(model=model, messages=messages)

        # 1. Pre-hook verified
        assert gw.call_count == 1
        assert resp["_verdict"] == "allow"

        # 2. Execution verified
        assert resp["content"] == "Integration test response"

        # 3. Post-hook accounting verified
        acct = resp["_accounting"]
        assert acct["tokens_in"] == 8
        assert acct["tokens_out"] == 5
        assert acct["cost_usd"] == 0.00013
        assert acct["latency_ms"] == 20.0

        # 4. JSONL sink record verified
        lines = (tmp_path / "integration_calls.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["model"] == model
        assert record["tokens_in"] == 8
        assert record["tokens_out"] == 5
        assert record["cost_usd"] == 0.00013
        assert record["latency_ms"] == 20.0
        assert record["run_id"] == "run-int-1"
        assert record["agent_id"] == "agent-int-1"

        # 5. Budget state updated verified
        assert len(budget_state.records) == 1
        assert budget_state.records[0]["tokens"] == 13
        assert budget_state.records[0]["cost_usd"] == 0.00013
        assert budget_state.records[0]["calls"] == 1

