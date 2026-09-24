"""FR-5 — four-dimension budget schema.

The system shall accept a budget over exactly four dimensions:
  max_tokens, max_cost_usd, max_wall_seconds, max_calls.

GPU and RAM are **not** budget dimensions (ADR-002).  Unknown dimensions
are rejected at validation time.

Budget state tracks consumption against each dimension and exposes
``remaining()`` and ``is_exhausted()`` for the admission controller.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BudgetDimension(str, enum.Enum):
    """The four — and only four — budget dimensions (FR-5, ADR-002)."""

    MAX_TOKENS = "max_tokens"
    MAX_COST_USD = "max_cost_usd"
    MAX_WALL_SECONDS = "max_wall_seconds"
    MAX_CALLS = "max_calls"


class Budget(BaseModel):
    """Immutable budget envelope over the exactly four dimensions (FR-5, ADR-002).

    Parameters
    ----------
    max_tokens : int
        Maximum total tokens allowed across all calls.
    max_cost_usd : float
        Maximum total dollar cost allowed.
    max_wall_seconds : float
        Maximum wall-clock execution time in seconds.
    max_calls : int
        Maximum total number of LLM calls.

    Raises
    ------
    ValidationError / ValueError
        If any unknown dimension name is supplied (e.g. gpu, ram), or if
        any of the four required dimensions is missing or negative.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    max_tokens: int = Field(ge=0, description="Maximum total tokens allowed")
    max_cost_usd: float = Field(ge=0.0, description="Maximum total dollar cost allowed")
    max_wall_seconds: float = Field(ge=0.0, description="Maximum wall-clock execution time in seconds")
    max_calls: int = Field(ge=0, description="Maximum total number of model calls allowed")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def limit(self, dim: BudgetDimension | str) -> int | float:
        """Return the configured limit for *dim*."""
        dim_key = dim.value if isinstance(dim, BudgetDimension) else str(dim)
        if dim_key == BudgetDimension.MAX_TOKENS.value:
            return self.max_tokens
        elif dim_key == BudgetDimension.MAX_COST_USD.value:
            return self.max_cost_usd
        elif dim_key == BudgetDimension.MAX_WALL_SECONDS.value:
            return self.max_wall_seconds
        elif dim_key == BudgetDimension.MAX_CALLS.value:
            return self.max_calls
        raise ValueError(f"Unknown budget dimension: {dim!r}")

    @classmethod
    def validate_dimensions(cls, raw: dict[str, Any]) -> None:
        """Reject a dict containing any key not in :class:`BudgetDimension`.

        This is the FR-5 contract test entry-point.

        Raises
        ------
        ValueError
            On unknown dimension names (e.g. ``gpu``, ``ram``) or missing dimensions.
        """
        valid_dims = {d.value for d in BudgetDimension}
        raw_keys = set(raw.keys())
        unknown = raw_keys - valid_dims
        if unknown:
            raise ValueError(
                f"Unknown budget dimension(s): {sorted(unknown)}. "
                f"Allowed dimensions are: {sorted(valid_dims)}"
            )
        missing = valid_dims - raw_keys
        if missing:
            raise ValueError(
                f"Missing required budget dimension(s): {sorted(missing)}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Budget:
        """Construct and validate a Budget from a dictionary."""
        cls.validate_dimensions(data)
        return cls(**data)


class BudgetState:
    """Mutable consumption tracker for a running execution.

    Created from a :class:`Budget` at run start and updated by the
    post-call hook after every completed LLM call.
    """

    def __init__(self, budget: Budget) -> None:
        self._budget = budget
        self._tokens: int = 0
        self._cost_usd: float = 0.0
        self._wall_seconds: float = 0.0
        self._calls: int = 0
        self._supervision_tokens: int = 0
        self._supervision_cost_usd: float = 0.0
        self._productive_tokens: int = 0
        self._productive_cost_usd: float = 0.0

    @property
    def budget(self) -> Budget:
        """Underlying Budget envelope."""
        return self._budget

    @property
    def tokens(self) -> int:
        return self._tokens

    @property
    def cost_usd(self) -> float:
        return self._cost_usd

    @property
    def wall_seconds(self) -> float:
        return self._wall_seconds

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def supervision_tokens(self) -> int:
        """Total tokens consumed by supervision/judge calls (FR-9, ADR-006)."""
        return self._supervision_tokens

    @property
    def supervision_cost_usd(self) -> float:
        """Total cost in USD consumed by supervision/judge calls (FR-9, ADR-006)."""
        return self._supervision_cost_usd

    @property
    def productive_tokens(self) -> int:
        """Total tokens consumed by productive agent calls."""
        return self._productive_tokens

    @property
    def productive_cost_usd(self) -> float:
        """Total cost in USD consumed by productive agent calls."""
        return self._productive_cost_usd

    def record(
        self,
        *,
        tokens: int = 0,
        cost_usd: float = 0.0,
        wall_seconds: float = 0.0,
        calls: int = 0,
        purpose: str = "productive",
    ) -> None:
        """Record consumption from one call (FR-9, ADR-006).

        Parameters
        ----------
        tokens:
            Tokens consumed by this call.
        cost_usd:
            Dollars consumed by this call.
        wall_seconds:
            Latency in seconds.
        calls:
            Call count increment (usually 1).
        purpose:
            Tag: 'productive' or 'supervision' (matching DATABASE.md).
        """
        if purpose not in ("productive", "supervision"):
            raise ValueError(
                f"Invalid purpose {purpose!r}; must be 'productive' or 'supervision'"
            )
        self._tokens += tokens
        self._cost_usd += cost_usd
        self._wall_seconds += wall_seconds
        self._calls += calls
        if purpose == "supervision":
            self._supervision_tokens += tokens
            self._supervision_cost_usd += cost_usd
        else:
            self._productive_tokens += tokens
            self._productive_cost_usd += cost_usd

    def consumed(self, dim: BudgetDimension | str) -> int | float:
        """Return consumed quantity for *dim*."""
        dim_key = dim.value if isinstance(dim, BudgetDimension) else str(dim)
        if dim_key == BudgetDimension.MAX_TOKENS.value:
            return self._tokens
        elif dim_key == BudgetDimension.MAX_COST_USD.value:
            return self._cost_usd
        elif dim_key == BudgetDimension.MAX_WALL_SECONDS.value:
            return self._wall_seconds
        elif dim_key == BudgetDimension.MAX_CALLS.value:
            return self._calls
        raise ValueError(f"Unknown budget dimension: {dim!r}")

    def remaining(self, dim: BudgetDimension | str) -> int | float:
        """Return remaining capacity for *dim*."""
        limit_val = self._budget.limit(dim)
        consumed_val = self.consumed(dim)
        return limit_val - consumed_val

    def is_exhausted(self) -> bool:
        """Return ``True`` if any dimension has zero or negative remaining."""
        for dim in BudgetDimension:
            if self.remaining(dim) <= 0:
                return True
        return False

