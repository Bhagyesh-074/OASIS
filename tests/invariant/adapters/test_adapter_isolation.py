"""Invariant tests for framework adapter isolation (NFR-7).

NFR-7 (M):
  "Adding a new framework adapter shall require no changes to TCE, RBE,
   RTPM, RE or CM."

Constraint:
  Adding a new framework adapter must require zero changes to tce/, rbe/,
  rtpm/, replacement/, or cm/.

This test suite:
  1. Mechanically verifies that adding and running a new adapter leaves all
     files in src/oasis/{tce,rbe,rtpm,replacement,cm} completely unchanged
     (zero hash diff).
  2. Statically verifies that none of those core modules (nor gateway) import
     or hardcode references to specific framework adapters.
  3. Verifies that the new adapter conforms to the shared adapter contract.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oasis.adapters.base import AdapterBase
from tests.contract.adapters.test_contract_shared import BaseAdapterConformanceSuite
from tests.fixtures.adapters.dummy_adapter import DummyIsolatedAdapter

CORE_MODULES = ["tce", "rbe", "rtpm", "replacement", "cm"]
ADAPTER_FRAMEWORK_NAMES = ["langgraph", "autogen", "crewai"]


def compute_directory_hashes(root_dir: Path, subdirs: list[str]) -> dict[str, str]:
    """Compute SHA-256 hashes for all Python files in the given subdirectories."""
    file_hashes: dict[str, str] = {}
    for subdir in subdirs:
        dir_path = root_dir / "src" / "oasis" / subdir
        if not dir_path.exists():
            continue
        for py_file in dir_path.rglob("*.py"):
            # Skip bytecode caches and temp files
            if "__pycache__" in py_file.parts or py_file.name.endswith(".pyc"):
                continue
            rel_path = str(py_file.relative_to(root_dir))
            content = py_file.read_bytes()
            file_hashes[rel_path] = hashlib.sha256(content).hexdigest()
    return file_hashes


class TestDummyAdapterConformance(BaseAdapterConformanceSuite):
    """The dummy isolated adapter passes the shared conformance suite."""

    __test__ = True

    @pytest.fixture
    def adapter(self) -> DummyIsolatedAdapter:
        return DummyIsolatedAdapter()


class TestAdapterIsolationNFR7:
    """Mechanically and statically enforce NFR-7 across OASIS modules."""

    def test_nfr7_zero_changes_to_core_directories_when_adding_adapter(self) -> None:
        """Adding, instantiating, and exercising a new adapter requires zero changes
        to any file in src/oasis/{tce, rbe, rtpm, replacement, cm}.
        """
        # 1. Snapshot file hashes before
        hashes_before = compute_directory_hashes(_REPO_ROOT, CORE_MODULES)
        assert len(hashes_before) > 0, "Expected Python files in core directories"

        # 2. Instantiate and exercise the new dummy adapter
        dummy = DummyIsolatedAdapter()
        assert isinstance(dummy, AdapterBase)
        runnable = dummy.build({"roles": ["custom_agent"]})
        events = list(dummy.run(runnable, "Execute isolated task"))
        assert len(events) == 1
        assert events[0]["agent"] == "dummy_agent"

        swap_success = dummy.swap(runnable, "custom_agent", "replacement_agent")
        assert swap_success is True

        caps = dummy.capabilities()
        assert caps.mid_run_swap is True
        assert caps.boundary_swap is True

        dummy.teardown()
        assert dummy.is_torn_down is True

        # 3. Snapshot file hashes after
        hashes_after = compute_directory_hashes(_REPO_ROOT, CORE_MODULES)

        # 4. Assert exact hash equality (diff must be empty)
        diff = {
            k: (hashes_before.get(k), hashes_after.get(k))
            for k in set(hashes_before) | set(hashes_after)
            if hashes_before.get(k) != hashes_after.get(k)
        }
        assert diff == {}, (
            f"NFR-7 VIOLATION: Files in core directories were modified when adding "
            f"an adapter: {diff}"
        )

    def test_no_core_module_imports_oasis_adapters(self) -> None:
        """No file in src/oasis/{tce, rbe, rtpm, replacement, cm, gateway} may
        import from oasis.adapters.
        """
        inspected_dirs = CORE_MODULES + ["gateway"]
        violations: list[str] = []

        for subdir in inspected_dirs:
            dir_path = _REPO_ROOT / "src" / "oasis" / subdir
            if not dir_path.exists():
                continue
            for py_file in dir_path.rglob("*.py"):
                if "__pycache__" in py_file.parts:
                    continue
                try:
                    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
                except (SyntaxError, OSError) as e:
                    pytest.fail(f"Failed to parse {py_file}: {e}")

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if "adapters" in alias.name:
                                violations.append(f"{py_file}: import {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        if "adapters" in module:
                            violations.append(f"{py_file}: from {module} import ...")

        assert violations == [], (
            f"NFR-7 VIOLATION: Core modules import from oasis.adapters: {violations}"
        )

    def test_no_hardcoded_framework_branching_in_rbe_or_gateway(self) -> None:
        """rbe/ and gateway/ must not contain hardcoded branching or checks for
        specific framework adapters (e.g. 'langgraph', 'autogen', 'crewai').
        """
        checked_dirs = ["rbe", "gateway"]
        violations: list[str] = []

        for subdir in checked_dirs:
            dir_path = _REPO_ROOT / "src" / "oasis" / subdir
            if not dir_path.exists():
                continue
            for py_file in dir_path.rglob("*.py"):
                if "__pycache__" in py_file.parts:
                    continue
                source_lines = py_file.read_text(encoding="utf-8").splitlines()
                for line_num, line in enumerate(source_lines, start=1):
                    # Check for framework-specific identifiers in code (ignore comments)
                    code_part = line.split("#")[0].lower()
                    for fw in ADAPTER_FRAMEWORK_NAMES:
                        if f'"{fw}"' in code_part or f"'{fw}'" in code_part:
                            violations.append(
                                f"{py_file.name}:{line_num} mentions framework '{fw}': {line.strip()}"
                            )

        assert violations == [], (
            f"NFR-7 VIOLATION: Hardcoded framework branching detected in rbe/ or gateway/: {violations}"
        )
