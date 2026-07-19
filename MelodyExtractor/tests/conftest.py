"""Shared pytest fixtures. Fixture WAV/MIDI pairs are generated on demand
(deterministically) into tests/fixtures/ by generate_fixtures.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"

# Allow `import synth_util` from any test module regardless of invocation cwd.
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Ensure the fixture corpus exists (generated deterministically) and return its dir."""
    gen = TESTS_DIR / "fixtures" / "generate_fixtures.py"
    if gen.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("generate_fixtures", gen)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.generate_all(FIXTURES_DIR)
    return FIXTURES_DIR
