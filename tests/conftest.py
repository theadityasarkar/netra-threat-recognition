"""Shared pytest fixtures for the NETRA test-suite.

A single small synthetic dataset is built once per session and reused across the graph
and analysis tests — the full pipeline is only run once, keeping the suite fast.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

# Make ``src`` importable when pytest is invoked from the repo root.
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.graph.builder import build_graph            # noqa: E402
from src.ingestion.loader import load_dataset         # noqa: E402
from src.ingestion.sample_data import generate_sample_dataset  # noqa: E402
from src.pipeline import run_pipeline                 # noqa: E402

# Small but structurally rich network — enough syndicates/edges for the GNN to converge,
# small enough that the whole suite runs in a few seconds.
N_PERSONS = 150
N_SYNDICATES = 5
SEED = 7
GNN_EPOCHS = 150


@pytest.fixture(scope="session")
def raw() -> dict:
    """Raw synthetic frames straight from the generator (pre-validation)."""
    return generate_sample_dataset(n_persons=N_PERSONS, n_syndicates=N_SYNDICATES, seed=SEED)


@pytest.fixture(scope="session")
def canonical(raw):
    """``(persons, relationships, crimes, documents, report)`` after ingestion/validation."""
    return load_dataset(raw["persons"], raw["relationships"], raw["crime_records"],
                        raw["documents"])


@pytest.fixture(scope="session")
def graph(canonical):
    """NetworkX graph built from the canonical frames, with co-accused inference."""
    persons, relationships, crimes, _documents, _report = canonical
    return build_graph(persons, relationships, crimes, infer_coaccused=True)


@pytest.fixture(scope="session")
def result(raw):
    """Full pipeline result, computed once for the whole session."""
    return run_pipeline(
        raw["persons"], raw["relationships"], raw["crime_records"],
        documents=raw["documents"], gnn_epochs=GNN_EPOCHS)
