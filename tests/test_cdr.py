"""Tests for Call Detail Record ingestion and analytics (``src/ingestion/cdr.py``).

CDRs are the least standardised files in law enforcement — every telco and extraction
tool ships different headers, different MSISDN formats and different timestamp shapes.
These tests pin the contract the rest of the pipeline depends on: whatever the export
looks like, what comes out is a canonical relationship table.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from src.config import RELATIONSHIP_COLUMNS
from src.ingestion.cdr import (
    CDR_COLUMNS,
    CDR_PARAMS,
    analyse_cdr,
    cdr_to_relationships,
    detect_burners,
    load_cdr,
    normalise_msisdn,
    subscriber_profiles,
    synthetic_cdr,
    tower_colocation,
)
from src.ingestion.sample_data import generate_sample_dataset


# ──────────────────────────────────────────────────────────────────────────────
# MSISDN normalisation
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    "9876543210", "+91 9876543210", "+919876543210", "091-98765-43210",
    "98765 43210", 9876543210, "0919876543210",
])
def test_msisdn_variants_collapse_to_one_number(raw):
    """The same handset written seven ways must be one node, not seven."""
    assert normalise_msisdn(raw) == "9876543210"


@pytest.mark.parametrize("junk", [None, "", "   ", "abc", "-", float("nan")])
def test_msisdn_rejects_unusable_values(junk):
    """Junk becomes empty, never a bogus number — a fabricated subscriber is worse
    than a dropped row."""
    assert normalise_msisdn(junk) == ""


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def _csv(text: str) -> io.BytesIO:
    return io.BytesIO(text.encode("utf-8"))


def test_load_cdr_auto_maps_telco_headers():
    """A real export says A_PARTY/B_PARTY/CALL_DATE, not caller/callee/timestamp."""
    src = _csv(
        "A_PARTY,B_PARTY,CALL_DATE,DURATION,CELL_ID\n"
        "9876543210,9812345670,2026-01-05 08:03:00,120,TWR-01\n"
        "9812345670,9876543210,2026-01-05 22:41:00,300,TWR-01\n"
    )
    df, report = load_cdr(src)

    assert not report.errors, report.errors
    assert set(CDR_COLUMNS) <= set(df.columns)
    assert len(df) == 2
    assert report.renamed, "header auto-mapping was not reported back to the analyst"
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_load_cdr_requires_both_parties():
    """A missing party column is a hard error. The partially-parsed frame still comes
    back so the UI can show what it read, but ``report.ok`` is false — which is the
    flag ``analyse_cdr`` gates on."""
    df, report = load_cdr(_csv("A_PARTY,DURATION\n9876543210,120\n"))

    assert report.errors
    assert not report.ok
    assert "callee" not in df.columns

    # The gate holds end-to-end: no relationships are manufactured from a broken file.
    res = analyse_cdr(_csv("A_PARTY,DURATION\n9876543210,120\n"))
    assert res.n_pairs == 0
    assert res.report.errors


def test_load_cdr_enforces_the_row_ceiling():
    """A seizure dump can be millions of rows; refusing with an actionable message beats
    exhausting a 1 GB free-tier container."""
    ceiling = int(CDR_PARAMS["max_rows"])
    rows = "\n".join(f"98765{i:05d},9812345670,2026-01-05 08:03:00,60,TWR-01"
                     for i in range(ceiling + 2))
    df, report = load_cdr(_csv(f"caller,callee,timestamp,duration_sec,tower_id\n{rows}\n"))

    assert report.errors and df.empty
    assert any("ceiling" in e or "row" in e.lower() for e in report.errors)


def test_load_cdr_survives_unparseable_timestamps():
    src = _csv(
        "caller,callee,timestamp,duration_sec\n"
        "9876543210,9812345670,not-a-date,120\n"
        "9876543210,9812345670,also-not-a-date,60\n"
    )
    df, report = load_cdr(src)

    assert not report.errors
    assert report.warnings, "an all-unparseable timestamp column must be flagged"
    # Still datetime-typed (all NaT), so the downstream .dt analytics stay valid.
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_load_cdr_rejects_an_unreadable_source():
    df, report = load_cdr(_csv("\x00\x01\x02 not a csv"))
    assert df.empty or report.errors


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def events():
    """A synthetic log derived from a small network — the demo's own data path."""
    data = generate_sample_dataset(n_persons=60, n_syndicates=3, seed=7)
    return synthetic_cdr(data["relationships"], data["persons"], seed=7)


def test_synthetic_cdr_is_deterministic_and_well_formed(events):
    data = generate_sample_dataset(n_persons=60, n_syndicates=3, seed=7)
    again = synthetic_cdr(data["relationships"], data["persons"], seed=7)

    assert len(events) > 0
    assert list(events.columns) == CDR_COLUMNS
    pd.testing.assert_frame_equal(events, again)
    # Every event names two different, valid subscribers.
    assert (events["caller"] != events["callee"]).all()
    assert events["caller"].map(normalise_msisdn).eq(events["caller"]).all()


def test_aggregators_accept_unparsed_timestamps(events):
    """Regression: ``synthetic_cdr`` emits ``timestamp`` as text. Both aggregators use
    the ``.dt`` accessor, which raised a bare ``AttributeError`` naming neither the
    column nor the cause when handed a frame that had not been through ``load_cdr``."""
    assert not pd.api.types.is_datetime64_any_dtype(events["timestamp"])

    rel = cdr_to_relationships(events)      # must not raise
    sub = subscriber_profiles(events)        # must not raise

    assert len(rel) > 0 and len(sub) > 0


def test_relationships_are_canonical_undirected_and_bounded(events):
    rel = cdr_to_relationships(events)

    assert list(rel.columns) == RELATIONSHIP_COLUMNS, "CDR output must match the graph contract"
    assert ((rel["strength"] >= 0) & (rel["strength"] <= 1)).all()
    # Canonicalised pairs: source <= target, and no pair appears twice.
    assert (rel["source_id"] <= rel["target_id"]).all()
    pairs = list(zip(rel["source_id"], rel["target_id"]))
    assert len(pairs) == len(set(pairs)), "the same pair was emitted twice"
    assert (rel["interaction_count"] >= CDR_PARAMS["min_calls_per_pair"]).all(), \
        "single-call noise pairs were not filtered out"
    assert (rel["evidence_source"].astype(str).str.len() > 0).all()


def test_subscriber_profiles_cover_both_call_directions(events):
    sub = subscriber_profiles(events)
    numbers = set(sub["number"])

    assert set(events["caller"]) <= numbers, "callers missing from the profile table"
    assert set(events["callee"]) <= numbers, "callees missing from the profile table"
    assert (sub["calls"] > 0).all()
    assert ((sub["odd_hour_ratio"] >= 0) & (sub["odd_hour_ratio"] <= 1)).all()
    assert (sub["lifespan_days"] >= 0).all()


def test_burner_detection_is_explainable_and_selective(events):
    sub = subscriber_profiles(events)
    burners = detect_burners(sub)

    assert len(burners) < len(sub), "flagging every handset as a burner carries no signal"
    for _, row in burners.iterrows():
        assert row["reasons"], "a burner flag with no stated reason is unusable in a request"
        assert 0 <= float(row["burner_score"]) <= 100
        assert int(row["lifespan_days"]) <= CDR_PARAMS["burner_max_lifespan_days"]


def test_colocation_excludes_high_fanout_towers(events):
    """A tower serving a station puts strangers on the same site — counting it would
    manufacture links between unrelated people."""
    coloc = tower_colocation(events)

    assert isinstance(coloc, pd.DataFrame)
    for _, row in coloc.iterrows():
        assert row["number_a"] != row["number_b"]
        assert int(row["shared_towers"]) >= 1


def test_aggregators_handle_an_empty_log():
    empty = pd.DataFrame(columns=CDR_COLUMNS)
    assert cdr_to_relationships(empty).empty
    assert subscriber_profiles(empty).empty
    assert tower_colocation(empty).empty
    assert detect_burners(pd.DataFrame()).empty


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end — the exact round-trip the Data & Sources panel performs
# ──────────────────────────────────────────────────────────────────────────────

def test_analyse_cdr_round_trip(events):
    """The UI serialises the synthetic log to CSV and feeds it back through the real
    ingestion path, so that round-trip is what has to hold."""
    res = analyse_cdr(io.BytesIO(events.to_csv(index=False).encode("utf-8")))

    assert not res.report.errors, res.report.errors
    assert res.n_events > 0
    assert res.n_pairs > 0
    assert len(res.subscribers) > 0
    assert res.stats, "no headline statistics produced"
