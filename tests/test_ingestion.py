"""Ingestion-layer tests: schema validation, column auto-mapping, ID parsing."""

from __future__ import annotations

import pandas as pd

from src.config import PERSON_REQUIRED, RELATIONSHIP_REQUIRED
from src.ingestion.loader import (
    load_persons,
    load_relationships,
    parse_id_list,
    prepare_dataset,
)


def test_sample_dataset_has_required_columns(raw):
    persons, rels, crimes = raw["persons"], raw["relationships"], raw["crime_records"]
    assert len(persons) > 0 and len(rels) > 0
    for col in PERSON_REQUIRED:
        assert col in persons.columns
    for col in RELATIONSHIP_REQUIRED:
        assert col in rels.columns
    assert "persons_involved" in crimes.columns


def test_load_dataset_is_clean_and_consistent(canonical):
    persons, rels, crimes, documents, report = canonical
    assert report.ok, f"unexpected ingestion errors: {report.errors}"
    # person_id is always a string key.
    assert persons["person_id"].map(type).eq(str).all()
    # Every relationship endpoint resolves to a known person (stubs are created if needed).
    ids = set(persons["person_id"])
    assert set(rels["source_id"]).issubset(ids)
    assert set(rels["target_id"]).issubset(ids)
    # Canonical crime frame exposes the parsed ID list used for co-accused inference.
    if len(crimes):
        assert "_ids" in crimes.columns
    # The sample corpus flows through ingestion and every document keeps its text.
    assert len(documents)
    assert (documents["text"].str.len() > 0).all()


def test_column_auto_mapping(raw):
    """A ``strength`` column supplied under the alias ``weight`` is mapped back."""
    rels = raw["relationships"].rename(columns={"strength": "weight"})
    assert "weight" in rels.columns and "strength" not in rels.columns
    loaded, report = load_relationships(rels)
    assert "strength" in loaded.columns
    # The rename is recorded for the data-quality report.
    assert any(v == "strength" for v in report.renamed.values())


def test_missing_required_column_is_an_error():
    rels = pd.DataFrame({"source_id": ["P1", "P2"]})  # no target_id
    _, report = load_relationships(rels)
    assert not report.ok
    assert report.errors


def test_orphan_edges_are_kept_with_warning():
    """Links to unknown IDs are preserved (stubs get created later) but flagged —
    an evidence-preservation design choice: a bad roster must not silently drop links."""
    rels = pd.DataFrame({
        "source_id": ["P1", "P1"],
        "target_id": ["P2", "GHOST"],   # GHOST is not in valid_ids
        "relation_type": ["Associate", "Associate"],
    })
    loaded, report = load_relationships(rels, valid_ids={"P1", "P2"})
    assert len(loaded) == 2                       # nothing dropped
    # Endpoints are canonicalised (sorted) into an undirected pair, so GHOST may land
    # in either column — assert it survives somewhere.
    endpoints = set(loaded["source_id"]) | set(loaded["target_id"])
    assert "GHOST" in endpoints
    assert any("absent from persons" in w for w in report.warnings)


def test_prepare_dataset_creates_stubs_for_unknown_ids():
    persons = pd.DataFrame({"person_id": ["P1"], "name": ["A"]})
    rels = pd.DataFrame({"source_id": ["P1"], "target_id": ["P2"],
                         "relation_type": ["Associate"], "strength": [0.5]})
    out_persons, _, _, report = prepare_dataset(persons, rels)
    assert "P2" in set(out_persons["person_id"].astype(str))
    assert any("stub" in w.lower() for w in report.warnings)


def test_parse_id_list():
    assert parse_id_list("P1;P2;P3") == ["P1", "P2", "P3"]
    assert parse_id_list("  P1 ; P2 ") == ["P1", "P2"]
    assert parse_id_list("") == []
    assert parse_id_list(None) == []       # type: ignore[arg-type]
    assert parse_id_list(float("nan")) == []  # type: ignore[arg-type]


def test_load_persons_coerces_numeric_fields(raw):
    persons, report = load_persons(raw["persons"])
    assert report.ok
    assert pd.api.types.is_numeric_dtype(persons["prior_offences"])
    assert (persons["prior_offences"] >= 0).all()
