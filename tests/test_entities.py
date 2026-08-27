"""Tests for the NLP entity-extraction layer (``src/ingestion/entities.py``).

The extraction layer is the only place NETRA reads *unstructured* intelligence, so the
invariants that matter are: the India-specific patterns actually fire, one real-world
entity collapses to exactly one row no matter how it was written, every mention can be
traced back to the text that produced it, and a hostile document cannot crash the run.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import DOCUMENT_COLUMNS
from src.ingestion.entities import (
    extract_corpus,
    extract_entities,
    entity_key,
    front_business_candidates,
    mask_value,
    normalise_value,
)
from src.ingestion.loader import load_documents

# A single narrative carrying one instance of most supported entity types.
NARRATIVE = (
    "Complainant states that Aran Mehta, mobile 9876543210, transported four women "
    "from Siliguri to Mumbai in a Bolero bearing MH 02 AB 1234. A payment of "
    "Rs 4,50,000 was routed through account 123456789012 held at Kurla branch. "
    "Handset IMEI 356938035643809 was active at the tower. Rahat Travels Pvt Ltd "
    "is suspected of fronting the operation."
)


@pytest.fixture(scope="module")
def mentions():
    return extract_entities(NARRATIVE, doc_id="D0001")


def test_extraction_finds_core_entity_types(mentions):
    """The types an investigator would pull out by hand must all fire."""
    found = {m.entity_type for m in mentions}
    for expected in ("PHONE", "VEHICLE", "LOCATION", "BANK_ACCOUNT", "IMEI"):
        assert expected in found, f"{expected} not extracted; got {sorted(found)}"


def test_every_mention_is_traceable_to_its_source_span(mentions):
    """A lead an officer cannot verify against the text is worthless."""
    assert mentions, "no mentions extracted from the reference narrative"
    for m in mentions:
        assert 0 <= m.start < m.end <= len(NARRATIVE)
        # The recorded span must actually contain the raw string it claims.
        assert m.raw.strip() in NARRATIVE[m.start:m.end] or NARRATIVE[m.start:m.end] in m.raw
        assert m.raw.strip(), "empty raw mention"
        assert 0.0 < m.confidence <= 1.0
        assert m.context, "mention carries no verification context"
        assert m.doc_id == "D0001"


def test_spans_do_not_overlap(mentions):
    """Span arbitration must leave one winner per region of text — overlapping claims
    would double-count the same characters as two different entities."""
    spans = sorted((m.start, m.end) for m in mentions)
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start >= prev_end, f"overlapping spans at {next_start} < {prev_end}"


@pytest.mark.parametrize("etype,variants", [
    ("PHONE", ("9876543210", "+91 98765 43210", "098765-43210")),
    ("VEHICLE", ("MH 02 AB 1234", "MH02AB1234", "mh-02-ab-1234")),
])
def test_normalisation_collapses_formatting_variants(etype, variants):
    """The same real-world entity written three ways is still one entity — otherwise
    the register inflates and a phone shared by two suspects looks like two phones."""
    keys = {entity_key(etype, normalise_value(etype, v)) for v in variants}
    assert len(keys) == 1, f"{variants} produced {len(keys)} distinct keys: {keys}"


def test_masking_hides_the_body_but_keeps_the_tail():
    """Redaction has to stay useful: an officer must still be able to disambiguate two
    masked numbers, so the trailing digits survive."""
    masked = mask_value("PHONE", "9876543210")
    assert "9876543210" not in masked
    assert masked.endswith("210")


def test_corpus_extraction_aggregates_and_counts():
    """Two documents naming the same phone must yield one entity seen in two docs."""
    docs = {
        "D1": "Suspect contacted 9876543210 before the transfer.",
        "D2": "The same number 98765 43210 appears in the second intercept.",
        "D3": "No entities of interest in this line.",
    }
    res = extract_corpus(docs)

    assert res.n_entities >= 1
    assert res.n_mentions >= 2
    phones = res.entities[res.entities["entity_type"] == "PHONE"]
    assert len(phones) == 1, "the same phone written two ways became two entities"
    assert int(phones.iloc[0]["n_documents"]) == 2
    assert int(phones.iloc[0]["n_mentions"]) == 2
    # Provenance is preserved per entity.
    assert set(res.mentions["doc_id"]) <= {"D1", "D2", "D3"}


def test_corpus_extraction_accepts_a_dataframe():
    frame = pd.DataFrame({
        "doc_id": ["D0001", "D0002"],
        "text": [NARRATIVE, "Vehicle KA 05 MN 9090 seen at Kurla with 9812345670."],
    })
    res = extract_corpus(frame)
    assert res.n_entities > 0
    assert set(res.documents["doc_id"]) == {"D0001", "D0002"}
    # The co-occurrence link table is the bridge to the graph layer.
    assert {"source_id", "target_id"} <= set(res.links.columns) or res.links.empty


def test_extraction_survives_degenerate_input():
    """Empty, whitespace and punctuation-only documents must not raise — a corpus is
    agency data and will contain junk rows."""
    for junk in ("", "   ", "\n\t", "-----", "??? !!!", "123"):
        assert isinstance(extract_entities(junk, doc_id="X"), list)

    res = extract_corpus({"D1": "", "D2": "   "})
    assert res.n_entities == 0
    assert res.n_mentions == 0


def test_front_business_candidates_returns_a_frame():
    res = extract_corpus({"D1": NARRATIVE})
    candidates = front_business_candidates(res)
    assert isinstance(candidates, pd.DataFrame)


def test_load_documents_requires_only_a_text_column(tmp_path):
    """Provenance columns are optional; the loader backfills them and synthesises IDs."""
    csv = tmp_path / "docs.csv"
    csv.write_text("text\nFirst intercept.\nSecond intercept.\n", encoding="utf-8")

    df, report = load_documents(str(csv))

    assert not report.errors
    assert list(df.columns) == DOCUMENT_COLUMNS
    assert len(df) == 2
    assert df["doc_id"].is_unique and df["doc_id"].iloc[0]


def test_load_documents_drops_blank_text_with_a_warning(tmp_path):
    csv = tmp_path / "docs.csv"
    csv.write_text("doc_id,text\nD1,Real content here.\nD2,\nD3,   \n", encoding="utf-8")

    df, report = load_documents(str(csv))

    assert len(df) == 1
    assert df["doc_id"].iloc[0] == "D1"
    assert report.warnings, "silently dropping rows would hide data loss"


def test_load_documents_adopts_an_unnamed_content_column(tmp_path):
    """A pasted single-column export has no `text` header — infer it rather than reject."""
    csv = tmp_path / "docs.csv"
    csv.write_text("narrative\nSuspect seen at the docks.\n", encoding="utf-8")

    df, report = load_documents(str(csv))

    assert not report.errors
    assert len(df) == 1
    assert df["text"].iloc[0] == "Suspect seen at the docks."
