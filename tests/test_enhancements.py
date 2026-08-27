"""Tests for new system enhancements:
- Multilingual / Hindi entity extraction
- Neo4j Cypher export
- GraphML export
- AI Executive Intelligence Brief generation
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import pytest
import pandas as pd

from src.ingestion.entities import extract_entities
from src.ingestion.sample_data import generate_sample_dataset
from src.pipeline import run_pipeline
from src.reporting.report import (
    build_cypher_export,
    build_graphml_export,
    generate_ai_intelligence_brief,
)


def test_hindi_entity_extraction():
    """Verify Hindi Devanagari entities (person, location, weapons, contraband, currency)."""
    text = (
        "विशेष सूचना के आधार पर आरोपी रमेश कुमार और सरगना विजय शर्मा को थाना कोटवाली में पकड़ा गया। "
        "तलाशी में 1 तमंचा, 12 जिंदा कारतूस, ₹ 75,000 नकद और 250 ग्राम चरस बरामद हुई। "
        "आरोपी वाहन DL 01 AB 1234 में सवार थे और मोबाइल 9876543210 का उपयोग कर रहे थे।"
    )
    mentions = extract_entities(text, doc_id="HINDI-DOC-1")
    types_found = {m.entity_type: m.value for m in mentions}

    assert "PERSON" in types_found, f"Expected PERSON, got {types_found}"
    assert "LOCATION" in types_found, f"Expected LOCATION, got {types_found}"
    assert "CURRENCY" in types_found, f"Expected CURRENCY, got {types_found}"
    assert "WEAPON" in types_found, f"Expected WEAPON, got {types_found}"
    assert "CONTRABAND" in types_found, f"Expected CONTRABAND, got {types_found}"
    assert "PHONE" in types_found, f"Expected PHONE, got {types_found}"
    assert "VEHICLE" in types_found, f"Expected VEHICLE, got {types_found}"

    # Verify values
    persons = [m.value for m in mentions if m.entity_type == "PERSON"]
    assert any("रमेश कुमार" in p for p in persons)
    assert any("विजय शर्मा" in p for p in persons)


@pytest.fixture(scope="module")
def analysis_result():
    data = generate_sample_dataset(n_persons=70, n_syndicates=3, seed=42)
    return run_pipeline(
        data["persons"],
        data["relationships"],
        data["crime_records"],
        documents=data["documents"],
        enable_gnn=True,
        gnn_epochs=50,
    )


def test_cypher_export(analysis_result):
    """Verify Neo4j Cypher queries are generated cleanly."""
    cql = build_cypher_export(analysis_result, case_name="Op Test")
    assert "CREATE CONSTRAINT" in cql
    assert "MERGE (p:Person" in cql
    assert "MATCH (s:Person" in cql
    assert "COMMUNICATION" in cql or "FINANCIAL" in cql or "CO_ACCUSED" in cql


def test_graphml_export(analysis_result):
    """Verify GraphML produces valid parseable XML with nodes and edges."""
    xml_str = build_graphml_export(analysis_result)
    root = ET.fromstring(xml_str)
    assert root.tag.endswith("graphml")
    graph_elem = root.find("{http://graphml.graphdrawing.org/xmlns}graph")
    assert graph_elem is not None
    nodes = graph_elem.findall("{http://graphml.graphdrawing.org/xmlns}node")
    edges = graph_elem.findall("{http://graphml.graphdrawing.org/xmlns}edge")
    assert len(nodes) > 0
    assert len(edges) > 0


def test_ai_intelligence_brief(analysis_result):
    """Verify AI Executive Intelligence Briefing structure."""
    brief = generate_ai_intelligence_brief(analysis_result, case_name="Op Sentinel")
    assert "MINISTRY OF HOME AFFAIRS" in brief
    assert "STRATEGIC THREAT LANDSCAPE" in brief
    assert "ACTIONABLE OPERATIONAL RECOMMENDATIONS" in brief
    assert "GraphSAGE GNN Link Validation" in brief
