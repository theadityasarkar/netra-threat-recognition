"""Graph-building tests: structure invariants, feature matrix, ego/fragmentation."""

from __future__ import annotations

import math

import numpy as np

from src.graph.builder import (
    FEATURE_COLUMNS,
    build_graph,
    compute_node_features,
    ego_network,
    fragmentation_impact,
    graph_summary,
)


def test_every_person_becomes_a_node(canonical, graph):
    persons = canonical[0]
    assert graph.number_of_nodes() == len(persons)
    assert graph.number_of_edges() > 0


def test_edges_have_positive_weight(graph):
    for _, _, data in graph.edges(data=True):
        w = data.get("weight")
        assert w is not None and 0 < w <= 1.0 + 1e-9


def test_coaccused_inference_adds_edges(canonical):
    persons, rels, crimes, _documents, _report = canonical
    without = build_graph(persons, rels, crimes, infer_coaccused=False)
    with_ = build_graph(persons, rels, crimes, infer_coaccused=True)
    assert with_.number_of_edges() >= without.number_of_edges()


def test_node_feature_matrix(graph):
    feats = compute_node_features(graph)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert len(feats) == graph.number_of_nodes()
    arr = feats.to_numpy(dtype=float)
    assert not np.isnan(arr).any(), "feature matrix contains NaN"
    assert not np.isinf(arr).any(), "feature matrix contains inf"
    # Structural features must be non-negative.
    assert (feats["degree"] >= 0).all()
    assert (feats["weighted_degree"] >= 0).all()


def test_ego_network_bounds(graph):
    center = max(graph.degree, key=lambda kv: kv[1])[0]
    ego = ego_network(graph, center, radius=1)
    assert center in ego
    # A radius-1 ego net is the node plus its immediate neighbours.
    assert ego.number_of_nodes() == graph.degree(center) + 1


def test_fragmentation_impact_shape(graph):
    center = max(graph.degree, key=lambda kv: kv[1])[0]
    impact = fragmentation_impact(graph, center)
    assert isinstance(impact, dict)
    # Removing a node cannot create a negative number of new components.
    for key in ("new_fragments", "lcc_drop_pct"):
        if key in impact:
            assert impact[key] >= -1e-9


def test_graph_summary(graph):
    summary = graph_summary(graph)
    assert summary["nodes"] == graph.number_of_nodes()
    assert summary["edges"] == graph.number_of_edges()
    assert 0.0 <= summary.get("density", 0.0) <= 1.0
