"""Analysis-layer tests: centrality bounds, communities, GNN convergence,
anomaly-score determinism and key-player role assignment."""

from __future__ import annotations

import numpy as np

from src.analysis.centrality import compute_centralities
from src.analysis.communities import detect_communities, modularity_score
from src.config import OPERATIONAL_ROLES
from src.pipeline import run_pipeline


# ── Centrality ──────────────────────────────────────────────────────────────────

def test_centrality_bounds(graph):
    cent = compute_centralities(graph)
    assert len(cent) == graph.number_of_nodes()
    assert not cent.isna().any().any(), "centrality frame contains NaN"

    # PageRank is a probability distribution over nodes.
    assert abs(cent["pagerank"].sum() - 1.0) < 1e-3

    # Raw degree is a non-negative count bounded by n-1.
    assert (cent["degree"] >= 0).all()
    assert cent["degree"].max() <= graph.number_of_nodes() - 1

    # Normalised centralities live in [0, 1].
    for col in ("degree_centrality", "betweenness", "closeness", "eigenvector"):
        s = cent[col]
        assert s.min() >= -1e-6, f"{col} has negative values"
        assert s.max() <= 1.0 + 1e-6, f"{col} exceeds 1.0"


# ── Communities ─────────────────────────────────────────────────────────────────

def test_community_detection_covers_all_nodes(graph):
    part = detect_communities(graph, resolution=1.0, seed=42)
    assert set(part.keys()) == set(graph.nodes())
    assert len(set(part.values())) >= 2  # the planted syndicates must separate


def test_modularity_is_meaningful(graph):
    part = detect_communities(graph, resolution=1.0, seed=42)
    q = modularity_score(graph, part)
    # Structured syndicate data should partition well above random (~0).
    assert 0.2 < q <= 1.0


def test_community_detection_is_deterministic(graph):
    a = detect_communities(graph, resolution=1.0, seed=42)
    b = detect_communities(graph, resolution=1.0, seed=42)
    assert a == b


# ── GNN ──────────────────────────────────────────────────────────────────────────

def test_gnn_trains_and_converges(result):
    gnn = result.gnn
    n = result.n_actors
    assert gnn.embeddings.shape[0] == n
    assert gnn.embeddings.shape[1] == 32
    assert gnn.coords_2d.shape == (n, 2)
    # Held-out link-prediction AUC well above chance = the encoder learnt structure.
    assert gnn.auc > 0.7
    # Loss decreased over training.
    assert gnn.loss_history[-1] < gnn.loss_history[0]
    # Residuals are normalised anomaly signals.
    assert gnn.residuals.min() >= -1e-6
    assert gnn.residuals.max() <= 1.0 + 1e-6


# ── Determinism (GNN + anomaly scores) ───────────────────────────────────────────

def test_pipeline_is_deterministic(raw):
    kw = dict(gnn_epochs=80)
    r1 = run_pipeline(raw["persons"], raw["relationships"], raw["crime_records"], **kw)
    r2 = run_pipeline(raw["persons"], raw["relationships"], raw["crime_records"], **kw)

    # Same seed → identical embeddings.
    assert np.allclose(r1.gnn.embeddings, r2.gnn.embeddings, atol=1e-8)

    # Same seed → identical suspicion scores (Isolation Forest is seeded).
    s1 = r1.anomalies["suspicion_score"].sort_index()
    s2 = r2.anomalies["suspicion_score"].sort_index()
    assert s1.index.equals(s2.index)
    assert np.allclose(s1.to_numpy(), s2.to_numpy(), atol=1e-6)


# ── Anomaly outputs ───────────────────────────────────────────────────────────────

def test_anomaly_scores_and_explanations(result):
    a = result.anomalies
    assert (a["suspicion_score"] >= 0).all()
    assert (a["suspicion_score"] <= 100).all()
    assert set(a["band"].unique()).issubset({"Critical", "High", "Medium", "Low"})

    # Explainability contract: every flagged actor carries human-readable reasons.
    flagged = a[a["band"].isin(["Critical", "High", "Medium"])]
    for reasons in flagged["reasons"]:
        assert isinstance(reasons, list) and len(reasons) > 0


# ── Key players ───────────────────────────────────────────────────────────────────

def test_key_players_scored_and_roled(result):
    kp = result.key_players
    assert "influence_score" in kp.columns and "role" in kp.columns
    assert (kp["influence_score"] >= 0).all()
    assert set(kp["role"].unique()).issubset(set(OPERATIONAL_ROLES))


def test_arrest_priority_excludes_incarcerated(result):
    pri = result.priority
    if len(pri) and "status" in pri.columns:
        assert "In Judicial Custody" not in set(pri["status"])
        assert "Deceased" not in set(pri["status"])
    # Priority is a focused shortlist, not the whole roster.
    assert len(pri) <= len(result.key_players)
