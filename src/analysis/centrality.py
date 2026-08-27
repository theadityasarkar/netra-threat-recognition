"""
Layer 3a — centrality suite.

Six complementary measures plus k-core, because no single measure identifies a "key
player" on its own:

  • **degree / weighted degree** — visible reach, but easily inflated by chatter
  • **betweenness** — control of information and contraband flow (finds brokers whose
    degree is low; these are the arrests that actually fragment a network)
  • **closeness** — how fast this actor can reach the whole network
  • **eigenvector** — importance derived from *being connected to important people*
  • **PageRank** — random-walk influence, robust on directed/weighted evidence graphs
  • **k-core** — depth of embedding in the densest part of the organisation

Graphs above ``PERF["exact_centrality_max_nodes"]`` switch to pivot-sampled
betweenness so interaction latency stays under a second.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from src.config import PERF

CENTRALITY_COLUMNS = [
    "degree",
    "degree_centrality",
    "weighted_degree",
    "betweenness",
    "closeness",
    "eigenvector",
    "pagerank",
    "clustering",
    "core_number",
    "avg_neighbour_degree",
]


def compute_centralities(G: nx.Graph, exact: bool | None = None) -> pd.DataFrame:
    """Compute all centrality measures. Returns a frame indexed by ``person_id``."""
    n = G.number_of_nodes()
    if n == 0:
        return pd.DataFrame(columns=CENTRALITY_COLUMNS)

    if exact is None:
        exact = n <= PERF["exact_centrality_max_nodes"]

    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))
    degree_centrality = nx.degree_centrality(G)

    # Betweenness: 'weight' is treated as a *distance* by NetworkX, but our weight is
    # affinity — a strong tie is a SHORT path. Invert before use, otherwise strong ties
    # look expensive to traverse and the brokers come out wrong.
    dist = {(u, v): 1.0 / max(d["weight"], 1e-6) for u, v, d in G.edges(data=True)}
    nx.set_edge_attributes(G, dist, "distance")

    if exact:
        betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)
    else:
        k = min(PERF["betweenness_sample_k"], n)
        betweenness = nx.betweenness_centrality(G, k=k, weight="distance",
                                                normalized=True, seed=42)

    # Closeness on a disconnected graph must be computed per component, or every node
    # in a small component scores absurdly high.
    closeness: dict[str, float] = {}
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        if len(comp) == 1:
            closeness[next(iter(comp))] = 0.0
            continue
        local = nx.closeness_centrality(sub, distance="distance")
        # Scale by the component's share of the network so a 3-node clique cannot
        # outrank a well-placed actor inside the main organisation.
        scale = (len(comp) - 1) / (n - 1)
        closeness.update({k_: v * scale for k_, v in local.items()})

    try:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
        eigenvector = {k_: abs(float(v)) for k_, v in eigenvector.items()}
    except Exception:                                       # noqa: BLE001 — disconnected/degenerate
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6, weight="weight")
        except Exception:                                   # noqa: BLE001
            eigenvector = {node: 0.0 for node in G}

    pagerank = nx.pagerank(G, alpha=0.85, weight="weight", max_iter=200, tol=1e-8)
    clustering = nx.clustering(G, weight="weight")
    core_number = nx.core_number(G)

    try:
        avg_nbr_deg = nx.average_neighbor_degree(G)
    except Exception:                                       # noqa: BLE001
        avg_nbr_deg = {node: 0.0 for node in G}

    df = pd.DataFrame({
        "degree": pd.Series(degree, dtype=float),
        "degree_centrality": pd.Series(degree_centrality, dtype=float),
        "weighted_degree": pd.Series(weighted_degree, dtype=float),
        "betweenness": pd.Series(betweenness, dtype=float),
        "closeness": pd.Series(closeness, dtype=float),
        "eigenvector": pd.Series(eigenvector, dtype=float),
        "pagerank": pd.Series(pagerank, dtype=float),
        "clustering": pd.Series(clustering, dtype=float),
        "core_number": pd.Series(core_number, dtype=float),
        "avg_neighbour_degree": pd.Series(avg_nbr_deg, dtype=float),
    }).fillna(0.0)

    df.index.name = "person_id"
    df.attrs["exact"] = exact
    return df[CENTRALITY_COLUMNS]


def minmax(series: pd.Series) -> pd.Series:
    """Scale to 0–1. A flat series maps to zeros rather than dividing by zero."""
    lo, hi = float(series.min()), float(series.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def percentile_rank(series: pd.Series) -> pd.Series:
    """Percentile rank in 0–1 (ties averaged)."""
    if len(series) == 0:
        return series
    return series.rank(pct=True, method="average")


def centrality_correlations(cent: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between measures.

    Analytically useful: when betweenness and degree *disagree*, the network has
    brokers — the actors a headcount-based investigation would overlook.
    """
    cols = ["degree", "weighted_degree", "betweenness", "closeness",
            "eigenvector", "pagerank", "core_number"]
    present = [c for c in cols if c in cent.columns]
    if len(cent) < 3 or len(present) < 2:
        return pd.DataFrame()
    return cent[present].corr(method="spearman").round(3)
