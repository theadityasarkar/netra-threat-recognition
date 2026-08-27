"""
Layer 3d — key player detection.

The naive answer to *"who runs this network?"* is "whoever has the most contacts". That
answer is usually wrong: in a compartmentalised organisation the person with the most
contacts is a dispatcher, and the kingpin deliberately keeps a small, high-value
address book.

This module therefore does three things:

1. **Influence Index** — blends six normalised centrality measures with the weights in
   :data:`src.config.INFLUENCE_WEIGHTS`. Percentile ranks are used rather than raw
   min-max values so one runaway hub cannot flatten everyone else's score.
2. **Operational role inference** — reads each actor's centrality *profile* (not just
   its magnitude) to assign a role an investigator recognises.
3. **Disruption ranking** — simulates removing the top actors and reports the actual
   fragmentation caused, which is what turns a ranking into an arrest priority list.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from src.analysis.centrality import minmax, percentile_rank
from src.config import INFLUENCE_WEIGHTS, OPERATIONAL_ROLES
from src.graph.builder import fragmentation_impact


def compute_influence(
    G: nx.Graph,
    centralities: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Attach the Influence Index (0–100) and component percentiles to each actor."""
    if len(centralities) == 0:
        return pd.DataFrame()

    w = {**INFLUENCE_WEIGHTS, **(weights or {})}
    total = sum(w.values()) or 1.0
    w = {k: v / total for k, v in w.items()}

    df = centralities.copy()
    for metric in w:
        if metric not in df.columns:
            df[metric] = 0.0
        df[f"pct_{metric}"] = percentile_rank(df[metric])

    df["influence_raw"] = sum(w[m] * df[f"pct_{m}"] for m in w)
    # Rescale so the strongest actor in *this* dataset anchors at 100 — analysts
    # compare within a case file, not against an absolute national baseline.
    df["influence_score"] = (100.0 * minmax(df["influence_raw"])).round(2)
    df["influence_rank"] = df["influence_score"].rank(ascending=False, method="min").astype(int)
    return df


def infer_roles(G: nx.Graph, scored: pd.DataFrame) -> pd.Series:
    """Assign an operational role from each actor's centrality profile.

    The decision order matters — it encodes investigative doctrine:

      Kingpin      top-tier eigenvector *and* PageRank, dense core, high offence record
      Lieutenant   high PageRank and degree, but subordinate eigenvector
      Broker       betweenness far above degree — controls flow, few contacts
      Financier    financial links dominate the actor's edge composition
      Enforcer     embedded in the dense core, locally clustered, violent record
      Foot Soldier connected but peripheral in every measure
      Peripheral   degree ≤ 2 and bottom-quartile influence
    """
    if len(scored) == 0:
        return pd.Series(dtype=object)

    roles: dict[str, str] = {}

    p_eig = percentile_rank(scored["eigenvector"])
    p_pr = percentile_rank(scored["pagerank"])
    p_btw = percentile_rank(scored["betweenness"])
    p_deg = percentile_rank(scored["degree"])
    p_core = percentile_rank(scored["core_number"])
    p_clus = percentile_rank(scored["clustering"])

    for pid in scored.index:
        if pid not in G:
            roles[pid] = "Peripheral"
            continue

        attrs = G.nodes[pid]
        deg = G.degree(pid)
        priors = attrs.get("prior_offences", 0)

        edges = [G[pid][m] for m in G.neighbors(pid)]
        rel_types: list[str] = []
        for e in edges:
            rel_types.extend(e.get("relation_types") or [e.get("relation_type", "Associate")])
        n_rel = max(1, len(rel_types))
        fin_share = rel_types.count("Financial") / n_rel

        offences = " ".join(attrs.get("offence_list") or []).lower()
        violent = any(k in offences for k in
                      ("extortion", "kidnapping", "robbery", "arms", "trafficking"))

        eig, pr, btw, dg, core, clus = (p_eig[pid], p_pr[pid], p_btw[pid],
                                        p_deg[pid], p_core[pid], p_clus[pid])

        if deg <= 2 and scored.loc[pid, "influence_score"] < 25:
            role = "Peripheral"
        elif eig >= 0.94 and pr >= 0.90 and core >= 0.75 and priors >= 3:
            role = "Kingpin"
        elif fin_share >= 0.55 and deg >= 3:
            role = "Financier"
        elif btw - dg >= 0.22 and btw >= 0.80:
            # The signature broker: brokers flow far beyond what its degree suggests.
            role = "Broker"
        elif pr >= 0.85 and dg >= 0.80:
            role = "Lieutenant"
        elif core >= 0.80 and clus >= 0.60 and violent:
            role = "Enforcer"
        elif dg >= 0.35 or pr >= 0.35:
            role = "Foot Soldier"
        else:
            role = "Peripheral"

        roles[pid] = role

    return pd.Series(roles, name="role").reindex(scored.index).fillna("Peripheral")


def key_players_table(
    G: nx.Graph,
    centralities: pd.DataFrame,
    weights: dict[str, float] | None = None,
    top_n: int | None = None,
    with_fragmentation: int = 15,
) -> pd.DataFrame:
    """Full ranked key-player table, ready to render.

    Args:
        with_fragmentation: run the node-removal simulation for this many top actors
            (it is O(V+E) per node, so it is not run for the whole graph).
    """
    scored = compute_influence(G, centralities, weights)
    if len(scored) == 0:
        return pd.DataFrame()

    scored["role"] = infer_roles(G, scored)

    meta = pd.DataFrame({
        "name": {n: G.nodes[n].get("name", n) for n in scored.index if n in G},
        "alias": {n: "; ".join(G.nodes[n].get("alias_list") or []) for n in scored.index if n in G},
        "city": {n: G.nodes[n].get("city", "Unknown") for n in scored.index if n in G},
        "state": {n: G.nodes[n].get("state", "Unknown") for n in scored.index if n in G},
        "status": {n: G.nodes[n].get("status", "Active") for n in scored.index if n in G},
        "risk_category": {n: G.nodes[n].get("risk_category", "Low") for n in scored.index if n in G},
        "prior_offences": {n: G.nodes[n].get("prior_offences", 0) for n in scored.index if n in G},
        "offences": {n: "; ".join(G.nodes[n].get("offence_list") or []) for n in scored.index if n in G},
        "cases": {n: len(G.nodes[n].get("case_ids") or []) for n in scored.index if n in G},
    })

    out = meta.join(scored, how="right")
    out.index.name = "person_id"
    out = out.sort_values("influence_score", ascending=False)

    # Fragmentation simulation. Run it for the leaders by influence AND the top
    # betweenness actors — brokers and gateways have low influence but are exactly the
    # nodes whose removal fragments the network, so an influence-only shortlist would
    # never measure the arrests that matter most.
    frag_targets = set(out.head(max(0, with_fragmentation)).index)
    if with_fragmentation and "betweenness" in out.columns:
        frag_targets |= set(out.nlargest(max(0, with_fragmentation), "betweenness").index)

    frag_cols = {"new_fragments": {}, "lcc_drop_pct": {}, "isolated_created": {}}
    for pid in frag_targets:
        impact = fragmentation_impact(G, pid)
        frag_cols["new_fragments"][pid] = impact["new_fragments"]
        frag_cols["lcc_drop_pct"][pid] = round(impact["lcc_drop_pct"], 2)
        frag_cols["isolated_created"][pid] = impact["isolated_created"]

    for col, values in frag_cols.items():
        out[col] = pd.Series(values).reindex(out.index)

    # Disruption priority: influence tempered by *measured* structural damage. Signals
    # are normalised against the strongest observed impact in this dataset rather than
    # an absolute constant — in a resilient network the best available arrest may only
    # sever 3 % of the graph, and that arrest is still the right one to rank first.
    frag = out["new_fragments"].fillna(0).astype(float)
    lcc = out["lcc_drop_pct"].fillna(0).astype(float)
    frag_signal = frag / frag.max() if frag.max() > 0 else frag * 0.0
    lcc_signal = lcc / lcc.max() if lcc.max() > 0 else lcc * 0.0
    out["disruption_score"] = (
        100.0 * (0.45 * (out["influence_score"] / 100.0)
                 + 0.30 * lcc_signal
                 + 0.25 * frag_signal)
    ).round(2)

    if top_n:
        out = out.head(top_n)
    return out


def role_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Count and mean influence per operational role, in canonical order."""
    if len(table) == 0 or "role" not in table.columns:
        return pd.DataFrame()
    agg = (table.groupby("role")
           .agg(count=("role", "size"),
                mean_influence=("influence_score", "mean"),
                mean_degree=("degree", "mean"),
                mean_betweenness=("betweenness", "mean"))
           .round(3))
    order = [r for r in OPERATIONAL_ROLES if r in agg.index]
    return agg.reindex(order + [r for r in agg.index if r not in order])


def arrest_priority(table: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Actors ranked by disruption score — the operational shortlist.

    Actors already in custody are excluded: they cannot be arrested again, and leaving
    them in the queue is the fastest way to lose an analyst's trust in the tool.
    """
    if len(table) == 0:
        return pd.DataFrame()
    available = table[~table["status"].isin(["In Judicial Custody", "Deceased"])]
    cols = ["name", "role", "city", "state", "status", "risk_category",
            "influence_score", "disruption_score", "new_fragments", "lcc_drop_pct",
            "degree", "betweenness", "pagerank", "cases"]
    cols = [c for c in cols if c in available.columns]
    return (available.sort_values("disruption_score", ascending=False)
            .head(top_n)[cols])


def compare_metrics(table: pd.DataFrame, person_ids: list[str]) -> pd.DataFrame:
    """Normalised metric profiles for a handful of actors — feeds the radar chart."""
    metrics = ["degree", "weighted_degree", "betweenness", "closeness",
               "eigenvector", "pagerank", "core_number"]
    metrics = [m for m in metrics if m in table.columns]
    if not metrics or not person_ids:
        return pd.DataFrame()
    scaled = table[metrics].apply(minmax)
    return scaled.reindex([p for p in person_ids if p in scaled.index]).fillna(0.0)
