"""
Layer 3b — community detection and suspicious-cluster scoring.

Louvain modularity partitions the graph into *operational modules*. On top of the
partition, two products matter operationally:

  • **Cluster threat scoring** — a module is dangerous less because of its size than
    because of its cohesion, offence severity, geographic reach and insulation.
  • **Bridge/broker detection** — edges and actors that hold otherwise separate modules
    together. Cutting these is what actually fragments a network.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from src.config import CRIME_SEVERITY, RISK_CATEGORIES

_RISK_SCORE = {"Critical": 1.0, "High": 0.72, "Medium": 0.42, "Low": 0.15}


def detect_communities(G: nx.Graph, resolution: float = 1.0, seed: int = 42) -> dict[str, int]:
    """Partition the graph into modules. Returns ``{person_id: community_id}``.

    Louvain is preferred (fast, weighted, high modularity); greedy modularity and then
    connected components are fallbacks so the pipeline never dies on an odd graph.
    """
    if G.number_of_nodes() == 0:
        return {}
    if G.number_of_edges() == 0:
        return {n: i for i, n in enumerate(G.nodes())}

    try:
        communities = nx.community.louvain_communities(
            G, weight="weight", resolution=resolution, seed=seed)
    except Exception:                                       # noqa: BLE001
        try:
            communities = list(nx.community.greedy_modularity_communities(G, weight="weight"))
        except Exception:                                   # noqa: BLE001
            communities = list(nx.connected_components(G))

    # Largest module gets id 0 — stable, readable colour assignment in the UI.
    communities = sorted(communities, key=len, reverse=True)
    return {node: cid for cid, group in enumerate(communities) for node in group}


def modularity_score(G: nx.Graph, partition: dict[str, int]) -> float:
    """Newman modularity Q of the partition (higher = crisper module structure)."""
    if not partition or G.number_of_edges() == 0:
        return 0.0
    groups: dict[int, set] = {}
    for node, cid in partition.items():
        groups.setdefault(cid, set()).add(node)
    try:
        return float(nx.community.modularity(G, list(groups.values()), weight="weight"))
    except Exception:                                       # noqa: BLE001
        return 0.0


def community_profiles(
    G: nx.Graph,
    partition: dict[str, int],
    centralities: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per module, with structure, severity, geography and a threat score."""
    if not partition:
        return pd.DataFrame()

    groups: dict[int, list[str]] = {}
    for node, cid in partition.items():
        groups.setdefault(cid, []).append(node)

    rows: list[dict] = []
    for cid, members in sorted(groups.items()):
        sub = G.subgraph(members)
        size = len(members)
        internal = sub.number_of_edges()
        external = sum(
            1 for u in members for v in G.neighbors(u) if partition.get(v, cid) != cid)

        attrs = [G.nodes[m] for m in members]
        priors = [a.get("prior_offences", 0) for a in attrs]
        risks = [a.get("risk_category", "Low") for a in attrs]
        cities = {a.get("city", "Unknown") for a in attrs}
        states = {a.get("state", "Unknown") for a in attrs}

        offences: list[str] = []
        severities: list[int] = []
        for a in attrs:
            offences.extend(a.get("offence_list") or [])
            severities.extend(a.get("case_severities") or [])
        if not severities and offences:
            severities = [CRIME_SEVERITY.get(o, 3) for o in offences]

        density = (2 * internal / (size * (size - 1))) if size > 1 else 0.0
        # Conductance-style insulation: what share of the module's links stay inside?
        insulation = internal / max(1, internal + external)

        if centralities is not None and len(centralities):
            member_cent = centralities.reindex(members).fillna(0.0)
            lead = member_cent["pagerank"].idxmax() if len(member_cent) else None
            lead_name = G.nodes[lead]["name"] if lead in G.nodes else "—"
            top_between = float(member_cent["betweenness"].max())
        else:
            lead, lead_name, top_between = None, "—", 0.0

        dominant_offence = (pd.Series(offences).value_counts().idxmax()
                            if offences else "Unclassified")
        risk_mix = pd.Series(risks).value_counts()

        # Threat score: cohesion + severity + reach + record depth + insulation.
        # Weights sum to 1.0; each term is already normalised to 0–1.
        threat = (
            0.24 * density
            + 0.26 * (float(np.mean(severities)) / 5.0 if severities else 0.4)
            + 0.16 * min(1.0, len(states) / 5.0)
            + 0.16 * min(1.0, float(np.mean(priors)) / 8.0 if priors else 0.0)
            + 0.10 * insulation
            + 0.08 * min(1.0, size / 40.0)
        )

        rows.append({
            "community_id": cid,
            "label": f"Module {cid + 1}",
            "size": size,
            "internal_edges": internal,
            "external_edges": external,
            "density": round(density, 4),
            "insulation": round(insulation, 3),
            "mean_prior_offences": round(float(np.mean(priors)), 2) if priors else 0.0,
            "mean_severity": round(float(np.mean(severities)), 2) if severities else 0.0,
            "max_severity": int(max(severities)) if severities else 0,
            "cities": len(cities),
            "states": len(states),
            "state_list": ", ".join(sorted(states)[:4]) + ("…" if len(states) > 4 else ""),
            "dominant_offence": dominant_offence,
            "critical_members": int(risk_mix.get("Critical", 0)),
            "high_members": int(risk_mix.get("High", 0)),
            "lead_actor_id": lead,
            "lead_actor": lead_name,
            "top_betweenness": round(top_between, 5),
            "threat_score": round(100.0 * float(np.clip(threat, 0, 1)), 1),
        })

    df = pd.DataFrame(rows).sort_values("threat_score", ascending=False)
    return df.reset_index(drop=True)


def bridge_analysis(G: nx.Graph, partition: dict[str, int]) -> pd.DataFrame:
    """Actors linking multiple modules, ranked by how much they hold the network together."""
    if not partition:
        return pd.DataFrame()

    rows: list[dict] = []
    for n in G.nodes():
        own = partition.get(n)
        nbr_modules = {partition.get(m) for m in G.neighbors(n)} - {None}
        external = nbr_modules - {own}
        if not external:
            continue
        ext_edges = sum(1 for m in G.neighbors(n) if partition.get(m) != own)
        deg = G.degree(n)
        rows.append({
            "person_id": n,
            "name": G.nodes[n]["name"],
            "own_module": own,
            "modules_touched": len(external) + 1,
            "external_links": ext_edges,
            "total_links": deg,
            # A pure broker's links are almost all external — that ratio is the signal.
            "external_ratio": round(ext_edges / max(1, deg), 3),
            "risk_category": G.nodes[n].get("risk_category", "Low"),
            "city": G.nodes[n].get("city", "Unknown"),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["bridge_score"] = (
        0.5 * (df["modules_touched"] / max(1, df["modules_touched"].max()))
        + 0.3 * df["external_ratio"]
        + 0.2 * (df["external_links"] / max(1, df["external_links"].max()))
    ).round(4)
    return df.sort_values("bridge_score", ascending=False).reset_index(drop=True)


def critical_edges(G: nx.Graph, partition: dict[str, int], top_n: int = 15) -> pd.DataFrame:
    """Inter-module links whose removal would most disrupt connectivity.

    Ranked by edge betweenness restricted to cross-module edges — these are the
    surveillance targets that reveal how two organisations coordinate.
    """
    if G.number_of_edges() == 0:
        return pd.DataFrame()

    n = G.number_of_nodes()
    k = None if n <= 800 else min(300, n)
    try:
        eb = nx.edge_betweenness_centrality(G, k=k, weight="distance"
                                            if "distance" in next(iter(G.edges(data=True)))[2]
                                            else None,
                                            normalized=True, seed=42)
    except Exception:                                       # noqa: BLE001
        eb = nx.edge_betweenness_centrality(G, normalized=True)

    bridges = set()
    try:
        bridges = {tuple(sorted(e)) for e in nx.bridges(G)}
    except Exception:                                       # noqa: BLE001
        pass

    rows = []
    for (u, v), score in eb.items():
        cu, cv = partition.get(u), partition.get(v)
        if cu == cv:
            continue
        e = G[u][v]
        rows.append({
            "source_id": u,
            "source": G.nodes[u]["name"],
            "target_id": v,
            "target": G.nodes[v]["name"],
            "module_a": cu,
            "module_b": cv,
            "relation": ", ".join(e.get("relation_types") or [e.get("relation_type", "—")]),
            "strength": e.get("strength", 0.0),
            "evidence": e.get("evidence_source", "—"),
            "edge_betweenness": round(float(score), 6),
            "is_cut_edge": tuple(sorted((u, v))) in bridges,
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(
        ["is_cut_edge", "edge_betweenness"], ascending=[False, False])
    return df.head(top_n).reset_index(drop=True)


def risk_distribution(G: nx.Graph) -> pd.Series:
    """Count of actors per risk band, in canonical order."""
    counts = pd.Series(
        [G.nodes[n].get("risk_category", "Low") for n in G.nodes()]).value_counts()
    return counts.reindex(RISK_CATEGORIES).fillna(0).astype(int)
