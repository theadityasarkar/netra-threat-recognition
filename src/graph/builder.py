"""
Layer 2 — NetworkX graph construction and node feature engineering.

Two things happen here:

1. **Multi-relational graph assembly.** Recorded links become weighted edges, and
   co-accused edges are *inferred* from case files — if four people appear on the same
   FIR, that is six links no analyst had to enter by hand. This inference is typically
   what makes an otherwise sparse dataset analysable.

2. **Node feature matrix.** 24 numeric features per actor, spanning link structure,
   relation composition, evidentiary diversity, record history and temporal behaviour.
   The same matrix feeds the GNN encoder and the Isolation Forest, so an anomaly found
   by one is explainable in the same vocabulary as the other.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from src.config import RELATION_WEIGHTS

# Canonical feature order — the GNN and Isolation Forest both rely on this.
FEATURE_COLUMNS = [
    "degree",
    "weighted_degree",
    "avg_edge_strength",
    "max_edge_strength",
    "log_interactions",
    "avg_interactions",
    "relation_diversity",
    "share_financial",
    "share_communication",
    "share_coaccused",
    "share_kinship",
    "share_handler",
    "evidence_diversity",
    "prior_offences",
    "offence_type_count",
    "alias_count",
    "age",
    "case_count",
    "max_case_severity",
    "mean_case_severity",
    "dormancy_years",
    "neighbour_mean_priors",
    "neighbour_mean_degree",
    "geo_spread",
]

_KINSHIP = {"Family", "Shelter Provider"}
_HANDLER = {"Handler", "Courier Link"}


def _edge_weight(relation_type: str, strength: float) -> float:
    """Evidentiary weight of a link: tie strength scaled by relation reliability."""
    return float(np.clip(strength, 0.01, 1.0) * RELATION_WEIGHTS.get(relation_type, 0.5))


def build_graph(
    persons: pd.DataFrame,
    relationships: pd.DataFrame,
    crimes: pd.DataFrame | None = None,
    infer_coaccused: bool = True,
    max_case_size: int = 12,
) -> nx.Graph:
    """Assemble the criminal network as an undirected weighted NetworkX graph.

    Args:
        persons: canonical persons table.
        relationships: canonical relationships table.
        crimes: optional case table with an ``_ids`` or ``persons_involved`` column.
        infer_coaccused: add edges between co-accused on the same case.
        max_case_size: skip inference for cases with more accused than this — a
            50-person round-up would otherwise inject a meaningless 1,225-edge clique.

    Returns:
        ``nx.Graph`` with rich node and edge attributes.
    """
    G = nx.Graph()

    # ── Nodes ─────────────────────────────────────────────────────────────────
    for rec in persons.to_dict("records"):
        pid = str(rec["person_id"])
        aliases = [a.strip() for a in str(rec.get("alias") or "").split(";") if a.strip()]
        offences = [o.strip() for o in str(rec.get("offence_types") or "").split(";") if o.strip()]
        last = rec.get("last_activity")
        G.add_node(
            pid,
            person_id=pid,
            name=rec.get("name", pid),
            alias_list=aliases,
            alias_count=len(aliases),
            age=int(rec.get("age") or 0),
            gender=rec.get("gender", "U"),
            city=rec.get("city", "Unknown"),
            state=rec.get("state", "Unknown"),
            occupation=rec.get("occupation", "Unknown"),
            prior_offences=int(rec.get("prior_offences") or 0),
            offence_list=offences,
            first_arrest_year=int(rec.get("first_arrest_year") or 0),
            last_activity=last if pd.notna(last) else None,
            status=rec.get("status", "Active"),
            risk_category=rec.get("risk_category", "Low"),
            case_ids=[],
            case_severities=[],
        )

    # ── Recorded edges ────────────────────────────────────────────────────────
    for rec in relationships.to_dict("records"):
        s, t = str(rec["source_id"]), str(rec["target_id"])
        if s == t or s not in G or t not in G:
            continue
        rel = rec.get("relation_type", "Associate")
        strength = float(rec.get("strength") or 0.5)
        G.add_edge(
            s, t,
            relation_type=rel,
            relation_types=[rel],
            strength=strength,
            weight=_edge_weight(rel, strength),
            interaction_count=int(rec.get("interaction_count") or 1),
            first_seen=rec.get("first_seen"),
            last_seen=rec.get("last_seen"),
            evidence_source=rec.get("evidence_source", "Unspecified"),
            evidence_sources=[rec.get("evidence_source", "Unspecified")],
            location=rec.get("location", "Unknown"),
            inferred=False,
            case_ids=[],
        )

    # ── Inferred co-accused edges ─────────────────────────────────────────────
    inferred_added = 0
    if infer_coaccused and crimes is not None and len(crimes):
        id_col = "_ids" if "_ids" in crimes.columns else "persons_involved"
        for rec in crimes.to_dict("records"):
            ids = rec.get(id_col)
            if isinstance(ids, str):
                ids = [i.strip() for i in ids.replace(",", ";").split(";") if i.strip()]
            ids = [i for i in (ids or []) if i in G]
            if len(ids) < 1 or len(ids) > max_case_size:
                continue

            case_id = rec.get("case_id", "")
            severity = int(rec.get("severity") or 3)
            for pid in ids:
                G.nodes[pid]["case_ids"].append(case_id)
                G.nodes[pid]["case_severities"].append(severity)

            # Co-accused strength rises with case severity and falls as the group grows:
            # two people on a murder FIR is a stronger signal than eight on a raid.
            co_strength = float(np.clip(0.45 + 0.09 * severity - 0.03 * len(ids), 0.2, 0.98))
            for a, b in combinations(sorted(ids), 2):
                if G.has_edge(a, b):
                    e = G[a][b]
                    e["case_ids"].append(case_id)
                    if "Co-Accused" not in e["relation_types"]:
                        e["relation_types"].append("Co-Accused")
                    # Corroboration across sources strengthens an existing tie.
                    e["strength"] = float(min(1.0, e["strength"] + 0.06))
                    e["weight"] = max(e["weight"], _edge_weight("Co-Accused", co_strength))
                else:
                    G.add_edge(
                        a, b,
                        relation_type="Co-Accused",
                        relation_types=["Co-Accused"],
                        strength=co_strength,
                        weight=_edge_weight("Co-Accused", co_strength),
                        interaction_count=1,
                        first_seen=rec.get("date"),
                        last_seen=rec.get("date"),
                        evidence_source="FIR Linkage",
                        evidence_sources=["FIR Linkage"],
                        location=rec.get("city", "Unknown"),
                        inferred=True,
                        case_ids=[case_id],
                    )
                    inferred_added += 1

    G.graph["inferred_edges"] = inferred_added
    G.graph["recorded_edges"] = G.number_of_edges() - inferred_added
    return G


def compute_node_features(G: nx.Graph, reference_year: int = 2026) -> pd.DataFrame:
    """Build the 24-column node feature matrix indexed by ``person_id``."""
    rows: list[dict] = []
    degrees = dict(G.degree())

    for n, attrs in G.nodes(data=True):
        nbrs = list(G.neighbors(n))
        edges = [G[n][m] for m in nbrs]

        strengths = np.array([e["strength"] for e in edges]) if edges else np.array([0.0])
        weights = np.array([e["weight"] for e in edges]) if edges else np.array([0.0])
        interactions = np.array([e["interaction_count"] for e in edges]) if edges else np.array([0.0])

        rel_types: list[str] = []
        for e in edges:
            rel_types.extend(e.get("relation_types") or [e.get("relation_type", "Associate")])
        n_rel = max(1, len(rel_types))

        ev_sources = {e.get("evidence_source", "Unspecified") for e in edges}

        sev = attrs.get("case_severities") or []
        last = attrs.get("last_activity")
        if last is not None and pd.notna(last):
            dormancy = max(0.0, reference_year - float(pd.Timestamp(last).year))
        else:
            dormancy = 8.0                                   # unknown ⇒ treat as stale

        nbr_priors = [G.nodes[m].get("prior_offences", 0) for m in nbrs]
        nbr_degrees = [degrees[m] for m in nbrs]
        nbr_cities = {G.nodes[m].get("city", "Unknown") for m in nbrs}

        rows.append({
            "person_id": n,
            "degree": float(degrees[n]),
            "weighted_degree": float(weights.sum()),
            "avg_edge_strength": float(strengths.mean()),
            "max_edge_strength": float(strengths.max()),
            "log_interactions": float(np.log1p(interactions.sum())),
            "avg_interactions": float(interactions.mean()),
            "relation_diversity": float(len(set(rel_types))),
            "share_financial": rel_types.count("Financial") / n_rel,
            "share_communication": rel_types.count("Communication") / n_rel,
            "share_coaccused": rel_types.count("Co-Accused") / n_rel,
            "share_kinship": sum(r in _KINSHIP for r in rel_types) / n_rel,
            "share_handler": sum(r in _HANDLER for r in rel_types) / n_rel,
            "evidence_diversity": float(len(ev_sources)),
            "prior_offences": float(attrs.get("prior_offences", 0)),
            "offence_type_count": float(len(attrs.get("offence_list") or [])),
            "alias_count": float(attrs.get("alias_count", 0)),
            "age": float(attrs.get("age", 0)),
            "case_count": float(len(attrs.get("case_ids") or [])),
            "max_case_severity": float(max(sev)) if sev else 0.0,
            "mean_case_severity": float(np.mean(sev)) if sev else 0.0,
            "dormancy_years": float(dormancy),
            "neighbour_mean_priors": float(np.mean(nbr_priors)) if nbr_priors else 0.0,
            "neighbour_mean_degree": float(np.mean(nbr_degrees)) if nbr_degrees else 0.0,
            "geo_spread": float(len(nbr_cities)),
        })

    df = pd.DataFrame(rows).set_index("person_id")
    return df[FEATURE_COLUMNS]


def graph_summary(G: nx.Graph) -> dict:
    """Headline structural statistics for the overview KPIs."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    if n == 0:
        return {"nodes": 0, "edges": 0, "density": 0.0, "components": 0,
                "largest_component": 0, "avg_degree": 0.0, "isolates": 0,
                "avg_clustering": 0.0, "diameter_lcc": 0, "assortativity": 0.0,
                "inferred_edges": 0, "recorded_edges": 0}

    components = list(nx.connected_components(G))
    lcc = max(components, key=len) if components else set()

    # Diameter is O(V·E); only measure it when the component is small enough to matter.
    diameter = 0
    if 1 < len(lcc) <= 800:
        try:
            diameter = nx.diameter(G.subgraph(lcc))
        except Exception:                                   # noqa: BLE001
            diameter = 0

    try:
        assortativity = float(nx.degree_assortativity_coefficient(G))
        if not np.isfinite(assortativity):
            assortativity = 0.0
    except Exception:                                       # noqa: BLE001
        assortativity = 0.0

    return {
        "nodes": n,
        "edges": m,
        "density": float(nx.density(G)),
        "components": len(components),
        "largest_component": len(lcc),
        "avg_degree": 2 * m / n,
        "isolates": sum(1 for _, d in G.degree() if d == 0),
        "avg_clustering": float(nx.average_clustering(G)) if m else 0.0,
        "diameter_lcc": diameter,
        "assortativity": assortativity,
        "inferred_edges": G.graph.get("inferred_edges", 0),
        "recorded_edges": G.graph.get("recorded_edges", m),
    }


def ego_network(G: nx.Graph, center: str, radius: int = 1) -> nx.Graph:
    """Neighbourhood subgraph around one actor, for drill-down views."""
    if center not in G:
        return nx.Graph()
    return nx.ego_graph(G, center, radius=radius, undirected=True)


def fragmentation_impact(G: nx.Graph, node: str) -> dict:
    """Simulate removing one actor: how much does the network break apart?

    Returns the change in component count and the largest-component share, which is
    the quantitative answer to "who do we arrest first to disrupt this network?".
    """
    if node not in G or G.number_of_nodes() < 2:
        return {"components_before": 0, "components_after": 0, "new_fragments": 0,
                "lcc_before": 0, "lcc_after": 0, "lcc_drop_pct": 0.0, "isolated_created": 0}

    before_components = list(nx.connected_components(G))
    lcc_before = max((len(c) for c in before_components), default=0)

    H = G.copy()
    H.remove_node(node)
    after_components = list(nx.connected_components(H))
    lcc_after = max((len(c) for c in after_components), default=0)

    return {
        "components_before": len(before_components),
        "components_after": len(after_components),
        "new_fragments": max(0, len(after_components) - len(before_components)),
        "lcc_before": lcc_before,
        "lcc_after": lcc_after,
        "lcc_drop_pct": (100.0 * (lcc_before - lcc_after) / lcc_before) if lcc_before else 0.0,
        "isolated_created": sum(1 for _, d in H.degree() if d == 0)
                            - sum(1 for _, d in G.degree() if d == 0),
    }
