"""
Layer 3e — hybrid anomaly and suspicious-cluster detection.

Three detectors vote, because none is sufficient on its own:

======================  ======  ==========================================================
Component               Weight  What it catches
======================  ======  ==========================================================
GNN residual            30 %    actors whose link pattern defies the network's learnt
                                grammar — the model cannot reconstruct their neighbourhood
Isolation Forest        30 %    multivariate statistical outliers over 24 structural
                                features; catches odd *combinations*, not odd single values
Expert rule engine      40 %    eight investigator-doctrine rules, each with an explicit
                                human-readable justification
======================  ======  ==========================================================

Every score ships with the specific reasons that produced it. An unexplained score is
useless to an officer who has to justify a surveillance request, so ``reasons`` is a
first-class output, not a debug field.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from src.analysis.centrality import percentile_rank
from src.config import (
    ANOMALY_MODEL_GATE_FLOOR,
    ANOMALY_WEIGHTS,
    RULE_PARAMS,
    RULE_SATURATION,
    SUSPICION_BANDS,
)

RULE_CATALOGUE = {
    "cross_module_broker": {
        "label": "Cross-Module Broker",
        "weight": 1.00,
        "why": "Controls flow between otherwise separate organisations despite few contacts.",
    },
    "structural_bridge": {
        "label": "Multi-Module Connector",
        "weight": 0.85,
        "why": "Maintains active links into three or more distinct modules.",
    },
    "financial_mule": {
        "label": "Suspected Financial Mule",
        "weight": 0.90,
        "why": "Link profile is dominated by financial transfers despite a near-clean record.",
    },
    "isolated_high_severity": {
        "label": "Insulated High-Severity Actor",
        "weight": 0.95,
        "why": "Very serious offences on record but almost no recorded contacts — "
               "suggests unmapped associates rather than a lone offender.",
    },
    "identity_evasion": {
        "label": "Identity Evasion Pattern",
        "weight": 0.60,
        "why": "Unusually high number of recorded aliases.",
    },
    "dormancy_spike": {
        "label": "Reactivation After Dormancy",
        "weight": 0.75,
        "why": "Long inactive period followed by a burst of fresh, high-volume contacts.",
    },
    "hierarchy_inversion": {
        "label": "Hierarchy Inversion",
        "weight": 0.70,
        "why": "Sits structurally above contacts far more senior than their own record "
               "suggests — possible unrecorded seniority or a false identity.",
    },
    "structural_gatekeeper": {
        "label": "Structural Gatekeeper",
        "weight": 0.95,
        "why": "A cut-vertex: removing this actor physically disconnects part of the "
               "network, so they are the sole conduit to at least one cell.",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Expert rule engine
# ──────────────────────────────────────────────────────────────────────────────

def _gatekeeper_cut_sizes(G: nx.Graph, max_eval: int = 4000) -> dict[str, int]:
    """For each cut-vertex, the size of the largest cell it alone gates.

    Uses the block-cut tree rather than V removal simulations: each articulation point
    belongs to several biconnected components, and removing it splits its component
    into pieces made of those blocks. We report the size of the smallest piece severed,
    which is the number of actors that lose all contact with the main body.

    Returns ``{person_id: severed_cell_size}`` for cut-vertices only.
    """
    try:
        cut_vertices = list(nx.articulation_points(G))
    except Exception:                                       # noqa: BLE001
        return {}
    if not cut_vertices or len(cut_vertices) > max_eval:
        return {}

    sizes: dict[str, int] = {}
    # Group cut-vertices by component so each removal only touches a local subgraph.
    comp_of = {}
    for i, comp in enumerate(nx.connected_components(G)):
        for node in comp:
            comp_of[node] = i
    comps = {i: set(c) for i, c in enumerate(nx.connected_components(G))}

    for v in cut_vertices:
        comp = comps[comp_of[v]]
        sub = G.subgraph(comp - {v})
        pieces = sorted((len(c) for c in nx.connected_components(sub)), reverse=True)
        # pieces[0] is the surviving main body; everything after it is severed.
        sizes[v] = sum(pieces[1:]) if len(pieces) > 1 else 0
    return sizes


def apply_rules(
    G: nx.Graph,
    features: pd.DataFrame,
    centralities: pd.DataFrame,
    partition: dict[str, int],
    reference_year: int = 2026,
) -> tuple[pd.Series, pd.Series]:
    """Run the eight doctrine rules (R1–R8).

    Returns:
        ``(rule_score, triggered_rules)`` — score normalised to 0–1, and a Series of
        lists naming which rules fired for each actor.
    """
    nodes = list(G.nodes())
    if not nodes:
        return pd.Series(dtype=float), pd.Series(dtype=object)

    p = RULE_PARAMS
    cent = centralities.reindex(nodes).fillna(0.0)
    feat = features.reindex(nodes).fillna(0.0)

    pct_btw = percentile_rank(cent["betweenness"])
    pct_deg = percentile_rank(cent["degree"])
    pct_eig = percentile_rank(cent["eigenvector"])

    triggered: dict[str, list[str]] = {n: [] for n in nodes}

    # Cut-vertices (articulation points) — O(V+E) for the whole graph. A cut-vertex is
    # the *only* route to some part of the network. Most are trivial (one pendant
    # contact dangling off them), so we also measure how large a cell each one severs
    # and keep only the ones that gate a meaningful group.
    cut_sizes = _gatekeeper_cut_sizes(G)

    for n in nodes:
        attrs = G.nodes[n]
        deg = G.degree(n)
        nbr_modules = {partition.get(m) for m in G.neighbors(n)} - {None}
        external_modules = nbr_modules - {partition.get(n)}
        # R1 — broker: flow control disproportionate to contact count.
        if (pct_btw[n] >= p["broker_betweenness_pct"]
                and pct_deg[n] <= p["broker_degree_pct"]
                and external_modules):
            triggered[n].append("cross_module_broker")

        # R2 — connector spanning several modules at once.
        if len(nbr_modules) >= p["bridge_min_communities"]:
            triggered[n].append("structural_bridge")

        # R3 — money mule: financial hub with no criminal history.
        if (feat.loc[n, "share_financial"] >= p["mule_financial_ratio"]
                and attrs.get("prior_offences", 0) <= p["mule_max_offences"]
                and deg >= 3):
            triggered[n].append("financial_mule")

        # R4 — insulated serious offender.
        max_sev = feat.loc[n, "max_case_severity"]
        offences = attrs.get("offence_list") or []
        serious = max_sev >= p["isolate_min_severity"] or (
            len(offences) > 0 and attrs.get("prior_offences", 0) >= 5)
        if deg <= p["isolate_max_degree"] and serious:
            triggered[n].append("isolated_high_severity")

        # R5 — alias proliferation.
        if attrs.get("alias_count", 0) >= p["alias_min"]:
            triggered[n].append("identity_evasion")

        # R6 — dormant, then a burst of new contacts. The signal is not merely "has some
        # recent links" (nearly everyone does) but that the *majority* of this actor's
        # contact set was established in the last year after a long prior history.
        recent_links = 0
        for m in G.neighbors(n):
            first = G[n][m].get("first_seen")
            if first is not None and pd.notna(first):
                try:
                    if reference_year - pd.Timestamp(first).year <= 1:
                        recent_links += 1
                except Exception:                           # noqa: BLE001
                    pass
        entry = attrs.get("first_arrest_year", 0)
        long_history = bool(entry) and (reference_year - entry) >= p["dormancy_years"]
        if (long_history
                and recent_links >= p["dormancy_min_new_links"]
                and recent_links / max(1, deg) >= p["dormancy_new_link_ratio"]):
            triggered[n].append("dormancy_spike")

        # R7 — rank inversion: structurally senior to contacts with heavier records.
        if deg >= 3:
            nbr_priors = [G.nodes[m].get("prior_offences", 0) for m in G.neighbors(n)]
            own_priors = attrs.get("prior_offences", 0)
            if (pct_eig[n] >= 0.75
                    and np.mean(nbr_priors) - own_priors >= 3
                    and pct_eig[n] - pct_deg[n] >= p["hierarchy_gap"]):
                triggered[n].append("hierarchy_inversion")

        # R8 — cut-vertex gating a cell of real size.
        if (cut_sizes.get(n, 0) >= p["gatekeeper_min_cut"]
                and deg >= p["gatekeeper_min_degree"]):
            triggered[n].append("structural_gatekeeper")

    # Score = saturating sum of rule weights. Two strong rules should outrank four weak
    # ones, and no actor should max out on volume alone. The saturation denominator is
    # the realistic maximum co-firing rule mass (RULE_SATURATION), not the theoretical
    # sum of all rules — many rules are mutually exclusive.
    raw = pd.Series(
        {n: sum(RULE_CATALOGUE[r]["weight"] for r in rules) for n, rules in triggered.items()},
        dtype=float)
    denom = float(RULE_SATURATION)
    score = (1.0 - np.exp(-2.2 * raw / denom)) / (1.0 - np.exp(-2.2))
    triggered_series = pd.Series(triggered, dtype=object)
    triggered_series.attrs["cut_sizes"] = cut_sizes
    return score.clip(0, 1), triggered_series


# ──────────────────────────────────────────────────────────────────────────────
# Isolation Forest
# ──────────────────────────────────────────────────────────────────────────────

def isolation_forest_scores(features: pd.DataFrame, contamination: float = 0.08,
                            seed: int = 42) -> pd.Series:
    """Isolation Forest outlier score, normalised so 1.0 = most anomalous."""
    if len(features) < 12:
        return pd.Series(0.0, index=features.index, name="iforest")

    X = features.fillna(0.0).to_numpy(dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    X = (X - mu) / sd

    try:
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(
            n_estimators=300,
            contamination=float(np.clip(contamination, 0.01, 0.4)),
            max_samples=min(256, len(X)),
            random_state=seed,
            n_jobs=-1,
        ).fit(X)
        # score_samples: lower = more anomalous. Negate so higher = more anomalous.
        raw = -model.score_samples(X)
    except Exception:                                       # noqa: BLE001 — sklearn absent
        # Mahalanobis-style fallback keeps the pipeline running without sklearn.
        raw = np.linalg.norm(X, axis=1)

    s = pd.Series(raw, index=features.index, name="iforest")
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return pd.Series(0.0, index=features.index, name="iforest")
    return (s - lo) / (hi - lo)


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble
# ──────────────────────────────────────────────────────────────────────────────

def band_for(score: float) -> tuple[str, str]:
    """Map a 0–100 suspicion score to its band label and colour."""
    for threshold, label, colour in SUSPICION_BANDS:
        if score >= threshold:
            return label, colour
    return SUSPICION_BANDS[-1][1], SUSPICION_BANDS[-1][2]


def _explain(
    node: str,
    G: nx.Graph,
    rules: list[str],
    gnn_res: float,
    iforest: float,
    features: pd.DataFrame,
    centralities: pd.DataFrame,
    cut_sizes: dict[str, int] | None = None,
) -> list[str]:
    """Compose human-readable justifications, strongest signal first."""
    reasons = []
    for r in rules:
        text = f"**{RULE_CATALOGUE[r]['label']}** — {RULE_CATALOGUE[r]['why']}"
        if r == "structural_gatekeeper" and cut_sizes:
            severed = cut_sizes.get(node, 0)
            if severed:
                text += (f" Removing this actor severs **{severed} actor(s)** from the "
                         "rest of the network.")
        reasons.append(text)

    if gnn_res >= 0.80:
        reasons.append(
            "**GNN structural mismatch** — the trained encoder cannot reconstruct this "
            f"actor's connection pattern (residual {gnn_res:.2f}); their links do not "
            "follow the grammar the rest of the network obeys.")
    if iforest >= 0.80:
        # Name the two features that deviate most — a bare "outlier" claim is unusable.
        z = ((features - features.mean()) / features.std().replace(0, 1)).loc[node]
        top = z.abs().sort_values(ascending=False).head(2)
        drivers = ", ".join(f"{k.replace('_', ' ')} ({z[k]:+.1f}σ)" for k in top.index)
        reasons.append(f"**Statistical outlier** — unusual feature combination: {drivers}.")

    if node in centralities.index:
        row = centralities.loc[node]
        if row["degree"] <= 3 and percentile_rank(centralities["betweenness"]).get(node, 0) >= 0.9:
            reasons.append(
                f"**Disproportionate control** — only {int(row['degree'])} recorded contact(s) "
                "yet sits on a large share of the network's shortest paths.")

    if not reasons:
        reasons.append("No individual rule fired; score reflects mild statistical deviation only.")
    return reasons


def detect_anomalies(
    G: nx.Graph,
    features: pd.DataFrame,
    centralities: pd.DataFrame,
    partition: dict[str, int],
    gnn_residuals: pd.Series | None = None,
    weights: dict[str, float] | None = None,
    contamination: float = 0.08,
    reference_year: int = 2026,
) -> pd.DataFrame:
    """Run the full ensemble. Returns one row per actor, sorted most suspicious first."""
    nodes = list(G.nodes())
    if not nodes:
        return pd.DataFrame()

    w = {**ANOMALY_WEIGHTS, **(weights or {})}
    total = sum(w.values()) or 1.0
    w = {k: v / total for k, v in w.items()}

    feat = features.reindex(nodes).fillna(0.0)
    rule_score, triggered = apply_rules(G, feat, centralities, partition, reference_year)
    iforest = isolation_forest_scores(feat, contamination)

    if gnn_residuals is not None and len(gnn_residuals):
        gnn = gnn_residuals.reindex(nodes).fillna(gnn_residuals.median())
    else:
        gnn = pd.Series(0.0, index=nodes)

    # Connectivity gate: for actors that trigger NO expert rule, damp the ML signals in
    # proportion to how embedded they are in the network. A well-connected actor keeps
    # full ML weight; a barely-connected loner does not get flagged on statistical
    # oddness alone. Rule-flagged actors are never gated — a rule is domain justification
    # enough to warrant the full model contribution.
    degrees = pd.Series({n: G.degree(n) for n in nodes}, dtype=float)
    floor = float(ANOMALY_MODEL_GATE_FLOOR)
    engagement = (degrees.clip(upper=5.0) / 5.0)
    gate = (floor + (1.0 - floor) * engagement).clip(floor, 1.0)
    has_rule = rule_score.reindex(nodes).fillna(0.0) > 0
    gate[has_rule] = 1.0

    combined = (w["gnn_residual"] * gnn * gate
                + w["isolation_forest"] * iforest * gate
                + w["rule_engine"] * rule_score)
    suspicion = (100.0 * combined.clip(0, 1)).round(2)

    bands = suspicion.apply(lambda s: band_for(float(s)))
    cut_sizes = triggered.attrs.get("cut_sizes", {})

    rows = []
    for n in nodes:
        attrs = G.nodes[n]
        rules = triggered.get(n, []) or []
        rows.append({
            "person_id": n,
            "name": attrs.get("name", n),
            "alias": "; ".join(attrs.get("alias_list") or []),
            "city": attrs.get("city", "Unknown"),
            "state": attrs.get("state", "Unknown"),
            "status": attrs.get("status", "Active"),
            "risk_category": attrs.get("risk_category", "Low"),
            "module": partition.get(n, -1),
            "degree": int(G.degree(n)),
            "prior_offences": attrs.get("prior_offences", 0),
            "suspicion_score": float(suspicion[n]),
            "band": bands[n][0],
            "band_colour": bands[n][1],
            "gnn_residual": round(float(gnn[n]), 4),
            "iforest_score": round(float(iforest[n]), 4),
            "rule_score": round(float(rule_score[n]), 4),
            "n_rules": len(rules),
            "rules": [RULE_CATALOGUE[r]["label"] for r in rules],
            "rule_keys": rules,
            "reasons": _explain(n, G, rules, float(gnn[n]), float(iforest[n]),
                                feat, centralities, cut_sizes),
            "severed_cell": int(cut_sizes.get(n, 0)),
        })

    df = pd.DataFrame(rows).set_index("person_id")
    return df.sort_values("suspicion_score", ascending=False)


# ──────────────────────────────────────────────────────────────────────────────
# Suspicious clusters
# ──────────────────────────────────────────────────────────────────────────────

def suspicious_clusters(
    G: nx.Graph,
    partition: dict[str, int],
    anomalies: pd.DataFrame,
    profiles: pd.DataFrame,
    min_size: int = 3,
) -> pd.DataFrame:
    """Rank modules by how much anomalous activity they concentrate.

    A module matters when it combines *cohesion* (tight internal structure = command
    and control) with *anomaly density* (several flagged actors) and *severity*.
    """
    if not partition or len(profiles) == 0:
        return pd.DataFrame()

    rows = []
    for rec in profiles.to_dict("records"):
        cid = rec["community_id"]
        members = [n for n, c in partition.items() if c == cid]
        if len(members) < min_size:
            continue

        sub = anomalies.reindex(members).dropna(subset=["suspicion_score"])
        if len(sub) == 0:
            continue

        flagged = sub[sub["band"].isin(["Critical", "High"])]
        anomaly_density = len(flagged) / len(members)

        # Cluster suspicion: module threat, concentration of flagged actors, and the
        # peak individual score, so one Critical actor cannot alone carry a module.
        cluster_score = (
            0.42 * (rec["threat_score"] / 100.0)
            + 0.33 * min(1.0, anomaly_density * 2.5)
            + 0.25 * (float(sub["suspicion_score"].max()) / 100.0)
        )

        rules_seen: list[str] = []
        for r in sub["rules"]:
            rules_seen.extend(r or [])
        signature = (pd.Series(rules_seen).value_counts().head(3).index.tolist()
                     if rules_seen else [])

        rows.append({
            "community_id": cid,
            "label": rec["label"],
            "size": len(members),
            "flagged_actors": len(flagged),
            "anomaly_density": round(anomaly_density, 3),
            "mean_suspicion": round(float(sub["suspicion_score"].mean()), 1),
            "peak_suspicion": round(float(sub["suspicion_score"].max()), 1),
            "density": rec["density"],
            "insulation": rec["insulation"],
            "mean_severity": rec["mean_severity"],
            "states": rec["states"],
            "state_list": rec["state_list"],
            "dominant_offence": rec["dominant_offence"],
            "lead_actor": rec["lead_actor"],
            "lead_actor_id": rec["lead_actor_id"],
            "signature_patterns": signature,
            "cluster_score": round(100.0 * float(np.clip(cluster_score, 0, 1)), 1),
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("cluster_score", ascending=False)
    df["band"] = df["cluster_score"].apply(lambda s: band_for(float(s))[0])
    return df.reset_index(drop=True)


def anomaly_summary(anomalies: pd.DataFrame) -> dict:
    """Counts per band plus rule-trigger frequencies, for the KPI strip."""
    if len(anomalies) == 0:
        return {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0,
                "rule_counts": {}, "mean_score": 0.0}
    counts = anomalies["band"].value_counts()
    all_rules: list[str] = []
    for r in anomalies["rules"]:
        all_rules.extend(r or [])
    return {
        "total": len(anomalies),
        "critical": int(counts.get("Critical", 0)),
        "high": int(counts.get("High", 0)),
        "medium": int(counts.get("Medium", 0)),
        "low": int(counts.get("Low", 0)),
        "rule_counts": pd.Series(all_rules).value_counts().to_dict() if all_rules else {},
        "mean_score": round(float(anomalies["suspicion_score"].mean()), 2),
    }
