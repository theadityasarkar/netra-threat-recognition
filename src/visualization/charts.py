"""
Layer 4b — analytical charts (Plotly).

A library of small, self-contained figures the dashboard composes into its pages.
Every chart takes an :class:`~src.pipeline.AnalysisResult` (or a slice of one) and
returns a themed ``go.Figure``. They never touch Streamlit, so they can be dropped into
the PDF report exporter unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import (
    COMMUNITY_COLORS,
    ENTITY_COLORS,
    PALETTE,
    RISK_COLORS,
    ROLE_COLORS,
    STATUS_COLORS,
    SUSPICION_BANDS,
)

# Bands, brightest first, for the suspicion colour ramp.
_BAND_COLOURS = {name: colour for _, name, colour in SUSPICION_BANDS}


def _base_layout(fig: go.Figure, height: int = 320, title: str | None = None,
                 showlegend: bool = False) -> go.Figure:
    """Apply the shared dark theme so every chart reads as one family."""
    # Omitting the ``title`` key is not the same as setting it to ``None``. Streamlit's
    # Plotly theme bolds ``layout.title.text`` client-side, so an explicit null reaches
    # the browser as JavaScript ``undefined`` and renders the literal word "undefined"
    # as the chart heading. Declare the title only when there is one.
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13, color=PALETTE["text"]),
                                     x=0.01))
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text_dim"], size=11),
        margin=dict(l=10, r=14, t=38 if title else 14, b=10),
        showlegend=showlegend,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=PALETTE["text_dim"])),
        colorway=COMMUNITY_COLORS,
    )
    fig.update_xaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False,
                     linecolor=PALETTE["border"], tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False,
                     linecolor=PALETTE["border"], tickfont=dict(size=10))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Distributions
# ──────────────────────────────────────────────────────────────────────────────

def risk_distribution_bar(result) -> go.Figure:
    """Head-count of actors in each risk category."""
    order = ["Critical", "High", "Medium", "Low"]
    counts = pd.Series(
        [result.graph.nodes[n].get("risk_category", "Low") for n in result.graph.nodes()]
    ).value_counts()
    counts = counts.reindex(order).fillna(0)
    fig = go.Figure(go.Bar(
        x=order, y=counts.values,
        marker_color=[RISK_COLORS[r] for r in order],
        text=counts.values.astype(int), textposition="outside",
        hovertemplate="%{x}: %{y} actors<extra></extra>",
    ))
    return _base_layout(fig, title="Risk category distribution")


def suspicion_histogram(result) -> go.Figure:
    """Distribution of suspicion scores, coloured by band boundary."""
    if not len(result.anomalies):
        return _base_layout(go.Figure(), title="Suspicion score distribution")
    scores = result.anomalies["suspicion_score"]
    fig = go.Figure(go.Histogram(
        x=scores, nbinsx=28, marker_color=PALETTE["accent"],
        marker_line_color=PALETTE["bg"], marker_line_width=1,
        hovertemplate="Score %{x}: %{y} actors<extra></extra>",
    ))
    # Threshold guides for each band cut-point.
    for thresh, name, colour in SUSPICION_BANDS:
        if thresh > 0:
            fig.add_vline(x=thresh, line_dash="dot", line_color=colour, line_width=1,
                          annotation_text=name, annotation_position="top",
                          annotation_font=dict(size=9, color=colour))
    fig.update_layout(bargap=0.05)
    return _base_layout(fig, title="Suspicion score distribution")


def band_donut(result, title: str | None = "Threat bands") -> go.Figure:
    """Actors per suspicion band — the headline anomaly breakdown.

    ``title`` is a parameter because the Dashboard already announces this chart with a
    styled section header; printing "Threat bands" twice, one line apart, reads as a
    rendering fault rather than a design.
    """
    if not len(result.anomalies):
        return _base_layout(go.Figure(), title=title)
    counts = result.anomalies["band"].value_counts()
    order = [name for _, name, _ in SUSPICION_BANDS]
    counts = counts.reindex(order).fillna(0)
    fig = go.Figure(go.Pie(
        labels=order, values=counts.values, hole=0.62,
        marker=dict(colors=[_BAND_COLOURS[b] for b in order],
                    line=dict(color=PALETTE["bg"], width=2)),
        textinfo="value", textfont=dict(size=12, color=PALETTE["text"]),
        hovertemplate="%{label}: %{value} actors (%{percent})<extra></extra>",
        sort=False,
    ))
    total = int(counts.sum())
    fig.add_annotation(text=f"<b>{total}</b><br>actors", showarrow=False,
                       font=dict(size=16, color=PALETTE["text"]))
    return _base_layout(fig, title=title, showlegend=True)


def role_distribution(result) -> go.Figure:
    """Head-count per inferred operational role, in doctrine order."""
    if not len(result.key_players) or "role" not in result.key_players.columns:
        return _base_layout(go.Figure(), title="Operational roles")
    order = ["Kingpin", "Lieutenant", "Broker", "Financier", "Enforcer",
             "Foot Soldier", "Peripheral"]
    counts = result.key_players["role"].value_counts().reindex(order).fillna(0)
    fig = go.Figure(go.Bar(
        y=order[::-1], x=counts.reindex(order[::-1]).values, orientation="h",
        marker_color=[ROLE_COLORS[r] for r in order[::-1]],
        text=counts.reindex(order[::-1]).values.astype(int), textposition="outside",
        hovertemplate="%{y}: %{x} actors<extra></extra>",
    ))
    return _base_layout(fig, title="Operational role hierarchy", height=300)


# ──────────────────────────────────────────────────────────────────────────────
# Key players
# ──────────────────────────────────────────────────────────────────────────────

def influence_ranking(result, top_n: int = 15) -> go.Figure:
    """Horizontal bar of the top actors by influence score."""
    if not len(result.key_players):
        return _base_layout(go.Figure(), title="Top actors by influence")
    top = result.key_players.nlargest(top_n, "influence_score")
    labels = [f"{result.graph.nodes[i].get('name', i).split()[0]} · {i}" for i in top.index]
    roles = top["role"] if "role" in top.columns else pd.Series("", index=top.index)
    fig = go.Figure(go.Bar(
        y=labels[::-1], x=top["influence_score"].values[::-1], orientation="h",
        marker=dict(
            color=[ROLE_COLORS.get(r, PALETTE["accent"]) for r in roles.values[::-1]],
            line=dict(color=PALETTE["bg"], width=0.5)),
        text=[f"{v:.1f}" for v in top["influence_score"].values[::-1]],
        textposition="outside",
        customdata=roles.values[::-1],
        hovertemplate="%{y}<br>Influence %{x:.1f} · %{customdata}<extra></extra>",
    ))
    return _base_layout(fig, title=f"Top {top_n} actors by influence", height=max(320, top_n * 26))


def centrality_scatter(result) -> go.Figure:
    """Betweenness vs PageRank — the classic broker-vs-hub quadrant chart.

    Actors high on betweenness but low on PageRank are *brokers*: low-profile
    connectors whose removal fragments the network. This is the single most
    important diagnostic plot for an analyst.
    """
    c = result.centralities
    if not len(c):
        return _base_layout(go.Figure(), title="Broker vs hub map")
    roles = (result.key_players["role"].reindex(c.index)
             if len(result.key_players) else pd.Series("Peripheral", index=c.index))
    susp = (result.anomalies["suspicion_score"].reindex(c.index).fillna(0)
            if len(result.anomalies) else pd.Series(0, index=c.index))
    names = [result.graph.nodes[i].get("name", i) for i in c.index]

    fig = go.Figure(go.Scatter(
        x=c["pagerank"], y=c["betweenness"], mode="markers",
        marker=dict(
            size=8 + 16 * (susp / 100.0),
            color=[ROLE_COLORS.get(r, PALETTE["text_dim"]) for r in roles],
            line=dict(width=0.5, color=PALETTE["bg"]), opacity=0.85),
        customdata=np.stack([names, roles.values, susp.values], axis=-1),
        hovertemplate=("<b>%{customdata[0]}</b><br>PageRank %{x:.4f} · "
                       "Betweenness %{y:.4f}<br>%{customdata[1]} · "
                       "Suspicion %{customdata[2]:.0f}<extra></extra>"),
    ))
    # Median guides split the plane into the four operational quadrants.
    fig.add_vline(x=float(c["pagerank"].median()), line_dash="dot",
                  line_color=PALETTE["border"], line_width=1)
    fig.add_hline(y=float(c["betweenness"].median()), line_dash="dot",
                  line_color=PALETTE["border"], line_width=1)
    fig.add_annotation(x=c["pagerank"].max(), y=c["betweenness"].max(),
                       text="hubs & brokers", showarrow=False,
                       font=dict(size=9, color=PALETTE["text_dim"]),
                       xanchor="right", yanchor="top")
    fig.update_layout(xaxis_title="PageRank (prominence)",
                      yaxis_title="Betweenness (brokerage)")
    return _base_layout(fig, title="Broker vs hub map", height=420)


def disruption_ranking(result, top_n: int = 10) -> go.Figure:
    """Arrest-priority chart: expected network damage from removing each actor."""
    if not len(result.priority):
        return _base_layout(go.Figure(), title="Disruption priority")
    top = result.priority.nlargest(top_n, "disruption_score") if "disruption_score" in result.priority.columns else result.priority.head(top_n)
    labels = [f"{result.graph.nodes[i].get('name', i).split()[0]} · {i}" for i in top.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels[::-1], x=top["disruption_score"].values[::-1], orientation="h",
        marker=dict(color=top["disruption_score"].values[::-1],
                    colorscale=[[0, PALETTE["accent_2"]], [1, PALETTE["critical"]]],
                    line=dict(color=PALETTE["bg"], width=0.5)),
        text=[f"{v:.1f}" for v in top["disruption_score"].values[::-1]],
        textposition="outside",
        customdata=np.stack([
            top.get("lcc_drop_pct", pd.Series(0, index=top.index)).values[::-1],
            top.get("new_fragments", pd.Series(0, index=top.index)).values[::-1],
        ], axis=-1),
        hovertemplate=("%{y}<br>Disruption %{x:.1f}<br>Largest-cluster drop "
                       "%{customdata[0]:.0f}%% · +%{customdata[1]:.0f} fragments<extra></extra>"),
    ))
    return _base_layout(fig, title=f"Top {top_n} disruption targets", height=max(300, top_n * 30))


# ──────────────────────────────────────────────────────────────────────────────
# Communities
# ──────────────────────────────────────────────────────────────────────────────

def community_threat_bars(result) -> go.Figure:
    """Per-module threat score, sized by member count."""
    p = result.profiles
    if not len(p):
        return _base_layout(go.Figure(), title="Module threat ranking")
    p = p.sort_values("threat_score", ascending=True)
    labels = [f"Module {int(c) + 1}" for c in p["community_id"]] if "community_id" in p.columns \
        else [f"Module {i + 1}" for i in range(len(p))]
    colours = [COMMUNITY_COLORS[int(c) % len(COMMUNITY_COLORS)]
               for c in (p["community_id"] if "community_id" in p.columns else range(len(p)))]
    fig = go.Figure(go.Bar(
        y=labels, x=p["threat_score"], orientation="h",
        marker=dict(color=colours, line=dict(color=PALETTE["bg"], width=0.5)),
        text=[f"{v:.0f}" for v in p["threat_score"]], textposition="outside",
        customdata=p["size"].values if "size" in p.columns else np.zeros(len(p)),
        hovertemplate="%{y}<br>Threat %{x:.0f} · %{customdata:.0f} members<extra></extra>",
    ))
    return _base_layout(fig, title="Module threat ranking", height=max(280, len(p) * 30))


def community_scatter(result) -> go.Figure:
    """Module map: size vs internal density, bubble-scaled by threat."""
    p = result.profiles
    if not len(p) or "size" not in p.columns:
        return _base_layout(go.Figure(), title="Module landscape")
    density = p["density"] if "density" in p.columns else pd.Series(0, index=p.index)
    threat = p["threat_score"] if "threat_score" in p.columns else pd.Series(20, index=p.index)
    comm = p["community_id"] if "community_id" in p.columns else pd.Series(range(len(p)))
    fig = go.Figure(go.Scatter(
        x=p["size"], y=density, mode="markers+text",
        marker=dict(size=12 + 34 * (threat / max(1.0, float(threat.max()))),
                    color=[COMMUNITY_COLORS[int(c) % len(COMMUNITY_COLORS)] for c in comm],
                    line=dict(color=PALETTE["bg"], width=1), opacity=0.85),
        text=[f"M{int(c) + 1}" for c in comm], textposition="middle center",
        textfont=dict(size=9, color=PALETTE["bg"]),
        customdata=threat.values,
        hovertemplate=("Module %{text}<br>%{x} members · density %{y:.3f}<br>"
                       "threat %{customdata:.0f}<extra></extra>"),
    ))
    fig.update_layout(xaxis_title="Members", yaxis_title="Internal density")
    return _base_layout(fig, title="Module landscape", height=380)


# ──────────────────────────────────────────────────────────────────────────────
# GNN
# ──────────────────────────────────────────────────────────────────────────────

def gnn_loss_curve(result) -> go.Figure:
    """Training loss over epochs — evidence the model actually converged."""
    hist = result.gnn.loss_history
    if not hist:
        return _base_layout(go.Figure(), title="GNN training loss")
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(hist) + 1)), y=hist, mode="lines",
        line=dict(color=PALETTE["accent"], width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
        hovertemplate="Epoch %{x}: loss %{y:.4f}<extra></extra>",
    ))
    fig.add_annotation(
        x=len(hist), y=hist[-1],
        text=f"AUC {result.gnn.auc:.3f}", showarrow=True, arrowhead=0,
        font=dict(size=11, color=PALETTE["low"]), ax=-40, ay=-20)
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Link-prediction loss")
    return _base_layout(fig, title="GNN training convergence")


def embedding_scatter(result, color_by: str = "module") -> go.Figure:
    """2-D PCA projection of the GNN embedding space.

    Clusters here are *behavioural* — actors who play the same structural role land
    together even across different organisations.
    """
    g = result.gnn
    if not len(g.node_ids):
        return _base_layout(go.Figure(), title="GNN embedding space")
    coords = g.coords_2d
    ids = g.node_ids
    names = [result.graph.nodes[i].get("name", i) for i in ids]

    if color_by == "suspicion" and len(result.anomalies):
        vals = result.anomalies["suspicion_score"].reindex(ids).fillna(0).values
        marker = dict(size=8, color=vals,
                      colorscale=[[0, PALETTE["low"]], [0.5, PALETTE["medium"]],
                                  [1, PALETTE["critical"]]],
                      colorbar=dict(title="Suspicion", thickness=10, len=0.5),
                      line=dict(width=0.4, color=PALETTE["bg"]), opacity=0.85)
    else:
        modules = [result.partition.get(i, -1) for i in ids]
        marker = dict(size=8,
                      color=[COMMUNITY_COLORS[m % len(COMMUNITY_COLORS)] if m >= 0
                             else PALETTE["text_dim"] for m in modules],
                      line=dict(width=0.4, color=PALETTE["bg"]), opacity=0.85)

    fig = go.Figure(go.Scatter(
        x=coords[:, 0], y=coords[:, 1], mode="markers", marker=marker,
        customdata=names,
        hovertemplate="<b>%{customdata}</b><br>PC1 %{x:.2f} · PC2 %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="PC-1", yaxis_title="PC-2")
    return _base_layout(fig, title="GNN embedding space (behavioural map)", height=420)


# ──────────────────────────────────────────────────────────────────────────────
# Anomaly composition & correlations
# ──────────────────────────────────────────────────────────────────────────────

def anomaly_composition(result, top_n: int = 12) -> go.Figure:
    """Stacked contribution of each signal to the top suspects' scores.

    Makes the ensemble legible: you can see at a glance whether a suspect was flagged
    by the expert rules, the Isolation Forest, the GNN, or a combination.
    """
    a = result.anomalies
    if not len(a):
        return _base_layout(go.Figure(), title="Anomaly signal composition")
    top = a.nlargest(top_n, "suspicion_score")
    labels = [f"{result.graph.nodes[i].get('name', i).split()[0]} · {i}" for i in top.index]

    # The stored component columns are 0–1; scale to the same 0–100 space as the
    # headline score so the bar heights are interpretable.
    from src.config import ANOMALY_WEIGHTS as W
    comps = [
        ("Expert rules", "rule_score", W["rule_engine"], PALETTE["critical"]),
        ("Isolation Forest", "iforest_score", W["isolation_forest"], PALETTE["accent_2"]),
        ("GNN residual", "gnn_residual", W["gnn_residual"], PALETTE["accent"]),
    ]
    fig = go.Figure()
    for name, col, weight, colour in comps:
        if col in top.columns:
            fig.add_trace(go.Bar(
                y=labels[::-1], x=(top[col].values[::-1] * weight * 100.0),
                orientation="h", name=name, marker_color=colour,
                hovertemplate=f"{name}: %{{x:.1f}} pts<extra></extra>"))
    fig.update_layout(barmode="stack", xaxis_title="Contribution to suspicion score")
    return _base_layout(fig, title=f"Signal composition · top {top_n} suspects",
                        height=max(320, top_n * 30), showlegend=True)


def rule_frequency(result) -> go.Figure:
    """How often each expert rule fired across the population."""
    a = result.anomalies
    if not len(a) or "rule_keys" not in a.columns:
        return _base_layout(go.Figure(), title="Expert-rule firing frequency")
    from src.analysis.anomaly import RULE_CATALOGUE
    counts: dict[str, int] = {}
    for keys in a["rule_keys"]:
        for k in (keys or []):
            counts[k] = counts.get(k, 0) + 1
    if not counts:
        return _base_layout(go.Figure(), title="Expert-rule firing frequency")
    labels, vals = [], []
    for k, cnt in sorted(counts.items(), key=lambda kv: kv[1]):
        labels.append(RULE_CATALOGUE.get(k, {}).get("label", k))
        vals.append(cnt)
    fig = go.Figure(go.Bar(
        y=labels, x=vals, orientation="h",
        marker=dict(color=PALETTE["high"], line=dict(color=PALETTE["bg"], width=0.5)),
        text=vals, textposition="outside",
        hovertemplate="%{y}: fired for %{x} actor(s)<extra></extra>",
    ))
    return _base_layout(fig, title="Expert-rule firing frequency", height=max(260, len(labels) * 34))


def correlation_heatmap(result) -> go.Figure:
    """Spearman correlation between centrality measures.

    High correlation between two measures means they carry redundant signal — useful
    for defending why the influence index blends complementary metrics rather than
    stacking six versions of degree.
    """
    corr = result.correlations
    if not len(corr):
        return _base_layout(go.Figure(), title="Centrality correlations")
    labels = [c.replace("_", " ").title() for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale=[[0, PALETTE["critical"]], [0.5, PALETTE["bg"]], [1, PALETTE["accent"]]],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=9, color=PALETTE["text"]),
        hovertemplate="%{y} ↔ %{x}: ρ=%{z:.2f}<extra></extra>",
        colorbar=dict(thickness=10, len=0.7),
    ))
    return _base_layout(fig, title="Centrality measure correlations (Spearman ρ)", height=420)


def geographic_bars(result, top_n: int = 12) -> go.Figure:
    """Actor count by state — where the network is physically concentrated."""
    states = pd.Series(
        [result.graph.nodes[n].get("state", "Unknown") for n in result.graph.nodes()]
    ).value_counts().head(top_n)
    if not len(states):
        return _base_layout(go.Figure(), title="Geographic footprint")
    fig = go.Figure(go.Bar(
        x=states.values[::-1], y=states.index[::-1], orientation="h",
        marker=dict(color=PALETTE["accent"], line=dict(color=PALETTE["bg"], width=0.5)),
        text=states.values[::-1], textposition="outside",
        hovertemplate="%{y}: %{x} actors<extra></extra>",
    ))
    return _base_layout(fig, title="Geographic footprint (by state)", height=max(280, len(states) * 26))


def relation_type_bars(result) -> go.Figure:
    """Edge count by relationship type, split recorded vs inferred co-accused."""
    rel_counts: dict[str, int] = {}
    inferred = 0
    for _, _, d in result.graph.edges(data=True):
        for rt in (d.get("relation_types") or [d.get("relation_type", "Associate")]):
            rel_counts[rt] = rel_counts.get(rt, 0) + 1
        if d.get("inferred"):
            inferred += 1
    if not rel_counts:
        return _base_layout(go.Figure(), title="Relationship composition")
    s = pd.Series(rel_counts).sort_values()
    fig = go.Figure(go.Bar(
        y=s.index, x=s.values, orientation="h",
        marker=dict(color=PALETTE["accent_2"], line=dict(color=PALETTE["bg"], width=0.5)),
        text=s.values, textposition="outside",
        hovertemplate="%{y}: %{x} links<extra></extra>",
    ))
    return _base_layout(fig, title="Relationship type composition", height=max(260, len(s) * 30))


# ──────────────────────────────────────────────────────────────────────────────
# Women Safety Division lens (PS 26189 owning department)
# ──────────────────────────────────────────────────────────────────────────────

# WSTI band ramp, mirroring the suspicion palette so the two threat scales read
# consistently across the dashboard.
_WS_BAND_COLOURS = {
    "Severe": PALETTE["critical"],
    "High": PALETTE["high"],
    "Moderate": PALETTE["medium"],
    "Low": PALETTE["low"],
}


def wsti_ranking(result, top_n: int = 12) -> go.Figure:
    """Highest Women-Safety Threat Index actors, coloured by band."""
    ws = getattr(result, "wsti", None)
    if ws is None or not len(ws):
        return _base_layout(go.Figure(), title="Women-Safety Threat Index")
    top = ws.nlargest(min(top_n, len(ws)), "wsti").iloc[::-1]
    labels = [f"{r['name']} ({r['person_id']})" for _, r in top.iterrows()]
    colours = [_WS_BAND_COLOURS.get(str(b), PALETTE["accent"]) for b in top["band"]]
    fig = go.Figure(go.Bar(
        x=top["wsti"], y=labels, orientation="h",
        marker=dict(color=colours, line=dict(color=PALETTE["bg"], width=0.5)),
        text=[f"{v:.1f}" for v in top["wsti"]], textposition="outside",
        customdata=np.stack([top["ws_cases"], top["band"]], axis=-1),
        hovertemplate="%{y}<br>WSTI %{x:.1f} · %{customdata[1]}"
                      "<br>%{customdata[0]} division case(s)<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 105])
    return _base_layout(fig, title="Women-Safety Threat Index — highest scoring actors",
                        height=max(300, len(top) * 30))


def ws_crime_breakdown(result) -> go.Figure:
    """Division caseload by offence type, cyber-enabled offences highlighted.

    The split speaks directly to the PS theme (*Blockchain & Cybersecurity*) with
    data rather than assertion.
    """
    bd = getattr(result, "ws_breakdown", None)
    if bd is None or not len(bd):
        return _base_layout(go.Figure(), title="Division caseload by offence")
    bd = bd.iloc[::-1]
    colours = [PALETTE["accent"] if c else PALETTE["accent_2"] for c in bd["cyber_enabled"]]
    fig = go.Figure(go.Bar(
        x=bd["cases"], y=bd["crime_type"], orientation="h",
        marker=dict(color=colours, line=dict(color=PALETTE["bg"], width=0.5)),
        text=bd["cases"], textposition="outside",
        customdata=np.stack([bd["avg_severity"], bd["cyber_enabled"],
                             bd["minor_victim"]], axis=-1),
        hovertemplate="%{y}<br>%{x} case(s) · avg severity %{customdata[0]}"
                      "<br>cyber-enabled: %{customdata[1]}"
                      "<br>minor victim: %{customdata[2]}<extra></extra>",
    ))
    return _base_layout(fig, title="Division caseload by offence (cyan = cyber-enabled)",
                        height=max(280, len(bd) * 30))


def ws_band_donut(result) -> go.Figure:
    """WSTI band composition across the division's actors."""
    ws = getattr(result, "wsti", None)
    if ws is None or not len(ws):
        return _base_layout(go.Figure(), title="WSTI bands")
    order = ["Severe", "High", "Moderate", "Low"]
    counts = ws["band"].value_counts()
    labels = [b for b in order if counts.get(b, 0)]
    values = [int(counts[b]) for b in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=[_WS_BAND_COLOURS[b] for b in labels],
                    line=dict(color=PALETTE["bg"], width=2)),
        textinfo="value", textfont=dict(size=12, color=PALETTE["text"]),
        hovertemplate="%{label}: %{value} actor(s) (%{percent})<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{len(ws)}</b><br>actors", showarrow=False,
                       font=dict(size=13, color=PALETTE["text"]))
    return _base_layout(fig, height=300, title="WSTI band composition", showlegend=True)


def corridor_bars(result) -> go.Figure:
    """Trafficking corridors ranked by composite risk."""
    cor = getattr(result, "corridors", None)
    if cor is None or not len(cor):
        return _base_layout(go.Figure(), title="Trafficking corridors")
    cor = cor.iloc[::-1]
    colours = [PALETTE["critical"] if m else PALETTE["high"] for m in cor["minor_victim"]]
    fig = go.Figure(go.Bar(
        x=cor["risk_score"], y=cor["corridor_id"], orientation="h",
        marker=dict(color=colours, line=dict(color=PALETTE["bg"], width=0.5)),
        text=[f"{v:.0f}" for v in cor["risk_score"]], textposition="outside",
        customdata=np.stack([cor["route"], cor["n_cities"], cor["n_actors"]], axis=-1),
        hovertemplate="%{y} · risk %{x:.0f}<br>%{customdata[0]}"
                      "<br>%{customdata[1]} cities · %{customdata[2]} actors<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 108])
    return _base_layout(fig, title="Trafficking corridors by risk (red = minor victim)",
                        height=max(240, len(cor) * 34))


# ──────────────────────────────────────────────────────────────────────────────
# Entity extraction (PS 26189 — entity-extraction requirement)
# ──────────────────────────────────────────────────────────────────────────────

def entity_type_summary(extraction) -> pd.DataFrame:
    """Per-type extraction tally: distinct entities, mentions, mean confidence.

    Kept next to the chart that consumes it so the aggregation and its rendering
    cannot drift apart, and exported because the Reports page tabulates the same
    numbers.
    """
    cols = ["entity_type", "label", "distinct", "mentions", "documents", "avg_confidence"]
    ents = getattr(extraction, "entities", None)
    if ents is None or not len(ents):
        return pd.DataFrame(columns=cols)
    rows = []
    for etype, grp in ents.groupby("entity_type", sort=False):
        rows.append({
            "entity_type": etype,
            "label": str(grp["label"].iloc[0]),
            "distinct": int(len(grp)),
            "mentions": int(grp["n_mentions"].sum()),
            "documents": int(grp["n_documents"].max()),
            "avg_confidence": round(float(grp["confidence"].mean()), 3),
        })
    return pd.DataFrame(rows, columns=cols).sort_values("distinct", ascending=False,
                                                        ignore_index=True)


def entity_type_bars(extraction) -> go.Figure:
    """Distinct entities recovered per type, coloured by the entity-type palette.

    This is the chart that answers the PS requirement *"extract entities such as
    people, locations, vehicles, phone numbers, organisations"* with a count.
    """
    s = entity_type_summary(extraction)
    if not len(s):
        return _base_layout(go.Figure(), title="Entities extracted by type")
    s = s.iloc[::-1]
    colours = [ENTITY_COLORS.get(t, PALETTE["accent"]) for t in s["entity_type"]]
    fig = go.Figure(go.Bar(
        x=s["distinct"], y=s["label"], orientation="h",
        marker=dict(color=colours, line=dict(color=PALETTE["bg"], width=0.5)),
        text=s["distinct"], textposition="outside",
        customdata=np.stack([s["mentions"], s["avg_confidence"]], axis=-1),
        hovertemplate="%{y}<br>%{x} distinct · %{customdata[0]} mention(s)"
                      "<br>avg confidence %{customdata[1]:.2f}<extra></extra>",
    ))
    return _base_layout(fig, title="Entities extracted by type",
                        height=max(300, len(s) * 28))


def entity_confidence_hist(extraction) -> go.Figure:
    """Confidence distribution of extracted mentions.

    Shows the analyst how much of the extraction is high-certainty pattern work
    (checksum-validated IMEIs, RTO plates) versus lower-confidence contextual
    inference that a human should verify before it reaches a dossier.
    """
    mentions = getattr(extraction, "mentions", None)
    if mentions is None or not len(mentions):
        return _base_layout(go.Figure(), title="Extraction confidence")
    fig = go.Figure(go.Histogram(
        x=mentions["confidence"], nbinsx=20,
        marker=dict(color=PALETTE["accent"], line=dict(color=PALETTE["bg"], width=0.5)),
        hovertemplate="confidence %{x:.2f}: %{y} mention(s)<extra></extra>",
    ))
    fig.update_xaxes(title_text="confidence", range=[0, 1.02])
    fig.update_yaxes(title_text="mentions")
    return _base_layout(fig, title="Extraction confidence distribution", height=290)


def source_type_bars(documents: pd.DataFrame) -> go.Figure:
    """Ingested narrative documents by intelligence source type.

    Evidence for the PS's *"collect and process data from multiple sources"*
    requirement: the analyst sees the corpus really is multi-source, not one
    FIR feed relabelled.
    """
    if documents is None or not len(documents) or "source_type" not in documents:
        return _base_layout(go.Figure(), title="Documents by source")
    counts = documents["source_type"].value_counts().sort_values()
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h",
        marker=dict(color=PALETTE["accent_2"], line=dict(color=PALETTE["bg"], width=0.5)),
        text=counts.values, textposition="outside",
        hovertemplate="%{y}: %{x} document(s)<extra></extra>",
    ))
    return _base_layout(fig, title="Narrative documents by source type",
                        height=max(250, len(counts) * 34))
