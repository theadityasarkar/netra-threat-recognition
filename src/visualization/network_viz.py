"""
Layer 4a — interactive network canvas (Plotly).

Plotly is used rather than PyVis because it renders inside Streamlit without an iframe
round-trip, keeps a single visual language with the rest of the dashboard, and exports
cleanly to static images for the PDF report.

Design decisions that matter for a graph this dense:

  • Edges are drawn as **one** Scatter trace with ``None`` separators. Drawing 1,000
    individual traces would make the browser crawl; this keeps it to a single draw call.
  • Edges are split into two traces (normal / highlighted) so cross-module links can be
    emphasised without per-edge traces.
  • Labels are shown only for the top-N actors by the active size metric — labelling
    every node is illegible and slow.
  • Layouts are seeded, so the graph does not jump between reruns.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from src.config import (
    COMMUNITY_COLORS,
    PALETTE,
    PERF,
    RISK_COLORS,
    ROLE_COLORS,
    STATUS_COLORS,
)

LAYOUTS = {
    "Force-Directed (Spring)": "spring",
    "Kamada-Kawai (Structural)": "kamada",
    "Community Clustered": "clustered",
    "Circular": "circular",
    "GNN Embedding Space": "embedding",
}

COLOR_MODES = ["Operational Module", "Inferred Role", "Risk Category",
               "Suspicion Score", "Status"]

SIZE_METRICS = {
    "PageRank": "pagerank",
    "Betweenness": "betweenness",
    "Degree": "degree",
    "Eigenvector": "eigenvector",
    "Influence Score": "influence_score",
    "Suspicion Score": "suspicion_score",
    "Uniform": None,
}


# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────

def compute_layout(
    G: nx.Graph,
    kind: str = "spring",
    partition: dict[str, int] | None = None,
    embedding_coords: dict[str, tuple[float, float]] | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Node positions for the chosen layout. Always seeded for stability."""
    if G.number_of_nodes() == 0:
        return {}
    if G.number_of_nodes() == 1:
        return {next(iter(G.nodes())): np.array([0.0, 0.0])}

    if kind == "embedding" and embedding_coords:
        return {n: np.asarray(embedding_coords.get(n, (0.0, 0.0)), dtype=float) for n in G}

    if kind == "circular":
        return nx.circular_layout(G)

    if kind == "kamada":
        # Kamada-Kawai is O(V²) in memory; fall back on large graphs.
        if G.number_of_nodes() <= 700:
            try:
                return nx.kamada_kawai_layout(G, weight="distance")
            except Exception:                               # noqa: BLE001
                pass
        return nx.spring_layout(G, seed=seed, weight="weight", k=1.2 / np.sqrt(len(G)))

    if kind == "clustered" and partition:
        # Place each module on a ring, then run a short local spring inside each so
        # modules read as visually distinct territories.
        modules: dict[int, list[str]] = {}
        for node, cid in partition.items():
            if node in G:
                modules.setdefault(cid, []).append(node)

        pos: dict[str, np.ndarray] = {}
        n_mod = max(1, len(modules))
        radius = 1.0 + 0.28 * np.sqrt(n_mod)
        for i, (cid, members) in enumerate(sorted(modules.items())):
            angle = 2 * np.pi * i / n_mod
            centre = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            sub = G.subgraph(members)
            spread = 0.18 + 0.5 * (len(members) / max(1, len(G))) ** 0.5
            if len(members) == 1:
                pos[members[0]] = centre
                continue
            local = nx.spring_layout(sub, seed=seed, k=0.9 / np.sqrt(len(members)),
                                     iterations=40)
            for node, p in local.items():
                pos[node] = centre + spread * np.asarray(p)
        for node in G:
            pos.setdefault(node, np.array([0.0, 0.0]))
        return pos

    iterations = 120 if G.number_of_nodes() <= 400 else 60
    return nx.spring_layout(G, seed=seed, weight="weight", iterations=iterations,
                            k=1.1 / np.sqrt(len(G)))


# ──────────────────────────────────────────────────────────────────────────────
# Colour resolution
# ──────────────────────────────────────────────────────────────────────────────

def _node_colours(G: nx.Graph, nodes: list[str], mode: str, result) -> tuple[list, dict]:
    """Return per-node colours plus a legend mapping ``{label: colour}``."""
    if mode == "Operational Module":
        colours, legend = [], {}
        for n in nodes:
            cid = result.partition.get(n, -1)
            c = COMMUNITY_COLORS[cid % len(COMMUNITY_COLORS)] if cid >= 0 else PALETTE["text_dim"]
            colours.append(c)
            # Only the modules big enough to matter earn a legend entry.
            if cid >= 0:
                legend[f"Module {cid + 1}"] = c
        top = dict(sorted(legend.items(), key=lambda kv: int(kv[0].split()[-1]))[:10])
        return colours, top

    if mode == "Inferred Role":
        roles = result.key_players["role"] if "role" in result.key_players.columns else {}
        colours = [ROLE_COLORS.get(roles.get(n, "Peripheral"), PALETTE["text_dim"])
                   for n in nodes]
        present = {roles.get(n, "Peripheral") for n in nodes}
        return colours, {r: c for r, c in ROLE_COLORS.items() if r in present}

    if mode == "Risk Category":
        colours = [RISK_COLORS.get(G.nodes[n].get("risk_category", "Low"),
                                   PALETTE["text_dim"]) for n in nodes]
        present = {G.nodes[n].get("risk_category", "Low") for n in nodes}
        return colours, {r: c for r, c in RISK_COLORS.items() if r in present}

    if mode == "Status":
        colours = [STATUS_COLORS.get(G.nodes[n].get("status", "Active"),
                                     PALETTE["text_dim"]) for n in nodes]
        present = {G.nodes[n].get("status", "Active") for n in nodes}
        return colours, {s: c for s, c in STATUS_COLORS.items() if s in present}

    # Suspicion Score — continuous, so hand back numbers and let Plotly's colourbar
    # do the work (signalled by an empty legend).
    scores = result.anomalies["suspicion_score"] if len(result.anomalies) else {}
    return [float(scores.get(n, 0.0)) for n in nodes], {}


def _hover_text(G: nx.Graph, node: str, result) -> str:
    a = G.nodes[node]
    role = result.key_players["role"].get(node, "—") if len(result.key_players) else "—"
    infl = result.key_players["influence_score"].get(node) if len(result.key_players) else None
    susp = result.anomalies["suspicion_score"].get(node) if len(result.anomalies) else None
    cent = result.centralities.loc[node] if node in result.centralities.index else None

    lines = [
        f"<b>{a.get('name', node)}</b>  <span style='color:{PALETTE['text_dim']}'>({node})</span>",
        f"Role: <b>{role}</b> · Module {result.partition.get(node, -1) + 1}",
        f"{a.get('city', '—')}, {a.get('state', '—')} · {a.get('status', '—')}",
        f"Risk: <b>{a.get('risk_category', '—')}</b> · "
        f"{a.get('prior_offences', 0)} prior offence(s) · {len(a.get('case_ids') or [])} case(s)",
    ]
    if a.get("alias_list"):
        lines.append(f"Alias: {'; '.join(a['alias_list'][:2])}")
    if cent is not None:
        lines.append(
            f"Degree {int(cent['degree'])} · PageRank {cent['pagerank']:.4f} · "
            f"Betweenness {cent['betweenness']:.4f}")
    if infl is not None:
        lines.append(f"Influence <b>{infl:.1f}</b>" +
                     (f" · Suspicion <b>{susp:.1f}</b>" if susp is not None else ""))
    if a.get("offence_list"):
        lines.append(f"Offences: {', '.join(a['offence_list'][:3])}")
    return "<br>".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main figure
# ──────────────────────────────────────────────────────────────────────────────

def network_figure(
    G: nx.Graph,
    result,
    layout: str = "spring",
    color_mode: str = "Operational Module",
    size_metric: str | None = "pagerank",
    highlight: set[str] | None = None,
    label_top_n: int = 12,
    show_cross_module: bool = True,
    height: int = 680,
    title: str | None = None,
) -> go.Figure:
    """Render the criminal network.

    Args:
        highlight: actor IDs to ring in white (selection / alert focus).
        label_top_n: how many of the biggest nodes get a permanent text label.
        show_cross_module: draw inter-module links in a distinct warning colour.
    """
    fig = go.Figure()
    if G.number_of_nodes() == 0:
        fig.add_annotation(text="No actors match the current filters",
                           showarrow=False, font=dict(color=PALETTE["text_dim"], size=15))
        fig.update_layout(height=height, paper_bgcolor=PALETTE["bg"],
                          plot_bgcolor=PALETTE["bg"])
        return fig

    nodes = list(G.nodes())
    embedding_coords = None
    if layout == "embedding" and len(result.gnn.node_ids):
        coords = result.gnn.coords_2d
        embedding_coords = {pid: (float(coords[i, 0]), float(coords[i, 1]))
                            for i, pid in enumerate(result.gnn.node_ids)}

    pos = compute_layout(G, layout, result.partition, embedding_coords)

    # ── Edges: two batched traces (intra-module, cross-module) ────────────────
    intra_x: list[float | None] = []
    intra_y: list[float | None] = []
    cross_x: list[float | None] = []
    cross_y: list[float | None] = []

    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        is_cross = result.partition.get(u) != result.partition.get(v)
        if is_cross and show_cross_module:
            cross_x += [x0, x1, None]
            cross_y += [y0, y1, None]
        else:
            intra_x += [x0, x1, None]
            intra_y += [y0, y1, None]

    fig.add_trace(go.Scatter(
        x=intra_x, y=intra_y, mode="lines", hoverinfo="skip",
        line=dict(color=PALETTE["edge"], width=0.8),
        opacity=0.55, name="Internal link", showlegend=False))

    if cross_x:
        fig.add_trace(go.Scatter(
            x=cross_x, y=cross_y, mode="lines", hoverinfo="skip",
            line=dict(color=PALETTE["high"], width=1.5),
            opacity=0.85, name="Cross-module link", showlegend=True))

    # ── Node sizing ───────────────────────────────────────────────────────────
    if size_metric:
        if size_metric in result.centralities.columns:
            raw = result.centralities[size_metric].reindex(nodes).fillna(0.0)
        elif len(result.key_players) and size_metric in result.key_players.columns:
            raw = result.key_players[size_metric].reindex(nodes).fillna(0.0)
        elif len(result.anomalies) and size_metric in result.anomalies.columns:
            raw = result.anomalies[size_metric].reindex(nodes).fillna(0.0)
        else:
            raw = None
    else:
        raw = None

    if raw is not None and float(raw.max()) > float(raw.min()):
        # Square-root scaling: node *area* should track the metric, not the radius,
        # otherwise a 10× PageRank looks like a 100× difference.
        norm = (raw - raw.min()) / (raw.max() - raw.min())
        sizes = 8.0 + 26.0 * np.sqrt(norm.to_numpy())
    else:
        sizes = np.full(len(nodes), 12.0)

    colours, legend = _node_colours(G, nodes, color_mode, result)
    continuous = not legend

    highlight = highlight or set()
    line_widths = [2.6 if n in highlight else 0.7 for n in nodes]
    line_colours = ["#ffffff" if n in highlight else PALETTE["bg"] for n in nodes]

    # ── Labels for the most important nodes only ──────────────────────────────
    if raw is not None and label_top_n > 0:
        top_ids = set(raw.nlargest(min(label_top_n, len(raw))).index)
    else:
        top_ids = set()
    top_ids |= highlight
    text = [G.nodes[n].get("name", n).split()[0] if n in top_ids else "" for n in nodes]

    marker = dict(
        size=sizes,
        color=colours,
        line=dict(width=line_widths, color=line_colours),
        opacity=0.94,
    )
    if continuous:
        marker.update(
            colorscale=[[0.0, PALETTE["low"]], [0.45, PALETTE["medium"]],
                        [0.7, PALETTE["high"]], [1.0, PALETTE["critical"]]],
            cmin=0, cmax=100,
            colorbar=dict(title=dict(text="Suspicion", side="right"),
                          thickness=12, len=0.55, x=1.01,
                          tickfont=dict(color=PALETTE["text_dim"], size=10)),
        )

    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes],
        y=[pos[n][1] for n in nodes],
        mode="markers+text" if top_ids else "markers",
        marker=marker,
        text=text,
        textposition="top center",
        textfont=dict(size=9, color=PALETTE["text"]),
        customdata=nodes,
        hovertext=[_hover_text(G, n, result) for n in nodes],
        hoverinfo="text",
        showlegend=False,
        name="Actors",
    ))

    # Discrete legend entries via invisible proxy traces.
    for label, colour in legend.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=label,
            marker=dict(size=9, color=colour), showlegend=True, hoverinfo="skip"))

    # Declare a title only when there is one — see ``charts._base_layout`` for why an
    # explicit ``title=None`` renders the word "undefined" over the graph.
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14, color=PALETTE["text"])))
    fig.update_layout(
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=PALETTE["text_dim"])),
        margin=dict(l=8, r=8, t=44 if title else 30, b=8),
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        hoverlabel=dict(bgcolor=PALETTE["surface_2"], bordercolor=PALETTE["border"],
                        font=dict(color=PALETTE["text"], size=11), align="left"),
        dragmode="pan",
    )
    return fig


def ego_figure(
    G: nx.Graph,
    result,
    center: str,
    radius: int = 1,
    height: int = 520,
) -> go.Figure:
    """Focused neighbourhood view: who this actor touches, and how."""
    if center not in G:
        return network_figure(nx.Graph(), result, height=height)

    ego = nx.ego_graph(G, center, radius=radius, undirected=True)
    fig = network_figure(
        ego, result, layout="kamada" if ego.number_of_nodes() <= 120 else "spring",
        color_mode="Inferred Role", size_metric="pagerank",
        highlight={center}, label_top_n=min(18, ego.number_of_nodes()),
        show_cross_module=True, height=height,
        title=f"Ego network · {G.nodes[center].get('name', center)} · "
              f"depth {radius} · {ego.number_of_nodes() - 1} contacts",
    )
    return fig


def filter_graph(
    G: nx.Graph,
    relation_types: list[str] | None = None,
    min_strength: float = 0.0,
    risk_categories: list[str] | None = None,
    modules: list[int] | None = None,
    partition: dict[str, int] | None = None,
    largest_component_only: bool = False,
    max_nodes: int | None = None,
    rank_by: "object" = None,
) -> nx.Graph:
    """Apply the sidebar filters, returning a subgraph.

    ``max_nodes`` caps the render for responsiveness; when it bites, the *most
    important* nodes by ``rank_by`` are kept rather than an arbitrary slice, and the
    result records ``graph["truncated"]`` so the UI can say so out loud.
    """
    H = G.copy()

    if relation_types:
        keep = set(relation_types)
        drop = [(u, v) for u, v, d in H.edges(data=True)
                if not (set(d.get("relation_types") or [d.get("relation_type")]) & keep)]
        H.remove_edges_from(drop)

    if min_strength > 0:
        H.remove_edges_from([(u, v) for u, v, d in H.edges(data=True)
                             if d.get("strength", 0) < min_strength])

    if risk_categories:
        keep = set(risk_categories)
        H.remove_nodes_from([n for n in list(H.nodes())
                             if H.nodes[n].get("risk_category", "Low") not in keep])

    if modules is not None and partition:
        keep = set(modules)
        H.remove_nodes_from([n for n in list(H.nodes()) if partition.get(n) not in keep])

    if largest_component_only and H.number_of_nodes():
        comps = list(nx.connected_components(H))
        if comps:
            H = H.subgraph(max(comps, key=len)).copy()

    truncated = 0
    cap = max_nodes or PERF["max_render_nodes"]
    if H.number_of_nodes() > cap:
        if rank_by is not None:
            ranked = rank_by.reindex(list(H.nodes())).fillna(0.0).nlargest(cap).index
        else:
            ranked = [n for n, _ in sorted(H.degree(), key=lambda kv: -kv[1])[:cap]]
        truncated = H.number_of_nodes() - cap
        H = H.subgraph(list(ranked)).copy()

    H.graph["truncated"] = truncated
    return H
