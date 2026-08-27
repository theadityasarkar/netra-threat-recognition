"""Overview page — the at-a-glance intelligence summary."""

from __future__ import annotations

import streamlit as st

from src.config import PALETTE
from src.ui.components import metric_cards, section_header
from src.visualization import charts
from src.visualization.network_viz import filter_graph, network_figure


def render(result) -> None:
    g = result
    stats = g.anomaly_stats

    # ── Headline metrics ───────────────────────────────────────────────────────
    # ``anomaly_stats`` counts *actors* by suspicion band, which is not the same thing
    # as the alert queue: one alert can cover a cluster, and the Women-Safety lens raises
    # alerts for actors that carry no structural suspicion at all. Labelling these cards
    # "Critical alerts" made the Dashboard contradict the Reports tab — 1 against 8 for
    # the same run — so they name the actors they actually count.
    inferred = g.graph.graph.get("inferred_edges", 0)
    metric_cards([
        {"label": "Actors", "value": g.n_actors, "accent": PALETTE["accent"]},
        {"label": "Relationships", "value": g.n_links, "accent": PALETTE["accent_2"],
         "delta": f"{inferred} inferred" if inferred else None},
        {"label": "Operational modules", "value": g.n_modules, "accent": PALETTE["low"],
         "delta": f"Q = {g.modularity:.3f}"},
        {"label": "Critical-risk actors", "value": stats.get("critical", 0),
         "accent": PALETTE["critical"], "delta": "by suspicion band"},
        {"label": "High-risk actors", "value": stats.get("high", 0),
         "accent": PALETTE["high"], "delta": "by suspicion band"},
        {"label": "GNN AUC", "value": f"{g.gnn.auc:.3f}", "accent": PALETTE["accent"],
         "delta": f"{g.gnn.epochs_run} epochs"},
    ])

    # ── Network snapshot + threat bands ────────────────────────────────────────
    left, right = st.columns([2.1, 1])
    with left:
        section_header("Network at a glance", "coloured by operational module")
        rank_by = g.centralities["pagerank"] if len(g.centralities) else None
        H = filter_graph(g.graph, partition=g.partition, max_nodes=600, rank_by=rank_by)
        fig = network_figure(H, g, layout="spring", color_mode="Operational Module",
                             size_metric="pagerank", label_top_n=8, height=470)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        if H.graph.get("truncated"):
            st.caption(f"Showing the {H.number_of_nodes()} most central actors "
                       f"({H.graph['truncated']} lower-ranked hidden for clarity).")

    with right:
        section_header("Threat bands")
        st.plotly_chart(charts.band_donut(g, title=None), width="stretch",
                        config={"displayModeBar": False})
        # Top disruption target callout.
        if len(g.priority):
            lead = g.priority.iloc[0]
            st.markdown(f"""
            <div style="background:{PALETTE['surface']};border:1px solid {PALETTE['border']};
                 border-radius:10px;padding:0.8rem 1rem;margin-top:0.3rem;">
              <div style="font-size:0.7rem;color:{PALETTE['text_dim']};text-transform:uppercase;
                   letter-spacing:0.08em;">Top disruption target</div>
              <div style="font-size:1.1rem;font-weight:800;color:{PALETTE['text']};margin-top:0.2rem;">
                {lead['name']}</div>
              <div style="font-size:0.8rem;color:{PALETTE['accent']};">
                {lead.get('role','—')} · disruption {lead.get('disruption_score',0):.0f}/100</div>
            </div>""", unsafe_allow_html=True)

    # ── Analytical strip ───────────────────────────────────────────────────────
    section_header("Distribution analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.risk_distribution_bar(g), width="stretch",
                        config={"displayModeBar": False})
        st.plotly_chart(charts.geographic_bars(g), width="stretch",
                        config={"displayModeBar": False})
    with c2:
        st.plotly_chart(charts.role_distribution(g), width="stretch",
                        config={"displayModeBar": False})
        st.plotly_chart(charts.relation_type_bars(g), width="stretch",
                        config={"displayModeBar": False})

    # ── Module threat ranking ──────────────────────────────────────────────────
    section_header("Operational modules", f"{g.n_modules} detected · ranked by threat")
    m1, m2 = st.columns([1, 1])
    with m1:
        st.plotly_chart(charts.community_threat_bars(g), width="stretch",
                        config={"displayModeBar": False})
    with m2:
        st.plotly_chart(charts.community_scatter(g), width="stretch",
                        config={"displayModeBar": False})

    # ── Analysis provenance ────────────────────────────────────────────────────
    with st.expander("⚙️ Analysis provenance & timings"):
        cols = st.columns(4)
        timings = g.timings
        stage_names = [("graph_build", "Graph build"), ("centrality", "Centrality"),
                       ("communities", "Communities"), ("gnn", "GNN training"),
                       ("key_players", "Key players"), ("anomaly", "Anomaly ensemble"),
                       ("total", "Total")]
        for i, (key, label) in enumerate(stage_names):
            if key in timings:
                cols[i % 4].metric(label, f"{timings[key]:.2f}s")
        st.caption(f"Analysis generated {g.generated_at}. "
                   f"All timings exclude Streamlit caching — a re-render is instant.")
