"""Network Explorer page — the interactive graph canvas with filters and drill-down."""

from __future__ import annotations

import streamlit as st

from src.config import PALETTE, RELATION_TYPES, RISK_CATEGORIES
from src.ui.components import actor_profile, empty_state, legend_row, section_header
from src.visualization.network_viz import (
    COLOR_MODES,
    LAYOUTS,
    SIZE_METRICS,
    ego_figure,
    filter_graph,
    network_figure,
)


def render(result) -> None:
    g = result

    # ── Controls ───────────────────────────────────────────────────────────────
    section_header("Network Explorer", "filter, re-layout and drill into any actor")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        layout_label = st.selectbox("Layout", list(LAYOUTS.keys()), index=0)
        layout = LAYOUTS[layout_label]
    with c2:
        color_mode = st.selectbox("Colour by", COLOR_MODES, index=0)
    with c3:
        size_label = st.selectbox("Size by", list(SIZE_METRICS.keys()), index=0)
        size_metric = SIZE_METRICS[size_label]
    with c4:
        modules_present = sorted(set(g.partition.values()))
        module_labels = {f"Module {m + 1}": m for m in modules_present}
        picked = st.multiselect("Modules", list(module_labels.keys()),
                                default=list(module_labels.keys())[:min(8, len(module_labels))])
        module_filter = [module_labels[p] for p in picked] if picked else None

    f1, f2, f3, f4 = st.columns([2, 2, 1.4, 1.4])
    with f1:
        rel_filter = st.multiselect("Relationship types", RELATION_TYPES, default=[])
    with f2:
        risk_filter = st.multiselect("Risk categories", RISK_CATEGORIES, default=[])
    with f3:
        min_strength = st.slider("Min link strength", 0.0, 1.0, 0.0, 0.05)
    with f4:
        show_cross = st.toggle("Highlight cross-module", value=True)
        largest_only = st.toggle("Largest component", value=False)

    # ── Build filtered subgraph ────────────────────────────────────────────────
    rank_metric = size_metric if (size_metric and size_metric in g.centralities.columns) else "pagerank"
    rank_by = g.centralities[rank_metric] if rank_metric in g.centralities.columns else None
    H = filter_graph(
        g.graph,
        relation_types=rel_filter or None,
        min_strength=min_strength,
        risk_categories=risk_filter or None,
        modules=module_filter,
        partition=g.partition,
        largest_component_only=largest_only,
        rank_by=rank_by,
    )

    # ── Selection state ────────────────────────────────────────────────────────
    if "selected_actor" not in st.session_state:
        st.session_state.selected_actor = None

    canvas, sidebar = st.columns([2.4, 1])

    with canvas:
        if H.number_of_nodes() == 0:
            empty_state("No actors match the current filters. Loosen them to see the network.")
        else:
            highlight = {st.session_state.selected_actor} if st.session_state.selected_actor in H else None
            fig = network_figure(
                H, g, layout=layout, color_mode=color_mode, size_metric=size_metric,
                highlight=highlight, label_top_n=12, show_cross_module=show_cross, height=640)
            st.plotly_chart(fig, width="stretch",
                            config={"displayModeBar": True, "scrollZoom": True,
                                    "displaylogo": False})
            meta = f"{H.number_of_nodes()} actors · {H.number_of_edges()} links shown"
            if H.graph.get("truncated"):
                meta += f" · {H.graph['truncated']} lower-ranked actors hidden"
            st.caption(meta)
            if color_mode == "Operational Module":
                legend_row([("Internal link", PALETTE["edge"]),
                            ("Cross-module link", PALETTE["high"])])

    with sidebar:
        section_header("Inspect actor")
        options = ["—"] + [g.label(n) for n in sorted(
            H.nodes(), key=lambda n: g.centralities["pagerank"].get(n, 0), reverse=True)]
        # Preserve selection across reruns.
        default_idx = 0
        if st.session_state.selected_actor:
            lbl = g.label(st.session_state.selected_actor)
            if lbl in options:
                default_idx = options.index(lbl)
        pick = st.selectbox("Select from visible actors", options, index=default_idx,
                            label_visibility="collapsed")
        if pick != "—":
            pid = pick.split("(")[-1].rstrip(")")
            st.session_state.selected_actor = pid
            actor_profile(g, pid)

            depth = st.slider("Ego-network depth", 1, 3, 1, key="ego_depth")
            with st.expander("🔗 Neighbourhood view", expanded=True):
                ego = ego_figure(g.graph, g, pid, radius=depth, height=380)
                st.plotly_chart(ego, width="stretch",
                                config={"displayModeBar": False})

            # Structural twins from the GNN embedding.
            from src.analysis.gnn import similar_actors
            twins = similar_actors(g.gnn, pid, top_n=5, exclude_neighbours_of=g.graph)
            if len(twins):
                st.markdown("**Structural twins** (GNN embedding)")
                st.caption("Actors who play a similar structural role — succession candidates.")
                for _, tw in twins.iterrows():
                    tname = g.graph.nodes[tw["person_id"]].get("name", tw["person_id"])
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:0.3rem 0.5rem;border-bottom:1px solid {PALETTE["border"]};">'
                        f'<span>{tname} <span class="mono" style="color:{PALETTE["accent"]}">'
                        f'{tw["person_id"]}</span></span>'
                        f'<span class="mono" style="color:{PALETTE["text_dim"]}">'
                        f'{tw["similarity"]:.2f}</span></div>', unsafe_allow_html=True)
        else:
            empty_state("Pick an actor to see their profile, neighbourhood and structural twins.",
                        icon="🔍")
