"""Key Players page — influence ranking, role hierarchy and arrest priority."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import PALETTE
from src.ui.components import metric_cards, section_header
from src.visualization import charts

# Diverging + sequential colour stops for the table heatmaps. Kept matplotlib-free
# (pandas' background_gradient needs matplotlib) so the stack stays light.
_RYG = ["#1a9850", "#f6d032", "#d61f3f"]          # green → amber → red  (≈ RdYlGn_r)
_BLU = ["#12314f", "#0072b5", "#38bdf8"]          # deep blue → accent → bright cyan


def _hx(h: str) -> tuple[int, int, int]:
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def _heat_css(series: pd.Series, stops: list[str]) -> list[str]:
    """Map each value to an ``rgba`` background across ``stops`` (low→high).

    Alpha scales with magnitude so high values pop and low values recede — a clean
    heatmap on the dark theme without any matplotlib colormap dependency.
    """
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    rng = (hi - lo) or 1.0
    rgb_stops = [_hx(s) for s in stops]
    segs = len(stops) - 1
    css: list[str] = []
    for x in v:
        t = 0.0 if np.isnan(x) else (x - lo) / rng
        pos = t * segs
        i = min(int(pos), segs - 1)
        f = pos - i
        r, g, b = (round(a + (c - a) * f) for a, c in zip(rgb_stops[i], rgb_stops[i + 1]))
        alpha = 0.28 + 0.60 * t
        css.append(f"background-color: rgba({r},{g},{b},{alpha:.2f}); color: #f0f4ff;")
    return css


def render(result) -> None:
    g = result
    kp = g.key_players
    if not len(kp):
        st.warning("No key players computed.")
        return

    roles = kp["role"].value_counts() if "role" in kp.columns else pd.Series(dtype=int)
    metric_cards([
        {"label": "Kingpins", "value": int(roles.get("Kingpin", 0)), "accent": PALETTE["critical"]},
        {"label": "Lieutenants", "value": int(roles.get("Lieutenant", 0)), "accent": PALETTE["high"]},
        {"label": "Brokers", "value": int(roles.get("Broker", 0)), "accent": PALETTE["accent"]},
        {"label": "Financiers", "value": int(roles.get("Financier", 0)), "accent": PALETTE["accent_2"]},
        {"label": "Enforcers", "value": int(roles.get("Enforcer", 0)), "accent": "#f472b6"},
        {"label": "Analysed", "value": len(kp), "accent": PALETTE["low"]},
    ])

    # ── Influence ranking + broker/hub map ─────────────────────────────────────
    c1, c2 = st.columns([1, 1])
    with c1:
        section_header("Influence ranking", "6-metric weighted index")
        st.plotly_chart(charts.influence_ranking(g, top_n=15), width="stretch",
                        config={"displayModeBar": False})
    with c2:
        section_header("Broker vs hub map", "high betweenness + low PageRank = broker")
        st.plotly_chart(charts.centrality_scatter(g), width="stretch",
                        config={"displayModeBar": False})

    # ── Arrest priority ────────────────────────────────────────────────────────
    section_header("Disruption / arrest priority",
                   "ranked by measured network damage on removal")
    st.plotly_chart(charts.disruption_ranking(g, top_n=10), width="stretch",
                    config={"displayModeBar": False})

    st.caption("💡 Note how the top disruption targets are often **brokers and gateways**, not the "
               "highest-influence actors — removing a low-profile connector fragments the network "
               "more than arresting a well-connected but replaceable hub.")

    if len(g.priority):
        show = g.priority.copy()
        show.index.name = "ID"
        rename = {"name": "Actor", "role": "Role", "city": "City", "state": "State",
                  "status": "Status", "risk_category": "Risk", "influence_score": "Influence",
                  "disruption_score": "Disruption", "new_fragments": "+Fragments",
                  "lcc_drop_pct": "Cluster drop %", "degree": "Degree", "cases": "Cases"}
        show = show.rename(columns={k: v for k, v in rename.items() if k in show.columns})
        cols = [c for c in ["Actor", "Role", "City", "State", "Status", "Risk", "Influence",
                            "Disruption", "+Fragments", "Cluster drop %", "Degree", "Cases"]
                if c in show.columns]
        st.dataframe(
            show[cols].style.apply(
                lambda s: _heat_css(s, _RYG),
                subset=[c for c in ["Disruption", "Influence"] if c in cols]),
            width="stretch", height=430)

    # ── Full ranked table ──────────────────────────────────────────────────────
    section_header("Full key-player table")
    display_cols = {
        "name": "Actor", "role": "Role", "influence_score": "Influence",
        "influence_rank": "Rank", "pagerank": "PageRank", "betweenness": "Betweenness",
        "eigenvector": "Eigenvector", "degree": "Degree", "core_number": "k-core",
        "module": "Module", "risk_category": "Risk", "status": "Status", "cases": "Cases",
    }
    tbl = kp.copy()
    if "module" in tbl.columns:
        tbl["module"] = tbl["module"].fillna(-1).astype(int) + 1
    tbl = tbl.rename(columns={k: v for k, v in display_cols.items() if k in tbl.columns})
    ordered = [v for v in display_cols.values() if v in tbl.columns]
    tbl.index.name = "ID"

    role_filter = st.multiselect(
        "Filter by role", sorted(kp["role"].unique()) if "role" in kp.columns else [], default=[])
    view = tbl
    if role_filter:
        view = tbl[tbl["Role"].isin(role_filter)]
    st.dataframe(
        view[ordered].style.format({
            "Influence": "{:.1f}", "PageRank": "{:.4f}", "Betweenness": "{:.4f}",
            "Eigenvector": "{:.4f}",
        }).apply(lambda s: _heat_css(s, _BLU), subset=["Influence"]),
        width="stretch", height=460)

    st.download_button(
        "⬇ Download key-player table (CSV)",
        data=kp.reset_index().to_csv(index=False).encode(),
        file_name="netra_key_players.csv", mime="text/csv")
