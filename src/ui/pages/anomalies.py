"""Anomalies page — suspicious actors, clusters and the AI signal breakdown."""

from __future__ import annotations

import streamlit as st

from src.config import PALETTE
from src.ui.components import actor_profile, empty_state, metric_cards, section_header
from src.visualization import charts


def render(result) -> None:
    g = result
    a = g.anomalies
    stats = g.anomaly_stats
    if not len(a):
        empty_state("No anomaly scores available.")
        return

    metric_cards([
        {"label": "Critical", "value": stats.get("critical", 0), "accent": PALETTE["critical"]},
        {"label": "High", "value": stats.get("high", 0), "accent": PALETTE["high"]},
        {"label": "Medium", "value": stats.get("medium", 0), "accent": PALETTE["medium"]},
        {"label": "Mean score", "value": f"{stats.get('mean_score', 0):.0f}", "accent": PALETTE["accent"]},
        {"label": "Suspicious clusters", "value": len(g.clusters), "accent": PALETTE["accent_2"]},
        {"label": "Rules fired", "value": sum(stats.get("rule_counts", {}).values()),
         "accent": PALETTE["low"]},
    ])

    # ── Distribution + signal composition ──────────────────────────────────────
    c1, c2 = st.columns([1, 1])
    with c1:
        section_header("Suspicion distribution")
        st.plotly_chart(charts.suspicion_histogram(g), width="stretch",
                        config={"displayModeBar": False})
    with c2:
        section_header("Signal composition", "how the ensemble built each score")
        st.plotly_chart(charts.anomaly_composition(g, top_n=12), width="stretch",
                        config={"displayModeBar": False})

    c3, c4 = st.columns([1.3, 1])
    with c3:
        section_header("Rule firing frequency")
        st.plotly_chart(charts.rule_frequency(g), width="stretch",
                        config={"displayModeBar": False})
    with c4:
        section_header("GNN behavioural map")
        color_by = st.radio("Colour embedding by", ["module", "suspicion"], horizontal=True,
                            label_visibility="collapsed")
        st.plotly_chart(charts.embedding_scatter(g, color_by=color_by),
                        width="stretch", config={"displayModeBar": False})

    # ── Top suspects with explanations ─────────────────────────────────────────
    section_header("Flagged actors", "select to see the evidence trail")
    flagged = a[a["band"].isin(["Critical", "High", "Medium"])]
    left, right = st.columns([1, 1.15])

    with left:
        top = flagged.head(40)
        options = [f"{row['name']} ({pid}) · {row['band']} {row['suspicion_score']:.0f}"
                   for pid, row in top.iterrows()]
        if options:
            pick = st.radio("Flagged actors", options, label_visibility="collapsed")
            sel_id = pick.split("(")[1].split(")")[0]
        else:
            sel_id = None
            empty_state("No actors above the medium threshold.")

    with right:
        if sel_id:
            actor_profile(g, sel_id)

    # ── Suspicious clusters ─────────────────────────────────────────────────────
    section_header("Suspicious clusters", "modules concentrating anomalous activity")
    if len(g.clusters):
        for _, row in g.clusters.head(6).iterrows():
            colour = row.get("band")
            band_colour = {"Critical": PALETTE["critical"], "High": PALETTE["high"],
                           "Medium": PALETTE["medium"], "Low": PALETTE["low"]}.get(colour, PALETTE["accent"])
            sig = " · ".join(row.get("signature_patterns") or []) or "mixed"
            with st.container():
                st.markdown(f"""
                <div style="background:{PALETTE['surface']};border:1px solid {PALETTE['border']};
                     border-left:4px solid {band_colour};border-radius:10px;padding:0.8rem 1rem;
                     margin-bottom:0.6rem;">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-weight:800;font-size:1.05rem;">{row['label']}</span>
                    <span class="mono" style="color:{band_colour};font-weight:700;">
                      {row['cluster_score']:.0f}/100 · {row.get('band','')}</span>
                  </div>
                  <div style="color:{PALETTE['text_dim']};font-size:0.82rem;margin-top:0.3rem;">
                    {int(row['size'])} members · {int(row['flagged_actors'])} flagged ·
                    lead: <b style="color:{PALETTE['text']}">{row.get('lead_actor','—')}</b> ·
                    {row['dominant_offence']} · {row.get('state_list','—')}</div>
                  <div style="color:{PALETTE['accent']};font-size:0.76rem;margin-top:0.3rem;">
                    Signature: {sig}</div>
                </div>""", unsafe_allow_html=True)
    else:
        empty_state("No suspicious clusters above threshold.")

    # ── Full anomaly table ─────────────────────────────────────────────────────
    with st.expander("📋 Full anomaly table"):
        show = a.copy()
        show["rules"] = show["rules"].apply(lambda r: ", ".join(r) if r else "—")
        show["module"] = show["module"].astype(int) + 1
        cols = ["name", "band", "suspicion_score", "rule_score", "iforest_score",
                "gnn_residual", "n_rules", "rules", "module", "degree", "prior_offences",
                "risk_category", "status"]
        cols = [c for c in cols if c in show.columns]
        st.dataframe(show[cols], width="stretch", height=420)
        st.download_button(
            "⬇ Download anomaly table (CSV)",
            data=a.drop(columns=["reasons"], errors="ignore").reset_index().to_csv(index=False).encode(),
            file_name="netra_anomalies.csv", mime="text/csv")
