"""Reports & Alerts page — the alert queue and export controls."""

from __future__ import annotations

import re

import streamlit as st

from src.config import APP_NAME, PALETTE
from src.reporting.alerts import alert_counts, build_alerts
from src.reporting.report import (
    build_csv_bundle,
    build_cypher_export,
    build_graphml_export,
    build_markdown_brief,
    build_pdf_dossier,
    generate_ai_intelligence_brief,
)
from src.ui.components import alert_card, metric_cards, section_header


def _slug(text: str) -> str:
    """Filesystem-safe lowercase slug for download filenames."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", (text or "case").strip()).strip("_").lower()
    return s or "case"


@st.cache_data(show_spinner=False)
def _cached_pdf(_result, case_name: str, sig: str) -> bytes | None:
    """Cache the PDF by a cheap signature so repeated downloads don't rebuild it."""
    alerts = build_alerts(_result)
    return build_pdf_dossier(_result, alerts, case_name=case_name)


def render(result) -> None:
    g = result
    alerts = build_alerts(g)
    counts = alert_counts(alerts)

    section_header("Reports & Alerts", "actionable intelligence, ready to export")

    # ── AI Executive Intelligence Briefing ─────────────────────────────────────
    ai_brief = generate_ai_intelligence_brief(g, alerts, case_name="Operation Sentinel")
    with st.expander("🤖 AI Executive Intelligence Briefing (MHA / NCRB Standard Format)", expanded=True):
        st.markdown(ai_brief)

    # ── Export controls ────────────────────────────────────────────────────────
    ctrl1, ctrl2 = st.columns([1.5, 3.5])
    with ctrl1:
        case_name = st.text_input("Operation / case name", value="Operation Sentinel")
    with ctrl2:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        b1, b2, b3, b4, b5 = st.columns(5)

    sig = f"{g.generated_at}|{g.n_actors}|{g.n_links}|{counts['total']}|{case_name}"

    with b1:
        md = build_markdown_brief(g, alerts, case_name=case_name)
        st.download_button("📄 Markdown", data=md.encode(),
                           file_name=f"{APP_NAME}_brief_{_slug(case_name)}.md",
                           mime="text/markdown", width="stretch")
    with b2:
        with st.spinner("Rendering PDF…"):
            pdf = _cached_pdf(g, case_name, sig)
        if pdf:
            st.download_button("📕 PDF Dossier", data=pdf,
                               file_name=f"{APP_NAME}_dossier_{_slug(case_name)}.pdf",
                               mime="application/pdf", width="stretch")
        else:
            st.button("📕 PDF (install reportlab)", disabled=True, width="stretch")
    with b3:
        bundle = build_csv_bundle(g, alerts)
        st.download_button("🗜 CSV Bundle", data=bundle,
                           file_name=f"{APP_NAME}_data_{_slug(case_name)}.zip",
                           mime="application/zip", width="stretch")
    with b4:
        cypher = build_cypher_export(g, case_name=case_name)
        st.download_button("🌐 Neo4j Cypher", data=cypher.encode(),
                           file_name=f"{APP_NAME}_graph_{_slug(case_name)}.cql",
                           mime="text/plain", width="stretch")
    with b5:
        graphml = build_graphml_export(g)
        st.download_button("🕸️ GraphML", data=graphml.encode(),
                           file_name=f"{APP_NAME}_network_{_slug(case_name)}.graphml",
                           mime="application/xml", width="stretch")

    # ── Alert summary ──────────────────────────────────────────────────────────
    metric_cards([
        {"label": "Total alerts", "value": counts["total"], "accent": PALETTE["accent"]},
        {"label": "Critical", "value": counts["Critical"], "accent": PALETTE["critical"]},
        {"label": "High", "value": counts["High"], "accent": PALETTE["high"]},
        {"label": "Medium", "value": counts["Medium"], "accent": PALETTE["medium"]},
    ])

    # ── Alert queue ────────────────────────────────────────────────────────────
    section_header("Priority alert queue", f"{counts['total']} alerts · most severe first")

    categories = sorted({a.category for a in alerts})
    severities = ["Critical", "High", "Medium", "Low"]
    fc1, fc2 = st.columns(2)
    with fc1:
        cat_filter = st.multiselect("Filter by category", categories, default=[])
    with fc2:
        sev_filter = st.multiselect("Filter by severity", severities, default=[])

    shown = [a for a in alerts
             if (not cat_filter or a.category in cat_filter)
             and (not sev_filter or a.severity in sev_filter)]

    if not shown:
        st.info("No alerts match the current filter.")
    for a in shown:
        alert_card(a)
