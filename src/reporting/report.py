"""
Layer 5b — report generation.

Three export formats, one source of truth (:class:`~src.pipeline.AnalysisResult`):

  • ``build_markdown_brief``  — a plain-text intelligence brief (fast, diff-able, and
    the fallback when ReportLab is unavailable).
  • ``build_pdf_dossier``     — a formatted PDF via ReportLab, suitable for circulation.
  • ``build_csv_bundle``      — a ZIP of every analysis frame for downstream tooling.

Nothing here imports Streamlit; ``app.py`` calls these and hands the bytes to
``st.download_button``.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from xml.dom import minidom

import pandas as pd

from src.config import (
    APP_FULL_NAME,
    APP_NAME,
    APP_ORG,
    APP_VERSION,
    CLASSIFICATION_BANNER,
    PROBLEM_STATEMENT,
)
from src.reporting.alerts import Alert, alert_counts, alerts_to_frame, build_alerts


# ──────────────────────────────────────────────────────────────────────────────
# Markdown brief
# ──────────────────────────────────────────────────────────────────────────────

def build_markdown_brief(result, alerts: list[Alert] | None = None,
                         case_name: str = "Untitled Operation") -> str:
    """A self-contained Markdown intelligence brief."""
    alerts = alerts if alerts is not None else build_alerts(result)
    counts = alert_counts(alerts)
    ts = result.generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    g = result

    lines: list[str] = []
    add = lines.append

    add(f"# {APP_NAME} — Criminal Network Intelligence Brief")
    add(f"**{APP_FULL_NAME}**  \n{APP_ORG}")
    add(f"\n> {CLASSIFICATION_BANNER}\n")
    add(f"- **Operation:** {case_name}")
    add(f"- **Generated:** {ts}")
    add(f"- **Engine version:** {APP_NAME} v{APP_VERSION}")
    add(f"- **Problem statement:** SIH 2026 · PS {PROBLEM_STATEMENT['id']} — "
        f"{PROBLEM_STATEMENT['title']}")
    add(f"- **Owning department:** {PROBLEM_STATEMENT['department']}")
    add("")

    # ── Executive summary ─────────────────────────────────────────────────────
    add("## 1. Executive Summary\n")
    add(f"The analysed network comprises **{g.n_actors} actors** connected by "
        f"**{g.n_links} relationships**, organised into **{g.n_modules} operational "
        f"modules** (modularity Q = {g.modularity:.3f}). "
        f"The AI ensemble raised **{counts['total']} alerts**: "
        f"{counts['Critical']} critical, {counts['High']} high, "
        f"{counts['Medium']} medium.\n")

    if len(g.priority):
        lead = g.priority.iloc[0]
        add(f"The highest-value disruption target is **{lead['name']}** "
            f"({lead.name}, role: {lead.get('role', '—')}), whose removal is projected to "
            f"cause the greatest structural fragmentation. ")
    if len(g.anomalies):
        top_an = g.anomalies.iloc[0]
        add(f"The most anomalous actor is **{top_an['name']}** "
            f"(suspicion {top_an['suspicion_score']:.0f}/100).\n")

    # ── Key players ───────────────────────────────────────────────────────────
    add("\n## 2. Key Players\n")
    if len(g.key_players):
        add("| Rank | Actor | ID | Role | Influence | Module | Risk |")
        add("|-----:|-------|----|------|----------:|-------:|------|")
        for i, (pid, row) in enumerate(g.key_players.head(10).iterrows(), 1):
            add(f"| {i} | {row['name']} | {pid} | {row.get('role', '—')} | "
                f"{row['influence_score']:.1f} | {int(row.get('module', 0)) + 1} | "
                f"{row.get('risk_category', '—')} |")
    add("")

    # ── Arrest priority ───────────────────────────────────────────────────────
    add("\n## 3. Disruption / Arrest Priority\n")
    add("*Ranked by projected network damage on removal (actors in custody excluded).*\n")
    if len(g.priority):
        add("| Rank | Actor | Role | Disruption | Influence | +Fragments | Main-cluster drop |")
        add("|-----:|-------|------|-----------:|----------:|-----------:|------------------:|")
        for i, (pid, row) in enumerate(g.priority.head(10).iterrows(), 1):
            add(f"| {i} | {row['name']} | {row.get('role', '—')} | "
                f"{row.get('disruption_score', 0):.1f} | {row.get('influence_score', 0):.0f} | "
                f"+{int(row.get('new_fragments', 0))} | {row.get('lcc_drop_pct', 0):.0f}% |")
    add("")

    # ── Suspicious clusters ───────────────────────────────────────────────────
    add("\n## 4. Suspicious Clusters\n")
    if len(g.clusters):
        for _, row in g.clusters.head(5).iterrows():
            add(f"### {row['label']} — score {row['cluster_score']:.0f}/100 ({row['band']})")
            add(f"- **Members:** {int(row['size'])} · **Flagged:** {int(row['flagged_actors'])} "
                f"· **Dominant offence:** {row['dominant_offence']}")
            add(f"- **Lead actor:** {row.get('lead_actor', '—')} · "
                f"**Geography:** {row.get('state_list', '—')}")
            if row.get("signature_patterns"):
                add(f"- **Signature:** {', '.join(row['signature_patterns'])}")
            add("")
    else:
        add("No suspicious clusters above threshold.\n")

    # ── Women Safety Division lens ────────────────────────────────────────────
    add("\n## 5. Women Safety Division Assessment\n")
    ws_stats = getattr(g, "ws_stats", {}) or {}
    if ws_stats.get("actors"):
        add(f"PS {PROBLEM_STATEMENT['id']} is owned by the {PROBLEM_STATEMENT['department']}. "
            f"Of the loaded caseload, **{ws_stats.get('cases', 0)} case(s)** "
            f"({ws_stats.get('case_share', 0.0):.1f}%) fall inside the division's mandate, "
            f"implicating **{ws_stats.get('actors', 0)} actors** — "
            f"{ws_stats.get('minor_victim_actors', 0)} tied to an offence against a minor and "
            f"{ws_stats.get('cyber_actors', 0)} to a cyber-enabled offence. "
            f"Peak WSTI {ws_stats.get('peak_wsti', 0.0):.0f}/100.\n")
        esc = getattr(g, "ws_escalation", None)
        if esc is not None and len(esc):
            add("**Escalation queue** — statutory aggravators promoted above raw score:\n")
            add("| Rank | Actor | ID | WSTI | Band | Division cases | Minor victim |")
            add("|-----:|-------|----|-----:|------|---------------:|:------------:|")
            for _, row in esc.head(8).iterrows():
                add(f"| {int(row.get('escalation_rank', 0))} | {row.get('name', '')} | "
                    f"{row.get('person_id', '')} | {float(row.get('wsti', 0)):.0f} | "
                    f"{row.get('band', '')} | {int(row.get('ws_cases', 0))} | "
                    f"{'⚠ yes' if row.get('minor_victim') else 'no'} |")
        cor = getattr(g, "corridors", None)
        if cor is not None and len(cor):
            add(f"\n**Trafficking corridors:** {len(cor)} route(s) inferred from case "
                "geography (verify against CDR tower and transport records).")
            for _, row in cor.head(4).iterrows():
                add(f"- `{row.get('corridor_id', '')}` — {row.get('route', '')} "
                    f"(risk {float(row.get('risk_score', 0)):.0f}, "
                    f"{int(row.get('n_actors', 0))} actors)")
    else:
        add("No case records in this dataset fall inside the Women Safety Division's "
            "mandate, so the WSTI was not computed.\n")

    # ── Alerts ────────────────────────────────────────────────────────────────
    add("\n## 6. Alert Queue\n")
    for a in alerts[:20]:
        add(f"### [{a.severity}] {a.title}")
        add(f"*{a.category} · score {a.score:.0f}*  \n{a.detail}\n")
        if a.recommended_action:
            add(f"**Recommended action:** {a.recommended_action}\n")
        if a.evidence:
            add("**Evidence:**")
            for e in a.evidence:
                add(f"- {e}")
        add("")

    # ── Methodology ───────────────────────────────────────────────────────────
    add("\n## 7. Methodology\n")
    add(f"- **Centrality suite:** degree, weighted degree, betweenness, closeness, "
        f"eigenvector, PageRank, k-core, clustering.")
    add(f"- **Community detection:** Louvain modularity (Q = {g.modularity:.3f}).")
    add(f"- **Graph neural network:** 2-layer GraphSAGE encoder, "
        f"held-out link-prediction AUC = {g.gnn.auc:.3f} over {g.gnn.epochs_run} epochs.")
    add(f"- **Anomaly ensemble:** GNN reconstruction residual + Isolation Forest + "
        f"8-rule expert engine, each score explainable.")
    add("\n---")
    add(f"\n*Generated by {APP_NAME} v{APP_VERSION}. All actor data is synthetic and for "
        "demonstration only.*")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CSV bundle
# ──────────────────────────────────────────────────────────────────────────────

def build_csv_bundle(result, alerts: list[Alert] | None = None) -> bytes:
    """ZIP archive of every analysis frame — for import into other tooling."""
    alerts = alerts if alerts is not None else build_alerts(result)
    buffer = io.BytesIO()

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        # List-typed cells (rules, reasons) become pipe-joined strings for CSV.
        out = df.copy()
        for col in out.columns:
            if out[col].apply(lambda v: isinstance(v, (list, tuple))).any():
                out[col] = out[col].apply(
                    lambda v: " | ".join(map(str, v)) if isinstance(v, (list, tuple)) else v)
        return out

    frames = {
        "actors.csv": result.persons,
        "relationships.csv": result.relationships,
        "centrality.csv": result.centralities.reset_index(),
        "key_players.csv": result.key_players.reset_index(),
        "arrest_priority.csv": result.priority.reset_index(),
        "anomalies.csv": result.anomalies.reset_index(),
        "suspicious_clusters.csv": result.clusters,
        "community_profiles.csv": result.profiles,
        "bridges.csv": result.bridges,
        "critical_links.csv": result.cut_edges,
        "alerts.csv": alerts_to_frame(alerts),
    }
    if len(result.crimes):
        frames["crime_records.csv"] = result.crimes

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, frame in frames.items():
            if frame is not None and len(frame):
                zf.writestr(name, _clean(frame).to_csv(index=False))
        zf.writestr("intelligence_brief.md", build_markdown_brief(result, alerts))

    buffer.seek(0)
    return buffer.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# PDF dossier (ReportLab)
# ──────────────────────────────────────────────────────────────────────────────

def build_pdf_dossier(result, alerts: list[Alert] | None = None,
                     case_name: str = "Untitled Operation",
                     figures: dict | None = None) -> bytes | None:
    """Formatted PDF intelligence dossier.

    Returns ``None`` if ReportLab is not installed, so the caller can fall back to the
    Markdown brief without crashing.

    Args:
        figures: optional ``{caption: png_bytes}`` of chart images to embed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm, mm
        from reportlab.platypus import (
            HRFlowable,
            Image,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except Exception:                                          # noqa: BLE001
        return None

    alerts = alerts if alerts is not None else build_alerts(result)
    counts = alert_counts(alerts)
    g = result
    ts = g.generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Palette — a printable variant of the on-screen theme (dark ink, cyan/red accents).
    INK = colors.HexColor("#0f1830")
    ACCENT = colors.HexColor("#0072b5")
    CRIT = colors.HexColor("#d61f3f")
    HIGH = colors.HexColor("#e07b00")
    MUTED = colors.HexColor("#5a6b82")
    LIGHT = colors.HexColor("#eef2f8")
    ROW_ALT = colors.HexColor("#f6f8fb")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("NetraTitle", parent=styles["Title"], textColor=INK,
                              fontSize=22, spaceAfter=2, leading=25))
    styles.add(ParagraphStyle("NetraSub", parent=styles["Normal"], textColor=ACCENT,
                              fontSize=11, spaceAfter=1, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle("NetraH2", parent=styles["Heading2"], textColor=INK,
                              fontSize=14, spaceBefore=14, spaceAfter=6))
    styles.add(ParagraphStyle("NetraH3", parent=styles["Heading3"], textColor=ACCENT,
                              fontSize=11.5, spaceBefore=8, spaceAfter=3))
    styles.add(ParagraphStyle("NetraBody", parent=styles["Normal"], textColor=INK,
                              fontSize=9.5, leading=14, alignment=TA_LEFT))
    styles.add(ParagraphStyle("NetraMuted", parent=styles["Normal"], textColor=MUTED,
                              fontSize=8, leading=11))
    styles.add(ParagraphStyle("NetraBanner", parent=styles["Normal"], textColor=colors.white,
                              fontSize=8, alignment=TA_CENTER, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle("Cell", parent=styles["Normal"], textColor=INK,
                              fontSize=8, leading=10))
    styles.add(ParagraphStyle("CellHead", parent=styles["Normal"], textColor=colors.white,
                              fontSize=8, leading=10, fontName="Helvetica-Bold"))

    story: list = []

    def banner():
        t = Table([[Paragraph(CLASSIFICATION_BANNER, styles["NetraBanner"])]],
                  colWidths=[17 * cm])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CRIT),
                               ("TOPPADDING", (0, 0), (-1, -1), 3),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        return t

    def table(data, col_widths, header=True, severity_col=None):
        tbl = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
        style = [
            ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
            ("TEXTCOLOR", (0, 1), (-1, -1), INK),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, ACCENT),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_ALT]),
        ]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), INK),
                      ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
        tbl.setStyle(TableStyle(style))
        return tbl

    # ── Cover ──────────────────────────────────────────────────────────────────
    story.append(banner())
    story.append(Spacer(1, 18))
    story.append(Paragraph(f"{APP_NAME}", styles["NetraTitle"]))
    story.append(Paragraph(APP_FULL_NAME, styles["NetraSub"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Criminal Network Intelligence Dossier", styles["NetraH2"]))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
    story.append(Spacer(1, 8))

    meta_rows = [
        ["Operation", case_name, "Generated", ts],
        ["Problem statement", f"SIH 2026 · PS {PROBLEM_STATEMENT['id']}",
         "Theme", PROBLEM_STATEMENT["theme"]],
        ["Organisation", APP_ORG, "Engine", f"{APP_NAME} v{APP_VERSION}"],
        ["Department", "NCRB · Women Safety Division", "Category",
         PROBLEM_STATEMENT["category"]],
        ["Actors", str(g.n_actors), "Relationships", str(g.n_links)],
        ["Operational modules", str(g.n_modules), "Modularity Q", f"{g.modularity:.3f}"],
        ["GNN link-prediction AUC", f"{g.gnn.auc:.3f}", "Alerts raised", str(counts["total"])],
    ]
    mt = Table([[Paragraph(f"<b>{a}</b>", styles["Cell"]), Paragraph(str(b), styles["Cell"]),
                 Paragraph(f"<b>{c}</b>", styles["Cell"]), Paragraph(str(d), styles["Cell"])]
                for a, b, c, d in meta_rows],
               colWidths=[4 * cm, 4.5 * cm, 4 * cm, 4.5 * cm])
    mt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.white),
                            ("TOPPADDING", (0, 0), (-1, -1), 5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.append(mt)
    story.append(Spacer(1, 10))

    # Alert severity strip.
    strip = Table([[
        Paragraph(f"<b>{counts['Critical']}</b><br/>CRITICAL", styles["NetraBanner"]),
        Paragraph(f"<b>{counts['High']}</b><br/>HIGH", styles["NetraBanner"]),
        Paragraph(f"<b>{counts['Medium']}</b><br/>MEDIUM", styles["NetraBanner"]),
        Paragraph(f"<b>{counts.get('Low', 0)}</b><br/>LOW", styles["NetraBanner"]),
    ]], colWidths=[4.25 * cm] * 4)
    strip.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), CRIT),
        ("BACKGROUND", (1, 0), (1, 0), HIGH),
        ("BACKGROUND", (2, 0), (2, 0), colors.HexColor("#c9a227")),
        ("BACKGROUND", (3, 0), (3, 0), colors.HexColor("#3f9e57")),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("BOX", (0, 0), (-1, -1), 2, colors.white),
        ("INNERGRID", (0, 0), (-1, -1), 2, colors.white),
    ]))
    story.append(strip)

    # ── Executive summary ────────────────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", styles["NetraH2"]))
    lead_txt = ""
    if len(g.priority):
        lead = g.priority.iloc[0]
        lead_txt = (f" The highest-value disruption target is <b>{lead['name']}</b> "
                    f"({lead.name}, {lead.get('role', '—')}); removing this actor is projected "
                    f"to fragment the network most severely "
                    f"(+{int(lead.get('new_fragments', 0))} components, "
                    f"{lead.get('lcc_drop_pct', 0):.0f}% main-cluster reduction).")
    summary = (
        f"The analysed network comprises <b>{g.n_actors} actors</b> linked by "
        f"<b>{g.n_links} relationships</b>, resolving into <b>{g.n_modules} operational "
        f"modules</b> (modularity Q = {g.modularity:.3f}). The hybrid AI ensemble raised "
        f"<b>{counts['total']} alerts</b> — {counts['Critical']} critical, {counts['High']} "
        f"high and {counts['Medium']} medium priority.{lead_txt}")
    story.append(Paragraph(summary, styles["NetraBody"]))

    # ── Key players table ─────────────────────────────────────────────────────
    if len(g.key_players):
        story.append(Paragraph("2. Key Players", styles["NetraH2"]))
        header = [Paragraph(h, styles["CellHead"]) for h in
                  ["#", "Actor", "ID", "Role", "Influence", "Module", "Risk"]]
        rows = [header]
        for i, (pid, row) in enumerate(g.key_players.head(10).iterrows(), 1):
            rows.append([
                Paragraph(str(i), styles["Cell"]),
                Paragraph(str(row["name"]), styles["Cell"]),
                Paragraph(str(pid), styles["Cell"]),
                Paragraph(str(row.get("role", "—")), styles["Cell"]),
                Paragraph(f"{row['influence_score']:.1f}", styles["Cell"]),
                Paragraph(str(int(row.get("module", 0)) + 1), styles["Cell"]),
                Paragraph(str(row.get("risk_category", "—")), styles["Cell"]),
            ])
        story.append(table(rows, [0.9 * cm, 4 * cm, 2 * cm, 3 * cm, 2.2 * cm, 1.8 * cm, 2.5 * cm]))

    # ── Arrest priority ───────────────────────────────────────────────────────
    if len(g.priority):
        story.append(Paragraph("3. Disruption / Arrest Priority", styles["NetraH2"]))
        story.append(Paragraph(
            "Ranked by projected network damage on removal. Actors already in custody are "
            "excluded.", styles["NetraMuted"]))
        story.append(Spacer(1, 4))
        header = [Paragraph(h, styles["CellHead"]) for h in
                  ["#", "Actor", "Role", "Disruption", "Influence", "+Frag", "Cluster drop"]]
        rows = [header]
        for i, (pid, row) in enumerate(g.priority.head(10).iterrows(), 1):
            rows.append([
                Paragraph(str(i), styles["Cell"]),
                Paragraph(str(row["name"]), styles["Cell"]),
                Paragraph(str(row.get("role", "—")), styles["Cell"]),
                Paragraph(f"{row.get('disruption_score', 0):.1f}", styles["Cell"]),
                Paragraph(f"{row.get('influence_score', 0):.0f}", styles["Cell"]),
                Paragraph(f"+{int(row.get('new_fragments', 0))}", styles["Cell"]),
                Paragraph(f"{row.get('lcc_drop_pct', 0):.0f}%", styles["Cell"]),
            ])
        story.append(table(rows, [0.9 * cm, 4 * cm, 2.8 * cm, 2.5 * cm, 2.2 * cm, 1.6 * cm, 2.6 * cm]))

    # ── Women Safety Division lens ────────────────────────────────────────────
    ws_stats = getattr(g, "ws_stats", {}) or {}
    if ws_stats.get("actors"):
        story.append(Paragraph("4. Women Safety Division Assessment", styles["NetraH2"]))
        story.append(Paragraph(
            f"PS {PROBLEM_STATEMENT['id']} is owned by the {PROBLEM_STATEMENT['department']}. "
            f"Of the loaded caseload, <b>{ws_stats.get('cases', 0)} case(s)</b> "
            f"({ws_stats.get('case_share', 0.0):.1f}%) fall inside the division's mandate, "
            f"implicating <b>{ws_stats.get('actors', 0)} actors</b> — "
            f"{ws_stats.get('minor_victim_actors', 0)} tied to an offence against a minor and "
            f"{ws_stats.get('cyber_actors', 0)} to a cyber-enabled offence. The Women-Safety "
            f"Threat Index (WSTI) below is scored independently of the structural anomaly "
            f"band, so a division-critical actor is surfaced even where the graph looks "
            f"unremarkable.", styles["NetraBody"]))
        esc = getattr(g, "ws_escalation", None)
        if esc is not None and len(esc):
            header = [Paragraph(h, styles["CellHead"]) for h in
                      ["#", "Actor", "ID", "WSTI", "Band", "Cases", "Minor"]]
            rows = [header]
            for _, row in esc.head(8).iterrows():
                rows.append([
                    Paragraph(str(int(row.get("escalation_rank", 0))), styles["Cell"]),
                    Paragraph(str(row.get("name", "")), styles["Cell"]),
                    Paragraph(str(row.get("person_id", "")), styles["Cell"]),
                    Paragraph(f"{float(row.get('wsti', 0)):.0f}", styles["Cell"]),
                    Paragraph(str(row.get("band", "")), styles["Cell"]),
                    Paragraph(str(int(row.get("ws_cases", 0))), styles["Cell"]),
                    Paragraph("YES" if row.get("minor_victim") else "—", styles["Cell"]),
                ])
            story.append(table(rows, [0.9 * cm, 4 * cm, 2 * cm, 1.8 * cm, 2.2 * cm,
                                      1.6 * cm, 1.6 * cm]))
        cor = getattr(g, "corridors", None)
        if cor is not None and len(cor):
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                f"<b>Trafficking corridors:</b> {len(cor)} route(s) inferred from case "
                "geography — a hypothesis to verify against CDR tower and transport records, "
                "not a mapped route.", styles["NetraMuted"]))
            for _, row in cor.head(4).iterrows():
                story.append(Paragraph(
                    f"• <b>{row.get('corridor_id', '')}</b>: {row.get('route', '')} "
                    f"(risk {float(row.get('risk_score', 0)):.0f}, "
                    f"{int(row.get('n_actors', 0))} actors)", styles["NetraBody"]))

    # ── Embedded figures ──────────────────────────────────────────────────────
    if figures:
        story.append(PageBreak())
        story.append(Paragraph("5. Visual Intelligence", styles["NetraH2"]))
        for caption, png in figures.items():
            if not png:
                continue
            try:
                img = Image(io.BytesIO(png))
                # Fit to content width preserving aspect ratio.
                max_w = 16 * cm
                if img.imageWidth > max_w:
                    ratio = max_w / img.imageWidth
                    img.drawWidth = max_w
                    img.drawHeight = img.imageHeight * ratio
                story.append(Paragraph(caption, styles["NetraH3"]))
                story.append(img)
                story.append(Spacer(1, 8))
            except Exception:                                  # noqa: BLE001
                continue

    # ── Alerts ────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(banner())
    story.append(Spacer(1, 8))
    story.append(Paragraph("6. Priority Alert Queue", styles["NetraH2"]))
    sev_colour = {"Critical": CRIT, "High": HIGH, "Medium": colors.HexColor("#c9a227"),
                  "Low": colors.HexColor("#3f9e57")}
    for a in alerts[:16]:
        chip = sev_colour.get(a.severity, MUTED)
        head = Table([[
            Paragraph(f"<b>{a.severity.upper()}</b>", styles["NetraBanner"]),
            Paragraph(f"<b>{a.title}</b>  —  <font color='#5a6b82'>{a.category} · "
                      f"score {a.score:.0f}</font>", styles["NetraBody"]),
        ]], colWidths=[2.4 * cm, 14.6 * cm])
        head.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), chip),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (1, 0), (1, 0), 8),
        ]))
        story.append(head)
        story.append(Paragraph(a.detail, styles["NetraBody"]))
        if a.recommended_action:
            story.append(Paragraph(f"<b>Action:</b> {a.recommended_action}", styles["NetraBody"]))
        if a.evidence:
            story.append(Paragraph("<b>Evidence:</b> " + "; ".join(a.evidence), styles["NetraMuted"]))
        story.append(Spacer(1, 7))

    # ── Methodology footer ────────────────────────────────────────────────────
    story.append(Paragraph("7. Methodology & Provenance", styles["NetraH2"]))
    method = (
        "Influence is a weighted blend of six centrality measures (PageRank, betweenness, "
        "weighted degree, eigenvector, closeness, k-core). Operational modules are found via "
        f"Louvain modularity (Q = {g.modularity:.3f}). A 2-layer GraphSAGE graph neural "
        f"network (held-out link-prediction AUC = {g.gnn.auc:.3f}, {g.gnn.epochs_run} epochs) "
        "produces structural embeddings and per-actor reconstruction residuals. The suspicion "
        "score is a hybrid ensemble of that residual, an Isolation Forest over 24 structural "
        "features, and an 8-rule expert engine — every score is individually explainable.")
    story.append(Paragraph(method, styles["NetraBody"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MUTED))
    story.append(Paragraph(
        f"Generated by {APP_NAME} v{APP_VERSION} · {ts} · All actor data is synthetic and for "
        "demonstration only. Findings are algorithmic leads, not evidence.", styles["NetraMuted"]))

    # ── Build with header/footer on each page ─────────────────────────────────
    def _decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(2 * cm, 1.1 * cm,
                          f"{APP_NAME} · {case_name} · {CLASSIFICATION_BANNER}")
        canvas.drawRightString(19 * cm, 1.1 * cm, f"Page {doc.page}")
        canvas.setStrokeColor(LIGHT)
        canvas.line(2 * cm, 1.5 * cm, 19 * cm, 1.5 * cm)
        canvas.restoreState()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm, bottomMargin=2 * cm,
        title=f"{APP_NAME} Intelligence Dossier — {case_name}",
        author=f"{APP_NAME} v{APP_VERSION}",
    )
    doc.build(story, onFirstPage=_decorate, onLaterPages=_decorate)
    buffer.seek(0)
    return buffer.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# Graph database & open formats export (Neo4j Cypher & GraphML)
# ──────────────────────────────────────────────────────────────────────────────

def build_cypher_export(result, case_name: str = "Operation Sentinel") -> str:
    """Generate Neo4j Cypher statements to import nodes and relationships."""
    lines = [
        f"// NETRA Criminal Network Export — {case_name}",
        f"// Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "// Compatible with Neo4j 4.x / 5.x",
        "",
        "// 1. Create Constraints",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE;",
        "",
    ]
    # Nodes
    if result.persons is not None and len(result.persons):
        lines.append("// 2. Create Person Nodes")
        for pid, row in result.persons.iterrows():
            name = str(row.get("name", pid)).replace("'", "\\'")
            role = str(row.get("role", "Unknown")).replace("'", "\\'")
            syndicate = str(row.get("syndicate", "Unknown")).replace("'", "\\'")
            risk = str(row.get("risk_category", "Unknown"))
            inf = float(row.get("influence_score", 0.0))
            susp = float(row.get("suspicion_score", 0.0))
            props = [
                f"id: '{pid}'",
                f"name: '{name}'",
                f"role: '{role}'",
                f"syndicate: '{syndicate}'",
                f"influence_score: {inf:.2f}",
                f"suspicion_score: {susp:.2f}",
                f"risk_category: '{risk}'",
            ]
            lines.append(f"MERGE (p:Person {{id: '{pid}'}}) SET p += {{{', '.join(props)}}};")

    # Edges
    if result.relationships is not None and len(result.relationships):
        lines.append("\n// 3. Create Relationship Edges")
        for _, row in result.relationships.iterrows():
            src = str(row["source_id"])
            tgt = str(row["target_id"])
            rel_type = re.sub(r"[^A-Za-z0-9_]", "_", str(row.get("relation_type", "RELATED_TO")).upper())
            if not rel_type or rel_type[0].isdigit():
                rel_type = f"REL_{rel_type}"
            weight = float(row.get("strength", 1.0))
            src_ev = str(row.get("evidence_source", "Database")).replace("'", "\\'")
            lines.append(
                f"MATCH (s:Person {{id: '{src}'}}), (t:Person {{id: '{tgt}'}}) "
                f"MERGE (s)-[r:{rel_type}]->(t) "
                f"SET r.weight = {weight:.2f}, r.evidence = '{src_ev}';"
            )
    return "\n".join(lines)


def build_graphml_export(result) -> str:
    """Generate GraphML XML string for Gephi, Cytoscape, Maltego, i2."""
    root = ET.Element("graphml", {
        "xmlns": "http://graphml.graphdrawing.org/xmlns",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"
    })
    # Node keys
    ET.SubElement(root, "key", {"id": "d0", "for": "node", "attr.name": "name", "attr.type": "string"})
    ET.SubElement(root, "key", {"id": "d1", "for": "node", "attr.name": "role", "attr.type": "string"})
    ET.SubElement(root, "key", {"id": "d2", "for": "node", "attr.name": "influence", "attr.type": "double"})
    ET.SubElement(root, "key", {"id": "d3", "for": "node", "attr.name": "suspicion", "attr.type": "double"})
    # Edge keys
    ET.SubElement(root, "key", {"id": "d4", "for": "edge", "attr.name": "relation_type", "attr.type": "string"})
    ET.SubElement(root, "key", {"id": "d5", "for": "edge", "attr.name": "weight", "attr.type": "double"})

    graph = ET.SubElement(root, "graph", {"id": "G", "edgedefault": "directed"})

    if result.persons is not None:
        for pid, row in result.persons.iterrows():
            n = ET.SubElement(graph, "node", {"id": str(pid)})
            d_name = ET.SubElement(n, "data", {"key": "d0"})
            d_name.text = str(row.get("name", pid))
            d_role = ET.SubElement(n, "data", {"key": "d1"})
            d_role.text = str(row.get("role", ""))
            d_inf = ET.SubElement(n, "data", {"key": "d2"})
            d_inf.text = str(row.get("influence_score", 0.0))
            d_susp = ET.SubElement(n, "data", {"key": "d3"})
            d_susp.text = str(row.get("suspicion_score", 0.0))

    if result.relationships is not None:
        for idx, row in result.relationships.iterrows():
            e = ET.SubElement(graph, "edge", {
                "id": f"e{idx}",
                "source": str(row["source_id"]),
                "target": str(row["target_id"]),
            })
            d_r = ET.SubElement(e, "data", {"key": "d4"})
            d_r.text = str(row.get("relation_type", "Communication"))
            d_w = ET.SubElement(e, "data", {"key": "d5"})
            d_w.text = str(row.get("strength", 1.0))

    rough = ET.tostring(root, "utf-8")
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ")


# ──────────────────────────────────────────────────────────────────────────────
# AI Executive Intelligence Brief
# ──────────────────────────────────────────────────────────────────────────────

def generate_ai_intelligence_brief(result, alerts: list[Alert] | None = None,
                                    case_name: str = "Operation Sentinel") -> str:
    """Generate structured AI Executive Intelligence Brief in MHA / NCRB Standard format."""
    alerts = alerts if alerts is not None else build_alerts(result)
    counts = alert_counts(alerts)
    ts = result.generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    g = result

    top_key = g.key_players.iloc[0]["name"] if len(g.key_players) else "Unknown"
    top_role = g.key_players.iloc[0].get("role", "Leader") if len(g.key_players) else ""
    top_disrupt = g.priority.iloc[0]["name"] if len(g.priority) else "Unknown"

    brief = [
        f"# 🇮🇳 MINISTRY OF HOME AFFAIRS · NCRB WOMEN SAFETY DIVISION",
        f"## 📑 AI EXECUTIVE INTELLIGENCE BRIEFING (RESTRICTED / LEA ONLY)",
        f"**Case Reference:** {case_name} | **Timestamp:** {ts} | **Engine:** NETRA AI v{APP_VERSION}",
        "---",
        "### 1. 🎯 STRATEGIC THREAT LANDSCAPE",
        f"- **Network Footprint:** {g.n_actors} actors identified across {g.n_modules} distinct operational syndicates/clusters with modularity index Q = {g.modularity:.3f}.",
        f"- **Apex Coordinator:** **{top_key}** identified as the primary operational coordinator (Role: {top_role}).",
        f"- **Critical Chokepoint for Disruption:** Target **{top_disrupt}** is flagged by network disruption simulation as the highest-yield interdiction point; tactical isolation of this node will cause maximum structural fracture across syndicates.",
        "",
        "### 2. 🚨 THREAT ENSEMBLE & ANOMALY ASSESSMENT",
        f"- **Total Intelligence Alerts:** {counts['total']} actionable leads generated ({counts['Critical']} Critical, {counts['High']} High priority).",
        f"- **GraphSAGE GNN Link Validation:** Self-supervised link prediction achieved {g.gnn.auc:.3f} AUC, confirming robust latent pattern detection without manual supervision.",
        f"- **Hidden Mule & Cut-Out Nodes:** Hybrid Isolation Forest ⊕ GNN residual ensemble isolated {len(g.anomalies)} structural anomalies operating beneath standard police thresholds.",
        "",
        "### 3. 🚺 WOMEN SAFETY DIVISION STATUTORY MANDATE",
    ]

    ws_stats = getattr(g, "ws_stats", {}) or {}
    if ws_stats.get("actors"):
        brief.extend([
            f"- **Statutory Impact:** {ws_stats.get('cases', 0)} case(s) ({ws_stats.get('case_share', 0.0):.1f}% of caseload) directly concern offences against women and minors.",
            f"- **High-Risk Syndicate Targets:** {ws_stats.get('actors', 0)} actors implicated, with {ws_stats.get('minor_victim_actors', 0)} connected to minor victim cases (POCSO mandate).",
            f"- **Peak Threat Score:** WSTI recorded at {ws_stats.get('peak_wsti', 0.0):.0f}/100.",
        ])
    else:
        brief.append("- *No direct Women Safety caseload detected in current batch.*")

    brief.extend([
        "",
        "### 4. ⚡ ACTIONABLE OPERATIONAL RECOMMENDATIONS",
        "1. **Interception & Surveillance:** Prioritize simultaneous electronic interception (CDR & Financial gateway surveillance) on top 3 ranked Key Players.",
        "2. **Corridor Interdiction:** Coordinate with regional police stations along detected transit corridors to monitor identified front entities and transport channels.",
        "3. **Financial Freezing:** Issue Section 91/102 CrPC notices to freeze crypto wallets and bank accounts associated with top flagged money mules.",
        "---",
        "*DISCLAIMER: This automated briefing is generated through algorithmic graph neural networks and heuristic expert ensembles for investigative lead generation.*"
    ])
    return "\n".join(brief)

