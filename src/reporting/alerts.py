"""
Layer 5a — alert synthesis.

Turns the raw analysis frames into a ranked queue of actionable alerts. An alert is
not just a high score: it is a score *plus* the reason, the recommended action, and the
evidence an officer would cite to justify acting on it.

The dashboard renders these as cards; the report exporter lists them verbatim. Keeping
the synthesis here (not in the UI) means the PDF and the screen never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import SUSPICION_BANDS

# Severity ordering for stable sort — Critical first.
_SEVERITY_RANK = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4}


@dataclass
class Alert:
    """A single actionable finding."""

    severity: str                 # Critical | High | Medium | Low | Info
    category: str                 # short kind, e.g. "Key Player", "Anomaly", "Cluster"
    title: str
    detail: str
    subject_id: str | None = None
    subject_name: str | None = None
    score: float = 0.0
    recommended_action: str = ""
    evidence: list[str] = field(default_factory=list)
    colour: str = "#8b98a9"

    def as_dict(self) -> dict:
        return {
            "severity": self.severity,
            "category": self.category,
            "title": self.title,
            "detail": self.detail,
            "subject_id": self.subject_id,
            "subject_name": self.subject_name,
            "score": round(float(self.score), 1),
            "recommended_action": self.recommended_action,
            "evidence": list(self.evidence),
        }


def _band_colour(name: str) -> str:
    for _, label, colour in SUSPICION_BANDS:
        if label == name:
            return colour
    return "#8b98a9"


def _severity_for_score(score: float) -> str:
    for threshold, label, _ in SUSPICION_BANDS:
        if score >= threshold:
            return label
    return "Low"


def build_alerts(result, max_alerts: int = 40) -> list[Alert]:
    """Synthesise the ranked alert queue from a completed analysis.

    Draws from four streams — anomalies, arrest-priority targets, suspicious clusters
    and critical inter-module links — then de-duplicates so one actor does not generate
    five near-identical cards, and returns them sorted by severity then score.
    """
    alerts: list[Alert] = []
    seen_subjects: set[str] = set()

    # ── 1. High-suspicion individuals ─────────────────────────────────────────
    if len(result.anomalies):
        flagged = result.anomalies[result.anomalies["band"].isin(["Critical", "High"])]
        for pid, row in flagged.iterrows():
            reasons = row.get("reasons") or []
            primary = reasons[0].split(" — ")[0].replace("**", "") if reasons else "Anomalous structure"
            role = row.get("role")
            action = _anomaly_action(row)
            alerts.append(Alert(
                severity=row["band"],
                category="Anomaly",
                title=f"{primary}: {row['name']}",
                detail=(f"Suspicion {row['suspicion_score']:.0f}/100 · "
                        f"{row['n_rules']} rule(s) triggered · module {int(row['module']) + 1}"
                        + (f" · role {role}" if role and pd.notna(role) else "")),
                subject_id=pid,
                subject_name=row["name"],
                score=float(row["suspicion_score"]),
                recommended_action=action,
                evidence=[r.replace("**", "") for r in reasons[:4]],
                colour=row.get("band_colour", _band_colour(row["band"])),
            ))
            seen_subjects.add(pid)

    # ── 2. Arrest-priority targets not already surfaced as anomalies ──────────
    if len(result.priority):
        for pid, row in result.priority.iterrows():
            if pid in seen_subjects:
                continue
            disruption = float(row.get("disruption_score", 0))
            if disruption < 30:
                continue
            frag = row.get("new_fragments", 0)
            lcc = row.get("lcc_drop_pct", 0)
            sev = "High" if disruption >= 60 else "Medium"
            alerts.append(Alert(
                severity=sev,
                category="Key Player",
                title=f"High-value arrest target: {row['name']}",
                detail=(f"Disruption {disruption:.0f}/100 · role {row.get('role', '—')} · "
                        f"influence {row.get('influence_score', 0):.0f}"),
                subject_id=pid,
                subject_name=row["name"],
                score=disruption,
                recommended_action=(
                    "Prioritise for arrest/surveillance — removal is projected to "
                    f"fragment the network into {int(frag) + 1} pieces and shrink the "
                    f"main component by {lcc:.0f}%."),
                evidence=[
                    f"Role: {row.get('role', '—')}",
                    f"Influence score {row.get('influence_score', 0):.0f}/100",
                    f"Projected fragments +{int(frag)}",
                    f"Largest-component drop {lcc:.0f}%",
                ],
                colour=_band_colour(sev),
            ))
            seen_subjects.add(pid)

    # ── 3. Suspicious clusters ────────────────────────────────────────────────
    if len(result.clusters):
        for _, row in result.clusters.iterrows():
            score = float(row["cluster_score"])
            if score < 45:
                continue
            sev = _severity_for_score(score)
            patterns = ", ".join(row.get("signature_patterns") or []) or "mixed activity"
            alerts.append(Alert(
                severity=sev,
                category="Cluster",
                title=f"Suspicious module: {row['label']}",
                detail=(f"Cluster score {score:.0f}/100 · {int(row['size'])} members · "
                        f"{int(row['flagged_actors'])} flagged · {row['dominant_offence']}"),
                subject_id=row.get("lead_actor_id"),
                subject_name=row.get("lead_actor"),
                score=score,
                recommended_action=(
                    f"Investigate as a coordinated cell led by {row.get('lead_actor', 'unknown')}. "
                    f"Dominant signature: {patterns}."),
                evidence=[
                    f"Members: {int(row['size'])} across {row.get('states', 0)} state(s)",
                    f"Flagged actors: {int(row['flagged_actors'])} "
                    f"(density {row.get('anomaly_density', 0):.0%})",
                    f"Internal density {row.get('density', 0):.2f} · "
                    f"insulation {row.get('insulation', 0):.0%}",
                    f"Dominant offence: {row['dominant_offence']}",
                ],
                colour=_band_colour(sev),
            ))

    # ── 4. Critical inter-module links (cut edges) ────────────────────────────
    if len(result.cut_edges):
        cut = result.cut_edges[result.cut_edges.get("is_cut_edge", False)] \
            if "is_cut_edge" in result.cut_edges.columns else result.cut_edges.head(3)
        for _, row in cut.head(4).iterrows():
            alerts.append(Alert(
                severity="Medium",
                category="Link",
                title=f"Critical link: {row['source']} ↔ {row['target']}",
                detail=(f"Sole/high-betweenness bridge between module "
                        f"{int(row['module_a']) + 1} and {int(row['module_b']) + 1} · "
                        f"{row['relation']}"),
                subject_id=row.get("source_id"),
                subject_name=f"{row['source']} ↔ {row['target']}",
                score=50.0,
                recommended_action=(
                    "Monitor this channel — it is how the two modules coordinate; "
                    "intercepting it maps the command relationship between them."),
                evidence=[
                    f"Relation: {row['relation']}",
                    f"Evidence basis: {row.get('evidence', '—')}",
                    f"Edge betweenness {row.get('edge_betweenness', 0):.4f}",
                    "Removal disconnects the two modules" if row.get("is_cut_edge")
                    else "High-traffic coordination channel",
                ],
                colour=_band_colour("Medium"),
            ))

    # ── 5. Women Safety Division escalations (PS 26189 owning department) ──────
    # The WSTI is a content score, independent of the structural anomaly band, so a
    # division-critical actor can be entirely absent from streams 1–4. Surfacing the
    # escalation queue here is what makes the alert list answer the actual brief and
    # not just generic network analysis. A minor-victim tie forces at least High.
    ws = getattr(result, "ws_escalation", None)
    if ws is None or not len(ws):
        ws = getattr(result, "wsti", None)
    if ws is not None and len(ws):
        for _, row in ws.head(8).iterrows():
            wsti = float(row.get("wsti", 0))
            minor = bool(row.get("minor_victim", False))
            if wsti < 50 and not minor:
                continue
            sev = "Critical" if (wsti >= 75 or minor) else "High"
            flags = row.get("flags") or []
            if isinstance(flags, str):
                flags = [f.strip() for f in flags.split("|") if f.strip()]
            pid = str(row.get("person_id", ""))
            title_pat = ("Minor-victim offender" if minor
                         else "Women-safety threat actor")
            alerts.append(Alert(
                severity=sev,
                category="Women Safety",
                title=f"{title_pat}: {row.get('name', pid)}",
                detail=(f"WSTI {wsti:.0f}/100 · {int(row.get('ws_cases', 0))} division "
                        f"case(s) · peak severity {int(row.get('peak_severity', 0))}/5"
                        + (" · POCSO escalation path" if minor else "")),
                subject_id=pid,
                subject_name=row.get("name"),
                score=wsti,
                recommended_action=(
                    "Refer to the Women Safety Division for priority action"
                    + (" under POCSO timelines — offence against a minor is a statutory "
                       "aggravator requiring time-bound handling." if minor
                       else ". Preserve digital evidence where the offence is cyber-enabled.")),
                evidence=flags[:4] or [f"WSTI {wsti:.0f}/100"],
                colour=_band_colour(sev),
            ))

    alerts.sort(key=lambda a: (_SEVERITY_RANK.get(a.severity, 9), -a.score))
    return alerts[:max_alerts]


def _anomaly_action(row: pd.Series) -> str:
    """Map the dominant rule to a concrete recommended action."""
    keys = row.get("rule_keys") or []
    if "structural_gatekeeper" in keys:
        severed = int(row.get("severed_cell", 0))
        return (f"Treat as a priority disruption target — sole conduit to a {severed}-actor "
                "cell; arrest isolates that cell from the network.")
    if "cross_module_broker" in keys:
        return ("Place under surveillance before acting — this broker links otherwise "
                "separate organisations; watching them maps both.")
    if "financial_mule" in keys:
        return ("Refer to financial-intelligence unit — trace the accounts; the mule leads "
                "to the money's origin and destination.")
    if "isolated_high_severity" in keys:
        return ("Re-investigate known associates — serious record with almost no mapped "
                "contacts implies the network around them is incomplete.")
    if "dormancy_spike" in keys:
        return ("Escalate monitoring — a reactivated actor building contacts fast is often "
                "standing up a new operation.")
    band = row.get("band", "Medium")
    if band == "Critical":
        return "Immediate review recommended — multiple independent signals converge here."
    return "Add to active watchlist and corroborate with field intelligence."


def alerts_to_frame(alerts: list[Alert]) -> pd.DataFrame:
    """Flatten alerts to a DataFrame for tables and CSV export."""
    if not alerts:
        return pd.DataFrame(columns=["severity", "category", "title", "subject_id",
                                     "score", "recommended_action"])
    return pd.DataFrame([{
        "severity": a.severity,
        "category": a.category,
        "title": a.title,
        "subject_id": a.subject_id or "",
        "subject_name": a.subject_name or "",
        "score": round(a.score, 1),
        "detail": a.detail,
        "recommended_action": a.recommended_action,
        "evidence": " | ".join(a.evidence),
    } for a in alerts])


def alert_counts(alerts: list[Alert]) -> dict[str, int]:
    """Severity tallies for the KPI strip."""
    out = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0, "total": len(alerts)}
    for a in alerts:
        out[a.severity] = out.get(a.severity, 0) + 1
    return out
