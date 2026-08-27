"""
Reusable UI components — HTML/CSS fragments the pages compose.

These render themed cards and badges via ``st.markdown(..., unsafe_allow_html=True)``.
Keeping them here means every page's metric strip, alert card and actor profile looks
identical and changes in one place.
"""

from __future__ import annotations

import html

import streamlit as st

from src.config import PALETTE, RISK_COLORS, ROLE_COLORS, STATUS_COLORS


def _esc(text) -> str:
    return html.escape(str(text))


def section_header(title: str, subtitle: str = "") -> None:
    """A left-accented section title with an optional right-aligned note."""
    sub = f'<span class="sub">{_esc(subtitle)}</span>' if subtitle else ""
    st.markdown(
        f'<div class="section-head"><div class="bar"></div>'
        f'<span class="txt">{_esc(title)}</span>{sub}</div>',
        unsafe_allow_html=True)


def metric_cards(metrics: list[dict]) -> None:
    """Render a responsive grid of metric cards.

    Each metric: ``{"label", "value", "accent"?, "delta"?, "delta_colour"?}``.
    """
    cards = []
    for m in metrics:
        accent = m.get("accent", PALETTE["accent"])
        delta = ""
        if m.get("delta"):
            dc = m.get("delta_colour", PALETTE["text_dim"])
            delta = f'<div class="metric-delta" style="color:{dc}">{_esc(m["delta"])}</div>'
        cards.append(
            f'<div class="metric-card" style="--accent:{accent}">'
            f'<div class="metric-value" style="color:{accent}">{_esc(m["value"])}</div>'
            f'<div class="metric-label">{_esc(m["label"])}</div>{delta}</div>')
    st.markdown(f'<div class="metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def chip(text: str, colour: str, filled: bool = False) -> str:
    """Return an inline chip span (caller places it)."""
    if filled:
        return (f'<span class="chip" style="background:{colour};color:#0a0e17;">'
                f'{_esc(text)}</span>')
    return (f'<span class="chip" style="color:{colour};border:1px solid {colour};'
            f'background:{colour}18;">{_esc(text)}</span>')


def risk_chip(risk: str) -> str:
    return chip(risk, RISK_COLORS.get(risk, PALETTE["text_dim"]), filled=risk in ("Critical", "High"))


def role_chip(role: str) -> str:
    return chip(role, ROLE_COLORS.get(role, PALETTE["text_dim"]))


def status_chip(status: str) -> str:
    return chip(status, STATUS_COLORS.get(status, PALETTE["text_dim"]))


def alert_card(alert) -> None:
    """Render a single :class:`~src.reporting.alerts.Alert` as a styled card."""
    ev = ""
    if alert.evidence:
        items = "".join(f"• {_esc(e)}<br>" for e in alert.evidence)
        ev = f'<div class="alert-evidence">{items}</div>'
    action = ""
    if alert.recommended_action:
        action = f'<div class="alert-action">▸ {_esc(alert.recommended_action)}</div>'
    st.markdown(f"""
    <div class="alert-card" style="--sev:{alert.colour}">
      <div class="head">
        <span class="alert-sev" style="background:{alert.colour}">{_esc(alert.severity)}</span>
        <span class="alert-title">{_esc(alert.title)}</span>
        <span class="alert-cat">{_esc(alert.category)} · {alert.score:.0f}</span>
      </div>
      <div class="alert-detail">{_esc(alert.detail)}</div>
      {action}{ev}
    </div>
    """, unsafe_allow_html=True)


def band_pill(band: str, colour: str, score: float) -> str:
    """Suspicion band pill with score — for inline use in profiles."""
    return (f'<span class="chip" style="background:{colour};color:#0a0e17;font-weight:700;">'
            f'{_esc(band)} · {score:.0f}</span>')


def actor_profile(result, person_id: str) -> None:
    """A full actor intelligence panel: identity, metrics, roles and reasons."""
    if person_id not in result.graph:
        st.warning("Actor not found in the current network.")
        return

    rec = result.actor(person_id)
    name = rec.get("name", person_id)
    module = rec.get("module", -1)

    role = rec.get("role", "—")
    infl = rec.get("influence_score")
    susp = rec.get("suspicion_score")
    band = rec.get("band")

    chips = [role_chip(role), risk_chip(rec.get("risk_category", "Low")),
             status_chip(rec.get("status", "Active"))]
    if band and susp is not None:
        colour = next((c for _, n, c in __import__("src.config", fromlist=["SUSPICION_BANDS"]).SUSPICION_BANDS
                       if n == band), PALETTE["text_dim"])
        chips.append(band_pill(band, colour, float(susp)))

    st.markdown(f"""
    <div class="profile-card">
      <div style="display:flex;align-items:baseline;gap:0.7rem;flex-wrap:wrap;">
        <span class="profile-name">{_esc(name)}</span>
        <span class="profile-id">{_esc(person_id)}</span>
        <span style="color:{PALETTE['text_dim']};font-size:0.85rem;">
          Module {int(module) + 1} · {_esc(rec.get('city', '—'))}, {_esc(rec.get('state', '—'))}</span>
      </div>
      <div style="margin:0.5rem 0;">{''.join(chips)}</div>
    </div>
    """, unsafe_allow_html=True)

    # Metric strip for this actor.
    cards = []
    if infl is not None:
        cards.append({"label": "Influence", "value": f"{infl:.0f}", "accent": PALETTE["accent"]})
    if susp is not None:
        colour = rec.get("band_colour", PALETTE["high"])
        sc = float(susp)
        colour = (PALETTE["critical"] if sc >= 80 else PALETTE["high"] if sc >= 65
                  else PALETTE["medium"] if sc >= 45 else PALETTE["low"])
        cards.append({"label": "Suspicion", "value": f"{sc:.0f}", "accent": colour})
    cards.append({"label": "Degree", "value": int(rec.get("degree", 0)), "accent": PALETTE["accent_2"]})
    cards.append({"label": "Prior offences", "value": int(rec.get("prior_offences", 0)),
                  "accent": PALETTE["high"]})
    if rec.get("pagerank") is not None:
        cards.append({"label": "PageRank", "value": f"{rec['pagerank']:.3f}", "accent": PALETTE["accent"]})
    if cards:
        metric_cards(cards)

    reasons = rec.get("reasons") or []
    if reasons and reasons[0] != "No individual rule fired; score reflects mild statistical deviation only.":
        st.markdown("**Why this actor is flagged**")
        for r in reasons:
            st.markdown(f'<div class="reason-item">{_md_inline(r)}</div>', unsafe_allow_html=True)


def _md_inline(text: str) -> str:
    """Render the ``**bold**`` markers we store in reasons as HTML, escaping the rest."""
    import re
    parts = re.split(r"(\*\*[^*]+\*\*)", str(text))
    out = []
    for p in parts:
        if p.startswith("**") and p.endswith("**"):
            out.append(f"<b>{_esc(p[2:-2])}</b>")
        else:
            out.append(_esc(p))
    return "".join(out)


def empty_state(message: str, icon: str = "📭") -> None:
    st.markdown(f"""
    <div style="text-align:center;padding:3rem 1rem;color:{PALETTE['text_dim']};">
      <div style="font-size:2.5rem;margin-bottom:0.5rem;">{icon}</div>
      <div style="font-size:0.95rem;">{_esc(message)}</div>
    </div>
    """, unsafe_allow_html=True)


def legend_row(items: list[tuple[str, str]]) -> None:
    """Inline colour legend: ``[(label, colour), ...]``."""
    spans = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:0.3rem;margin-right:0.9rem;'
        f'font-size:0.75rem;color:{PALETTE["text_dim"]};">'
        f'<span style="width:10px;height:10px;border-radius:3px;background:{c};"></span>'
        f'{_esc(l)}</span>' for l, c in items)
    st.markdown(f'<div style="margin:0.3rem 0;">{spans}</div>', unsafe_allow_html=True)
