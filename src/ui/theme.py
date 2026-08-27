"""
UI theme — a single injected stylesheet.

Streamlit's declarative widgets are kept; this only restyles them into the dark
"intelligence console" look. Keeping all CSS in one string (rather than scattered
``st.markdown`` calls) means the visual language is defined in exactly one place.

Animation policy
----------------
Every transition animates ``transform`` or ``opacity`` only. Those two properties
are composited on the GPU and never trigger layout or paint, so the motion costs
nothing measurable even on the shared vCPU of a free-tier container. Animating
``width``, ``height``, ``top`` or ``box-shadow`` instead would reflow the page on
every frame — visible jank on exactly the hardware a judge is likely to be running.
Durations stay under 400 ms so the interface reads as responsive, not decorative.
"""

from __future__ import annotations

import streamlit as st

from src.config import PALETTE


def inject_theme() -> None:
    """Inject the global stylesheet. Call once, right after ``st.set_page_config``."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, sans-serif;
    }}

    /* ── Motion primitives ─────────────────────────────────────────────────── */
    @keyframes netra-rise {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes netra-fade {{
        from {{ opacity: 0; }} to {{ opacity: 1; }}
    }}
    @keyframes netra-pulse {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50%      {{ opacity: 0.45; transform: scale(0.86); }}
    }}
    @keyframes netra-sweep {{
        from {{ transform: translateX(-100%); }}
        to   {{ transform: translateX(100%); }}
    }}
    /* Honour the OS reduced-motion setting: vestibular disorders are not a
       preference to override, and an accessibility regression in a government
       tool is a real defect. */
    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            animation-duration: 0.001ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.001ms !important;
        }}
    }}

    /* ── App shell ─────────────────────────────────────────────────────────── */
    .stApp {{
        background:
            radial-gradient(1200px 600px at 80% -5%, rgba(0,212,255,0.06), transparent),
            radial-gradient(900px 500px at 0% 0%, rgba(124,92,255,0.06), transparent),
            {PALETTE['bg']};
    }}
    .block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }}

    /* ── Typography ────────────────────────────────────────────────────────── */
    h1, h2, h3, h4 {{ color: {PALETTE['text']}; font-weight: 700; letter-spacing: -0.01em; }}
    h1 {{ font-size: 1.85rem; }}
    p, span, label, div {{ color: {PALETTE['text']}; }}
    .mono {{ font-family: 'JetBrains Mono', monospace; }}

    /* ── Top masthead ──────────────────────────────────────────────────────── */
    .netra-masthead {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.55rem 1.15rem; margin-bottom: 0.35rem; gap: 1rem;
        background: linear-gradient(135deg, {PALETTE['surface']} 0%, {PALETTE['surface_2']} 100%);
        border: 1px solid {PALETTE['border']}; border-radius: 14px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.35);
        position: relative; overflow: hidden;
        animation: netra-rise 0.4s ease-out both;
    }}
    /* A single scanning highlight across the masthead — the one ambient motion in
       the app. Pointer-events off so it can never intercept a click. */
    .netra-masthead::after {{
        content: ''; position: absolute; inset: 0; pointer-events: none;
        background: linear-gradient(100deg, transparent 35%,
                    rgba(0,212,255,0.07) 50%, transparent 65%);
        animation: netra-sweep 7s ease-in-out infinite;
    }}
    .netra-brand {{ display: flex; align-items: center; gap: 0.8rem; }}
    .netra-logo {{
        font-size: 1.7rem; width: 46px; height: 46px; display: grid; place-items: center;
        background: linear-gradient(135deg, {PALETTE['accent']}, {PALETTE['accent_2']});
        border-radius: 12px; box-shadow: 0 0 20px rgba(0,212,255,0.35);
        transition: transform 0.25s ease;
    }}
    .netra-masthead:hover .netra-logo {{ transform: rotate(-6deg) scale(1.05); }}
    .netra-title {{ font-size: 1.35rem; font-weight: 800; color: {PALETTE['text']}; line-height: 1.1; }}
    .netra-title .accent {{ color: {PALETTE['accent']}; }}
    .netra-subtitle {{ font-size: 0.72rem; color: {PALETTE['text_dim']}; letter-spacing: 0.14em;
        text-transform: uppercase; }}
    .netra-dept {{ font-size: 0.68rem; color: {PALETTE['accent']}; margin-top: 0.15rem;
        font-weight: 600; }}
    .netra-badge {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; letter-spacing: 0.1em;
        color: {PALETTE['critical']}; border: 1px solid {PALETTE['critical']};
        border-radius: 6px; padding: 0.25rem 0.6rem; background: rgba(255,59,92,0.08);
        display: inline-block; white-space: nowrap;
    }}
    .netra-badge .dot {{ animation: netra-pulse 2.2s ease-in-out infinite;
        display: inline-block; }}
    .netra-ps {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
        color: {PALETTE['accent']}; border: 1px solid rgba(0,212,255,0.35);
        background: rgba(0,212,255,0.07); border-radius: 6px;
        padding: 0.25rem 0.6rem; display: inline-block; margin-bottom: 0.3rem;
        white-space: nowrap;
    }}

    /* ── Problem-statement strip ────────────────────────────────────────────── */
    .ps-strip {{
        display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
        gap: 0.5rem; margin: 0.3rem 0 0.9rem 0;
    }}
    .ps-cell {{
        background: {PALETTE['surface']}; border: 1px solid {PALETTE['border']};
        border-radius: 10px; padding: 0.55rem 0.75rem;
        animation: netra-fade 0.5s ease-out both;
    }}
    .ps-cell .k {{ font-size: 0.6rem; color: {PALETTE['text_dim']}; text-transform: uppercase;
        letter-spacing: 0.09em; }}
    .ps-cell .v {{ font-size: 0.82rem; font-weight: 700; color: {PALETTE['text']};
        margin-top: 0.15rem; line-height: 1.25; }}

    /* Requirement → implementation traceability row */
    .trace-row {{
        display: grid; grid-template-columns: 1.1fr 1.6fr 0.8fr; gap: 0.7rem;
        padding: 0.6rem 0.8rem; border-bottom: 1px solid {PALETTE['border']};
        transition: transform 0.15s ease;
    }}
    .trace-row:hover {{ transform: translateX(3px); }}
    .trace-req {{ font-size: 0.83rem; font-weight: 600; color: {PALETTE['text']}; }}
    .trace-imp {{ font-size: 0.78rem; color: {PALETTE['text_dim']}; }}
    .trace-scr {{ font-size: 0.72rem; color: {PALETTE['accent']}; font-weight: 600;
        text-align: right; font-family: 'JetBrains Mono', monospace; }}

    /* ── Metric cards ──────────────────────────────────────────────────────── */
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.75rem; margin: 0.5rem 0 1rem 0; }}
    .metric-card {{
        background: linear-gradient(160deg, {PALETTE['surface']} 0%, rgba(18,24,38,0.6) 100%);
        border: 1px solid {PALETTE['border']}; border-radius: 12px; padding: 0.9rem 1rem;
        position: relative; overflow: hidden; transition: transform 0.15s, border-color 0.15s;
        animation: netra-rise 0.35s ease-out both;
    }}
    .metric-card:hover {{ transform: translateY(-2px); border-color: {PALETTE['accent']}; }}
    .metric-card::before {{ content: ''; position: absolute; top: 0; left: 0; width: 3px;
        height: 100%; background: var(--accent, {PALETTE['accent']}); }}
    .metric-value {{ font-size: 1.7rem; font-weight: 800; color: {PALETTE['text']};
        font-family: 'JetBrains Mono', monospace; line-height: 1.1; }}
    .metric-label {{ font-size: 0.7rem; color: {PALETTE['text_dim']}; text-transform: uppercase;
        letter-spacing: 0.08em; margin-top: 0.2rem; }}
    .metric-delta {{ font-size: 0.72rem; margin-top: 0.35rem; font-weight: 600; }}

    /* ── Section header ────────────────────────────────────────────────────── */
    .section-head {{ display: flex; align-items: center; gap: 0.6rem; margin: 1.2rem 0 0.4rem 0; }}
    .section-head .bar {{ width: 4px; height: 22px; border-radius: 3px;
        background: linear-gradient(180deg, {PALETTE['accent']}, {PALETTE['accent_2']}); }}
    .section-head .txt {{ font-size: 1.05rem; font-weight: 700; color: {PALETTE['text']}; }}
    .section-head .sub {{ font-size: 0.78rem; color: {PALETTE['text_dim']}; margin-left: auto; }}

    /* ── Badges & chips ────────────────────────────────────────────────────── */
    .chip {{ display: inline-block; font-size: 0.7rem; font-weight: 600; padding: 0.15rem 0.55rem;
        border-radius: 20px; margin: 0.1rem 0.15rem; font-family: 'JetBrains Mono', monospace; }}

    /* ── Data-source cards (Data & Sources screen) ─────────────────────────── */
    .src-card {{
        background: {PALETTE['surface']}; border: 1px solid {PALETTE['border']};
        border-left: 3px solid var(--tone, {PALETTE['accent']});
        border-radius: 10px; padding: 0.7rem 0.9rem; margin-bottom: 0.5rem;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .src-card:hover {{ transform: translateX(3px); }}
    .src-name {{ font-size: 0.88rem; font-weight: 700; color: {PALETTE['text']}; }}
    .src-path {{ font-size: 0.74rem; color: {PALETTE['text_dim']}; margin-top: 0.15rem; }}
    .src-state {{ font-size: 0.64rem; font-weight: 700; letter-spacing: 0.07em;
        text-transform: uppercase; font-family: 'JetBrains Mono', monospace; }}

    /* ── Entity pills & document cards (Entity Extraction screen) ──────────── */
    .ent-pill {{
        display: inline-flex; align-items: center; gap: 0.3rem;
        font-size: 0.74rem; font-weight: 600; padding: 0.2rem 0.6rem;
        border-radius: 7px; margin: 0.15rem 0.2rem 0.15rem 0;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid var(--tone, {PALETTE['accent']});
        background: color-mix(in srgb, var(--tone, {PALETTE['accent']}) 12%, transparent);
        color: var(--tone, {PALETTE['accent']});
        transition: transform 0.12s ease;
    }}
    .ent-pill:hover {{ transform: translateY(-1px); }}
    .doc-card {{
        background: {PALETTE['surface']}; border: 1px solid {PALETTE['border']};
        border-radius: 11px; padding: 0.85rem 1rem; margin-bottom: 0.6rem;
        animation: netra-rise 0.3s ease-out both;
    }}
    .doc-head {{ display: flex; align-items: baseline; gap: 0.6rem; flex-wrap: wrap;
        margin-bottom: 0.4rem; }}
    .doc-title {{ font-weight: 700; font-size: 0.92rem; color: {PALETTE['text']}; }}
    .doc-meta {{ font-size: 0.7rem; color: {PALETTE['text_dim']};
        font-family: 'JetBrains Mono', monospace; }}
    .doc-body {{ font-size: 0.82rem; color: {PALETTE['text_dim']}; line-height: 1.65;
        background: {PALETTE['bg']}; border: 1px solid {PALETTE['border']};
        border-radius: 8px; padding: 0.7rem 0.85rem; }}
    .doc-body mark {{ background: rgba(0,212,255,0.16); color: {PALETTE['text']};
        border-bottom: 1px solid {PALETTE['accent']}; border-radius: 3px;
        padding: 0 0.12rem; }}

    /* ── Women Safety lens ─────────────────────────────────────────────────── */
    .ws-banner {{
        background: linear-gradient(135deg, rgba(244,114,182,0.10), rgba(124,92,255,0.06));
        border: 1px solid rgba(244,114,182,0.30); border-radius: 12px;
        padding: 0.8rem 1.05rem; margin-bottom: 0.7rem;
        animation: netra-rise 0.35s ease-out both;
    }}
    .ws-banner .t {{ font-weight: 800; font-size: 0.98rem; color: #f472b6; }}
    .ws-banner .d {{ font-size: 0.8rem; color: {PALETTE['text_dim']}; margin-top: 0.2rem;
        line-height: 1.55; }}
    .ws-flag {{ display: inline-block; font-size: 0.66rem; font-weight: 700;
        padding: 0.12rem 0.45rem; border-radius: 5px; margin-right: 0.25rem;
        background: rgba(255,59,92,0.14); color: {PALETTE['critical']};
        border: 1px solid rgba(255,59,92,0.35); }}
    .corridor-card {{
        background: {PALETTE['surface']}; border: 1px solid {PALETTE['border']};
        border-left: 3px solid var(--tone, #f472b6); border-radius: 10px;
        padding: 0.75rem 0.95rem; margin-bottom: 0.5rem;
        transition: transform 0.15s ease;
    }}
    .corridor-card:hover {{ transform: translateX(3px); }}
    .corridor-route {{ font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
        color: {PALETTE['text']}; font-weight: 600; }}

    /* ── Alert cards ───────────────────────────────────────────────────────── */
    .alert-card {{
        background: {PALETTE['surface']}; border: 1px solid {PALETTE['border']};
        border-left: 4px solid var(--sev, {PALETTE['text_dim']}); border-radius: 10px;
        padding: 0.85rem 1rem; margin-bottom: 0.7rem;
        animation: netra-rise 0.3s ease-out both;
    }}
    .alert-card .head {{ display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.35rem; }}
    .alert-sev {{ font-size: 0.62rem; font-weight: 700; letter-spacing: 0.08em; padding: 0.15rem 0.5rem;
        border-radius: 5px; color: #0a0e17; text-transform: uppercase; }}
    .alert-title {{ font-weight: 700; font-size: 0.95rem; color: {PALETTE['text']}; }}
    .alert-cat {{ font-size: 0.68rem; color: {PALETTE['text_dim']}; margin-left: auto;
        font-family: 'JetBrains Mono', monospace; }}
    .alert-detail {{ font-size: 0.82rem; color: {PALETTE['text_dim']}; margin-bottom: 0.4rem; }}
    .alert-action {{ font-size: 0.82rem; color: {PALETTE['text']};
        background: rgba(0,212,255,0.06); border-radius: 7px; padding: 0.45rem 0.65rem;
        border: 1px solid rgba(0,212,255,0.15); }}
    .alert-evidence {{ font-size: 0.75rem; color: {PALETTE['text_dim']}; margin-top: 0.4rem;
        padding-left: 0.9rem; }}

    /* ── Actor profile panel ───────────────────────────────────────────────── */
    .profile-card {{ background: {PALETTE['surface']}; border: 1px solid {PALETTE['border']};
        border-radius: 12px; padding: 1rem 1.2rem;
        animation: netra-rise 0.3s ease-out both; }}
    .profile-name {{ font-size: 1.3rem; font-weight: 800; color: {PALETTE['text']}; }}
    .profile-id {{ font-family: 'JetBrains Mono', monospace; color: {PALETTE['accent']};
        font-size: 0.85rem; }}
    .reason-item {{ font-size: 0.83rem; color: {PALETTE['text']}; padding: 0.4rem 0.6rem;
        border-left: 2px solid {PALETTE['accent']}; background: rgba(0,212,255,0.04);
        border-radius: 5px; margin-bottom: 0.35rem; }}

    /* ── Storage / provenance pill ─────────────────────────────────────────── */
    .store-pill {{
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-size: 0.7rem; font-family: 'JetBrains Mono', monospace;
        padding: 0.25rem 0.6rem; border-radius: 20px;
        border: 1px solid var(--tone, {PALETTE['border']});
        background: color-mix(in srgb, var(--tone, {PALETTE['accent']}) 10%, transparent);
        color: var(--tone, {PALETTE['text_dim']});
    }}

    /* ── Streamlit widget overrides ────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: {PALETTE['surface']}; border-right: 1px solid {PALETTE['border']}; }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 0.25rem; background: transparent;
        border-bottom: 1px solid {PALETTE['border']}; flex-wrap: wrap; }}
    .stTabs [data-baseweb="tab"] {{ background: transparent; border-radius: 8px 8px 0 0;
        color: {PALETTE['text_dim']}; font-weight: 600; padding: 0.5rem 0.85rem;
        transition: color 0.15s ease; }}
    .stTabs [data-baseweb="tab"]:hover {{ color: {PALETTE['text']}; }}
    .stTabs [aria-selected="true"] {{ background: {PALETTE['surface_2']};
        color: {PALETTE['accent']}; border-bottom: 2px solid {PALETTE['accent']}; }}

    .stButton > button {{ border-radius: 9px; border: 1px solid {PALETTE['border']};
        background: {PALETTE['surface_2']}; color: {PALETTE['text']}; font-weight: 600;
        transition: transform 0.12s ease, border-color 0.15s ease, color 0.15s ease; }}
    .stButton > button:hover {{ border-color: {PALETTE['accent']}; color: {PALETTE['accent']};
        transform: translateY(-1px); }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {PALETTE['accent']}, {PALETTE['accent_2']});
        color: #05080f; border: none; box-shadow: 0 4px 16px rgba(0,212,255,0.3); }}

    .stDownloadButton > button {{ border-radius: 9px; border: 1px solid {PALETTE['accent']};
        background: rgba(0,212,255,0.08); color: {PALETTE['accent']}; font-weight: 600;
        width: 100%; transition: transform 0.12s ease; }}
    .stDownloadButton > button:hover {{ transform: translateY(-1px); }}

    [data-testid="stDataFrame"] {{ border: 1px solid {PALETTE['border']}; border-radius: 10px; }}
    div[data-testid="stExpander"] {{ border: 1px solid {PALETTE['border']}; border-radius: 10px;
        background: {PALETTE['surface']}; }}

    [data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace;
        color: {PALETTE['text']}; }}
    [data-testid="stMetricLabel"] {{ color: {PALETTE['text_dim']}; }}

    /* Progress bar picks up the brand gradient during the analysis run. */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {PALETTE['accent']}, {PALETTE['accent_2']}); }}

    /* Hide Streamlit chrome for a cleaner console look */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 9px; height: 9px; }}
    ::-webkit-scrollbar-track {{ background: {PALETTE['bg']}; }}
    ::-webkit-scrollbar-thumb {{ background: {PALETTE['border']}; border-radius: 5px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {PALETTE['accent']}; }}
    </style>
    """, unsafe_allow_html=True)


def masthead() -> None:
    """The top brand bar: identity, owning department, PS reference, classification.

    The problem-statement id sits in the masthead rather than buried in an About
    dialog on purpose — an evaluator should be able to tell at a glance which of the
    500 SIH statements this build answers, and every exported artefact carries the
    same reference so a screenshot is traceable to the brief.
    """
    from src.config import (
        APP_DEPARTMENT,
        APP_TAGLINE,
        CLASSIFICATION_BANNER,
        PROBLEM_STATEMENT,
    )
    ps = PROBLEM_STATEMENT
    st.markdown(f"""
    <div class="netra-masthead">
      <div class="netra-brand">
        <div class="netra-logo">🛡️</div>
        <div>
          <div class="netra-title"><span class="accent">NET</span>RA</div>
          <div class="netra-subtitle">{APP_TAGLINE}</div>
          <div class="netra-dept">{APP_DEPARTMENT}</div>
        </div>
      </div>
      <div style="text-align:right;">
        <div class="netra-ps">SIH 2026 · PS {ps['id']} · {ps['theme']}</div>
        <div class="netra-badge"><span class="dot">●</span> {CLASSIFICATION_BANNER}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def ps_strip() -> None:
    """Compact metadata strip for the official problem statement.

    Rendered on the Dashboard so the brief's own fields — id, organisation,
    department, category, theme, deadline — are visible as published rather than
    paraphrased.
    """
    from src.config import PROBLEM_STATEMENT

    ps = PROBLEM_STATEMENT
    cells = [
        ("Problem statement", f"PS {ps['id']}"),
        ("Organisation", ps["organization"]),
        ("Department", ps["department"]),
        ("Category", ps["category"]),
        ("Theme", ps["theme"]),
        ("Submission deadline", ps["deadline"]),
    ]
    html = "".join(
        f'<div class="ps-cell"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v in cells)
    st.markdown(f'<div class="ps-strip">{html}</div>', unsafe_allow_html=True)
