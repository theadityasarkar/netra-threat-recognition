"""
NETRA — AI-Powered Criminal Network Analysis System
Streamlit entry point.

Run with:  streamlit run app.py

Built for Smart India Hackathon 2026, Problem Statement 26189 (Ministry of Home
Affairs · NCRB Women Safety Division). This file is deliberately thin: it wires the
sidebar controls to the (cached) analysis pipeline and dispatches to the eight page
modules under ``src/ui/pages``. Everything heavy lives in ``src`` so the app stays
testable and the pipeline is reusable from a notebook, a CLI or the test-suite.

Production concerns handled here, not scattered through the pages:

* **Session state** — seeded once via :func:`src.ui.state.init_state` so every
  mutable UI value has one declared home.
* **Bounded caching** — :func:`src.ui.state.cached_data` caps retained pipeline
  results so a judge sweeping the sliders cannot OOM the 1 GB free-tier container.
* **UI-free cached compute** — cached functions never touch a Streamlit element, so
  the cache can replay a hit without reaching for a container that no longer exists.
  Progress is reported outside the cache; per-stage timings after the fact.
* **Per-page error isolation** — one view raising must not blank the others; each
  tab body runs under :func:`_safe_render`.
* **Durable case store** — every distinct run is persisted once through the SQLite
  case store, with an append-only audit trail.
"""

from __future__ import annotations

import streamlit as st

from src.config import (
    APP_FULL_NAME,
    APP_ICON,
    APP_NAME,
    APP_ORG,
    APP_VERSION,
    PALETTE,
    PROBLEM_STATEMENT,
    PS_REQUIREMENTS,
)
from src.ingestion.loader import ValidationReport, load_dataset
from src.ingestion.sample_data import generate_sample_dataset
from src.pipeline import run_pipeline
from src.reporting.alerts import build_alerts
from src.storage import database as db
from src.ui import state
from src.ui.components import section_header
from src.ui.pages import (
    anomalies,
    data_sources,
    entities,
    key_players,
    network,
    overview,
    reports,
    women_safety,
)
from src.ui.state import cached_data
from src.ui.theme import inject_theme, masthead

try:
    st.set_page_config(
        page_title=f"{APP_NAME} · Criminal Network Analysis",
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

inject_theme()


# ──────────────────────────────────────────────────────────────────────────────
# Cached compute — the whole point of the thin app layer.
#
# ``cached_data`` is ``st.cache_data`` with the project's memory bounds already
# applied (``max_entries``/``ttl`` from ``src.config.PERF``), so none of these
# caches can grow without limit on a shared host.
# ──────────────────────────────────────────────────────────────────────────────

@cached_data
def _load_sample(n_persons: int, n_syndicates: int, seed: int):
    """Synthetic multi-source dataset — actors, links, cases and intelligence text."""
    data = generate_sample_dataset(n_persons=n_persons, n_syndicates=n_syndicates, seed=seed)
    return (data["persons"], data["relationships"],
            data["crime_records"], data["documents"])


@cached_data
def _ingest_uploads(persons_bytes, rel_bytes, crime_bytes, doc_bytes):
    """Validate and canonicalise uploaded CSVs. Returns the 5-tuple from the loader."""
    return load_dataset(persons_bytes, rel_bytes, crime_bytes, doc_bytes)


@cached_data
def _run(persons, relationships, crimes, documents, *, infer_coaccused: bool,
         contamination: float, gnn_epochs: int, enable_gnn: bool, resolution: float,
         _validation: ValidationReport):
    """Cached full pipeline. Deliberately free of Streamlit calls.

    ``st.cache_data`` records the element writes a cached body makes and replays them
    on a later hit, so a cached function must not touch UI that lives outside it. This
    one used to take a ``_progress_cb`` wired to a :func:`st.progress` bar created by
    :func:`execute`; the recorded write pointed at a container from an earlier script
    run and Streamlit raised ``CacheReplayClosureError`` on the first rerun that hit the
    cache — the app died with "Analysis failed" the moment anyone touched a control.
    Progress is therefore reported outside the cache, and per-stage timings after the
    fact in the Dashboard's provenance expander.

    Leading-underscore ``_validation`` is skipped by Streamlit's hasher: it is metadata
    derived from the same frames, so it cannot change the result on its own.
    """
    return run_pipeline(
        persons, relationships, crimes, validation=_validation, documents=documents,
        infer_coaccused=infer_coaccused, contamination=contamination,
        gnn_epochs=gnn_epochs, enable_gnn=enable_gnn, resolution=resolution)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — data source and analysis parameters
# ──────────────────────────────────────────────────────────────────────────────

def sidebar() -> dict:
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:0.3rem 0 0.8rem 0;">
          <div style="font-size:2rem;">{APP_ICON}</div>
          <div style="font-weight:800;font-size:1.2rem;color:{PALETTE['text']};">
            <span style="color:{PALETTE['accent']}">NET</span>RA</div>
          <div style="font-size:0.66rem;color:{PALETTE['text_dim']};letter-spacing:0.1em;">
            v{APP_VERSION} · PS {PROBLEM_STATEMENT['id']} · {APP_ORG}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("##### 📥 Data source")
        source = st.radio("Data source", ["Sample dataset", "Upload CSV files"],
                          label_visibility="collapsed")

        cfg: dict = {"source": source}

        if source == "Sample dataset":
            st.caption("Synthetic multi-syndicate network with planted anomalies, a "
                       "Women-Safety caseload and intelligence narratives for the demo.")
            cfg["n_persons"] = st.slider("Network size (actors)", 60, 500, 220, 20)
            cfg["n_syndicates"] = st.slider("Number of syndicates", 3, 7, 6, 1)
            cfg["seed"] = int(st.number_input("Random seed", 0, 99999, 42, 1))
        else:
            st.caption("Upload persons + relationships (crime records and intelligence "
                       "reports optional). Headers are auto-mapped; a data-quality report "
                       "is shown after loading.")
            cfg["persons_file"] = st.file_uploader("Persons / actors CSV *", type="csv")
            cfg["rel_file"] = st.file_uploader("Relationships CSV *", type="csv")
            cfg["crime_file"] = st.file_uploader("Crime records CSV (optional)", type="csv")
            cfg["doc_file"] = st.file_uploader(
                "Intelligence reports CSV (optional)", type="csv",
                help="Free-text FIR narratives, surveillance notes or intercepts — a "
                     "`text` column is enough. Powers the Entity Extraction screen.")

        st.markdown("---")
        with st.expander("⚙️ Analysis parameters"):
            cfg["enable_gnn"] = st.toggle("Enable GNN analysis", value=True,
                                          help="2-layer GraphSAGE encoder for embeddings and "
                                               "structural anomaly residuals.")
            cfg["gnn_epochs"] = st.slider("GNN epochs", 50, 400, 200, 25,
                                          disabled=not cfg["enable_gnn"])
            cfg["contamination"] = st.slider("Anomaly contamination", 0.02, 0.30, 0.08, 0.01,
                                             help="Expected share of statistical outliers "
                                                  "(Isolation Forest).")
            cfg["resolution"] = st.slider("Community resolution", 0.4, 2.0, 1.0, 0.1,
                                          help="Higher = more, smaller modules.")
            cfg["infer_coaccused"] = st.toggle("Infer co-accused links from cases", value=True)

        with st.expander("🎛️ Presentation"):
            # Keys are pre-seeded by init_state(), so the widgets initialise from
            # session state — passing ``value=`` as well would trip Streamlit's
            # "default value ignored" warning.
            st.text_input("Case / operation name", key="case_name")
            st.toggle("🕶 Redact identifiers on screen", key="redact",
                      help="Mask names, phone numbers and IDs while screen-sharing. "
                           "Exported reports always retain the full values.")
            st.toggle("📌 Show PS-26189 traceability panel", key="show_ps_panel",
                      help="Requirement→implementation matrix on the Dashboard.")

        st.markdown("---")
        cfg["run"] = st.button("▶  Run Analysis", type="primary", width="stretch")

        _storage_pill()

        st.markdown(f"""
        <div style="margin-top:1rem;font-size:0.68rem;color:{PALETTE['text_dim']};line-height:1.5;">
          <b>{APP_FULL_NAME}</b><br>
          Free stack: Python · Streamlit · NetworkX · NumPy · Plotly.<br>
          All data synthetic. Findings are algorithmic leads, not evidence.
        </div>""", unsafe_allow_html=True)

        return cfg


def _storage_pill() -> None:
    """Report the active persistence regime honestly — durable vs ephemeral vs off."""
    status = db.storage_mode()
    tone = (PALETTE["low"] if status.durable
            else PALETTE["medium"] if status.enabled else PALETTE["text_dim"])
    label = {
        "durable": "CASE STORE · DURABLE",
        "ephemeral": "CASE STORE · EPHEMERAL",
        "disabled": "CASE STORE · OFF",
        "unavailable": "CASE STORE · OFF",
    }.get(status.mode, "CASE STORE")
    st.markdown(f'<span class="store-pill" style="--tone:{tone}">● {label}</span>',
                unsafe_allow_html=True)
    st.caption(status.message)


# ──────────────────────────────────────────────────────────────────────────────
# Data resolution
# ──────────────────────────────────────────────────────────────────────────────

def resolve_data(cfg: dict):
    """Return ``(persons, relationships, crimes, documents, validation)`` or ``None``.

    ``None`` means an upload is still incomplete (no persons/relationships file yet).
    """
    if cfg["source"] == "Sample dataset":
        persons, rel, crimes, documents = _load_sample(
            cfg["n_persons"], cfg["n_syndicates"], cfg["seed"])
        rep = ValidationReport()
        rep.info.append(
            f"Loaded synthetic dataset: {len(persons)} actors, {len(rel)} links, "
            f"{len(crimes)} cases, {len(documents)} intelligence documents "
            f"(seed {cfg['seed']}).")
        return persons, rel, crimes, documents, rep

    pf, rf = cfg.get("persons_file"), cfg.get("rel_file")
    if pf is None or rf is None:
        return None
    crime_bytes = cfg["crime_file"].getvalue() if cfg.get("crime_file") else None
    doc_bytes = cfg["doc_file"].getvalue() if cfg.get("doc_file") else None
    return _ingest_uploads(pf.getvalue(), rf.getvalue(), crime_bytes, doc_bytes)


def data_quality_banner(report: ValidationReport) -> None:
    """Surface validation results — nothing is repaired silently."""
    if report.errors:
        st.error("**Ingestion errors** — fix these and re-run:\n\n"
                 + "\n".join(f"- {e}" for e in report.errors))
    n_issues = len(report.warnings) + len(report.info)
    if n_issues:
        icon = "⚠️" if report.warnings else "✅"
        with st.expander(f"{icon} Data quality report · {len(report.warnings)} warning(s), "
                         f"{len(report.info)} note(s)"):
            for w in report.warnings:
                st.markdown(f"- ⚠️ {w}")
            for i in report.info:
                st.markdown(f"- ℹ️ {i}")
            if report.renamed:
                st.caption("Auto-mapped columns: "
                           + ", ".join(f"`{k}`→`{v}`" for k, v in report.renamed.items()))


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ──────────────────────────────────────────────────────────────────────────────

def execute(persons, rel, crimes, documents, report, cfg):
    """Run the (cached) pipeline behind a spinner.

    The spinner is created here, in the calling frame, and never written to from inside
    the cached body — see :func:`_run` for why that distinction is load-bearing rather
    than stylistic. Returns the :class:`~src.pipeline.AnalysisResult`, or ``None`` on
    failure (already reported on screen).
    """
    try:
        with st.spinner("Running analysis — graph · centrality · communities · GNN · "
                        "anomalies. The first pass takes a few seconds; cached reruns "
                        "are instant."):
            result = _run(
                persons, rel, crimes, documents,
                infer_coaccused=cfg.get("infer_coaccused", True),
                contamination=cfg.get("contamination", 0.08),
                gnn_epochs=cfg.get("gnn_epochs", 200),
                enable_gnn=cfg.get("enable_gnn", True),
                resolution=cfg.get("resolution", 1.0),
                _validation=report,
            )
    except Exception as exc:                                       # noqa: BLE001
        state.mark_run_complete(error=str(exc))
        db.log_action("analysis_failed", f"{type(exc).__name__}: {exc}")
        st.error(f"Analysis failed: {exc}")
        st.exception(exc)
        return None

    return result


def persist_once(result, cfg) -> None:
    """Persist a distinct run to the case store exactly once.

    Streamlit reruns the script on every interaction, so a naive save would write a
    row per mouse click. ``generated_at`` is stamped only when the pipeline body
    actually executes (a cache miss), so it is a natural per-analysis signature.
    """
    sig = f"{result.generated_at}|{result.n_actors}|{result.n_links}"
    if st.session_state.get("_saved_sig") == sig:
        return

    params = {k: cfg.get(k) for k in
              ("source", "n_persons", "n_syndicates", "seed", "enable_gnn",
               "gnn_epochs", "contamination", "resolution", "infer_coaccused")
              if cfg.get(k) is not None}
    data_source = "sample" if cfg["source"] == "Sample dataset" else "upload"
    try:
        alerts = build_alerts(result)
    except Exception:                                              # noqa: BLE001
        alerts = []

    run_id = db.save_run(result, case_name=state.get("case_name", "Untitled case"),
                         data_source=data_source, parameters=params, alerts=alerts)
    st.session_state["_saved_sig"] = sig
    state.put(saved_run_id=run_id or "")
    state.mark_run_complete()


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard tab — overview + PS-26189 traceability
# ──────────────────────────────────────────────────────────────────────────────

def dashboard(result) -> None:
    show_ps = bool(state.get("show_ps_panel", True))
    overview.render(result)
    if show_ps:
        traceability_matrix()


def traceability_matrix() -> None:
    """Render the PS 26189 requirement→implementation matrix as ``.trace-row`` cards.

    Lets an evaluator follow each mandated capability to the module that delivers it
    and the screen that shows it, without reading source. The strings come straight
    from :data:`~src.config.PS_REQUIREMENTS` — the published brief, not a paraphrase.
    """
    section_header("PS 26189 · requirement traceability",
                   "each mandated capability → the module that delivers it → its screen")

    head = (
        f'<div class="trace-row" style="border-bottom:2px solid {PALETTE["border"]};">'
        f'<div class="trace-req" style="color:{PALETTE["text_dim"]};text-transform:uppercase;'
        f'font-size:0.66rem;letter-spacing:0.09em;">PS requirement</div>'
        f'<div class="trace-imp" style="color:{PALETTE["text_dim"]};text-transform:uppercase;'
        f'font-size:0.66rem;letter-spacing:0.09em;">How NETRA delivers it</div>'
        f'<div class="trace-scr" style="color:{PALETTE["text_dim"]};text-transform:uppercase;'
        f'font-size:0.66rem;letter-spacing:0.09em;">Screen</div></div>')
    rows = "".join(
        f'<div class="trace-row">'
        f'<div class="trace-req">{r["requirement"]}</div>'
        f'<div class="trace-imp">{r["delivered_by"]}'
        f'<br><span class="mono" style="font-size:0.68rem;opacity:0.65;">{r["module"]}</span></div>'
        f'<div class="trace-scr">{r["screen"]}</div>'
        f'</div>'
        for r in PS_REQUIREMENTS)
    st.markdown(
        f'<div style="background:{PALETTE["surface"]};border:1px solid {PALETTE["border"]};'
        f'border-radius:12px;padding:0.3rem 0.7rem;margin-top:0.3rem;">{head}{rows}</div>',
        unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Landing screen (before first run / when uploads incomplete)
# ──────────────────────────────────────────────────────────────────────────────

def landing() -> None:
    st.markdown(f"""
    <div style="text-align:center;padding:2.6rem 1rem 1.4rem;">
      <div style="font-size:3.5rem;">{APP_ICON}</div>
      <h1 style="margin:0.3rem 0;">Welcome to <span style="color:{PALETTE['accent']}">NET</span>RA</h1>
      <p style="color:{PALETTE['text_dim']};max-width:660px;margin:0.5rem auto;font-size:1rem;">
        {APP_FULL_NAME}. Load the built-in synthetic dataset or upload criminal records,
        then press <b>Run Analysis</b> to extract entities, build the network graph, rank
        key players, and surface anomalies and Women-Safety threats.
      </p>
    </div>""", unsafe_allow_html=True)

    cols = st.columns(5)
    steps = [
        ("📥", "Ingest", "FIRs · CDRs · finance · intel text — auto-mapped"),
        ("🔎", "Extract", "12 entity types from unstructured narratives"),
        ("🕸️", "Map", "NetworkX multi-relational relationship graph"),
        ("🧠", "Analyse", "Centrality · Louvain · GraphSAGE GNN"),
        ("🚨", "Detect", "Hybrid anomalies · Women-Safety threat lens"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div style="background:{PALETTE['surface']};border:1px solid {PALETTE['border']};
             border-radius:12px;padding:1rem;text-align:center;height:158px;">
          <div style="font-size:1.8rem;">{icon}</div>
          <div style="font-weight:700;margin:0.4rem 0 0.2rem;color:{PALETTE['accent']};">{title}</div>
          <div style="font-size:0.75rem;color:{PALETTE['text_dim']};">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.info("👈 The **sample dataset** is selected by default — just press **▶ Run Analysis** "
            "in the sidebar to see the full system in action.")


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────────────────────────────────────

# (label, icon, render-callable). Labels mirror the screen names in
# ``PS_REQUIREMENTS`` so the traceability matrix and the tab bar stay in step.
PAGES = [
    ("Dashboard", "📊", dashboard),
    ("Data & Sources", "🗂️", data_sources.render),
    ("Entity Extraction", "🔎", entities.render),
    ("Network Graph", "🕸️", network.render),
    ("Key Players", "👑", key_players.render),
    ("Anomalies", "🚨", anomalies.render),
    ("Women Safety", "🚺", women_safety.render),
    ("Reports & Alerts", "📕", reports.render),
]


def _safe_render(render_fn, result, label: str) -> None:
    """Render one page, isolating a failure so the other tabs keep working."""
    try:
        render_fn(result)
    except Exception as exc:                                       # noqa: BLE001
        st.error(f"The **{label}** view could not be rendered: {exc}")
        with st.expander("Technical detail"):
            st.exception(exc)
        db.log_action("page_error", f"{label}: {type(exc).__name__}: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    inject_theme()
    state.init_state()
    masthead()
    cfg = sidebar()

    # First visit: auto-run the sample so the dashboard is never empty, but make an
    # upload wait for the button (there is nothing to analyse until files arrive).
    if not st.session_state.get("_booted"):
        st.session_state["_booted"] = True
        if cfg["source"] == "Sample dataset":
            state.put(has_run=True)
    if cfg["run"]:
        state.put(has_run=True)

    if not state.get("has_run"):
        landing()
        return

    resolved = resolve_data(cfg)
    if resolved is None:
        landing()
        st.warning("Upload both a persons CSV and a relationships CSV, then press "
                   "**Run Analysis**.")
        return

    persons, rel, crimes, documents, report = resolved
    data_quality_banner(report)
    if report.errors:
        return

    result = execute(persons, rel, crimes, documents, report, cfg)
    if result is None:
        return

    persist_once(result, cfg)

    # ── Dispatch to pages, each isolated from the others ────────────────────────
    tabs = st.tabs([f"{icon}  {label}" for label, icon, _ in PAGES])
    for tab, (label, _icon, render_fn) in zip(tabs, PAGES):
        with tab:
            _safe_render(render_fn, result, label)


if __name__ == "__main__":
    main()
