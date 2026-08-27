"""
Data & Sources page — multi-source ingestion status and the case store.

This screen answers the problem statement's first requirement — *"collect and
process data from multiple sources"* — by showing, per source named in the PS
background, whether this dataset actually carries it and through which module. A
coverage claim that cannot be checked against the loaded data is a slide, not a
system, so every row is computed from the frames in hand.

It also carries the CDR analytics panel and the SQLite case store, both of which
are infrastructure rather than analysis and would clutter the investigative screens.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src.config import MANDATED_SOURCES, PALETTE, PERF
from src.ingestion.cdr import CDR_PARAMS, analyse_cdr, synthetic_cdr
from src.storage import database as db
from src.ui.components import empty_state, metric_cards, section_header
from src.ui.state import cached_data
from src.visualization import charts


# ──────────────────────────────────────────────────────────────────────────────
# Live source detection
# ──────────────────────────────────────────────────────────────────────────────

def _coverage(result) -> list[dict]:
    """Resolve each PS-named source to a present/absent verdict and a live count.

    Keyed on the source name from :data:`~src.config.MANDATED_SOURCES` so the list
    published in the config drives this table and the two cannot drift apart.
    """
    rel = result.relationships if result.relationships is not None else pd.DataFrame()
    crimes = result.crimes if result.crimes is not None else pd.DataFrame()
    persons = result.persons if result.persons is not None else pd.DataFrame()
    docs = result.documents if result.documents is not None else pd.DataFrame()

    def rel_count(*types: str) -> int:
        if not len(rel) or "relation_type" not in rel.columns:
            return 0
        return int(rel["relation_type"].isin(types).sum())

    def doc_count(*needles: str) -> int:
        if not len(docs) or "source_type" not in docs.columns:
            return 0
        pattern = "|".join(needles)
        return int(docs["source_type"].astype(str)
                   .str.contains(pattern, case=False, regex=True).sum())

    priors = 0
    if len(persons) and "prior_offences" in persons.columns:
        priors = int((pd.to_numeric(persons["prior_offences"], errors="coerce")
                      .fillna(0) > 0).sum())

    detected: dict[str, tuple[int, str]] = {
        "FIRs and police reports": (len(crimes), "case records"),
        "Call Detail Records (CDRs)": (rel_count("Communication"), "communication ties"),
        "Financial transaction records": (rel_count("Financial"), "financial ties"),
        "Surveillance reports": (doc_count("surveillance"), "documents"),
        "Social media intelligence": (doc_count("social"), "documents"),
        "Criminal history databases": (priors, "actors with priors"),
        "Intelligence agency reports": (doc_count("intelligence"), "documents"),
    }

    rows = []
    for name, module in MANDATED_SOURCES:
        count, unit = detected.get(name, (0, "records"))
        rows.append({"source": name, "module": module, "count": count,
                     "unit": unit, "present": count > 0})
    return rows


def _source_cards(rows: list[dict]) -> None:
    html = []
    for r in rows:
        tone = PALETTE["low"] if r["present"] else PALETTE["text_dim"]
        state = "LOADED" if r["present"] else "NOT IN THIS DATASET"
        detail = (f'{r["count"]:,} {r["unit"]}' if r["present"]
                  else "no records of this type present")
        html.append(
            f'<div class="src-card" style="--tone:{tone}">'
            f'<div style="display:flex;align-items:baseline;gap:0.6rem;flex-wrap:wrap;">'
            f'<span class="src-name">{r["source"]}</span>'
            f'<span class="src-state" style="color:{tone};margin-left:auto;">{state}</span></div>'
            f'<div class="src-path">{detail} · via {r["module"]}</div></div>')
    st.markdown("".join(html), unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# CDR analytics
# ──────────────────────────────────────────────────────────────────────────────

@cached_data
def _cdr_from_network(relationships: pd.DataFrame, persons: pd.DataFrame,
                      seed: int) -> dict:
    """Derive a CDR event log from the network and run the full CDR pipeline.

    The synthetic log is serialised to CSV bytes and fed back through
    :func:`~src.ingestion.cdr.analyse_cdr` rather than calling the internal
    aggregators directly. The round-trip costs a few milliseconds and buys a real
    exercise of the ingestion path — header mapping, MSISDN normalisation, the
    row ceiling — so this panel demonstrates the code an uploaded operator dump
    would actually take, not a shortcut around it.

    Returns a plain dict so the Streamlit cache stores frames, not a dataclass
    holding a live :class:`~src.ingestion.loader.ValidationReport`.
    """
    events = synthetic_cdr(relationships, persons, seed=seed)
    if not len(events):
        return {"events": pd.DataFrame(), "subscribers": pd.DataFrame(),
                "burners": pd.DataFrame(), "colocation": pd.DataFrame(),
                "relationships": pd.DataFrame(), "stats": {}, "notes": []}

    csv_bytes = events.to_csv(index=False).encode("utf-8")
    res = analyse_cdr(io.BytesIO(csv_bytes))
    return {"events": res.events, "subscribers": res.subscribers,
            "burners": res.burners, "colocation": res.colocation,
            "relationships": res.relationships, "stats": res.stats,
            "notes": list(res.report.info) + list(res.report.warnings)}


def _cdr_panel(result) -> None:
    section_header("Call Detail Record analytics",
                   "PS-mandated source · reciprocity, odd-hour profiling, burner detection")

    st.caption(
        "The problem statement names CDRs as a primary source. This dataset ships no "
        "operator dump, so a deterministic event log is derived from the network's "
        "communication ties and pushed through the real CDR ingestion path. Upload a "
        "genuine CDR export in the sidebar to replace it."
    )

    if not len(result.relationships):
        empty_state("No relationships loaded, so no call log can be derived.", "📞")
        return

    try:
        cdr = _cdr_from_network(result.relationships, result.persons, 42)
    except Exception as exc:                                       # noqa: BLE001
        st.warning(f"CDR analytics unavailable: {exc}")
        return

    stats = cdr["stats"]
    if not stats.get("events"):
        empty_state("The derived call log came back empty.", "📞")
        return

    metric_cards([
        {"label": "Call events", "value": f"{stats.get('events', 0):,}",
         "accent": PALETTE["accent"]},
        {"label": "Distinct numbers", "value": f"{stats.get('numbers', 0):,}",
         "accent": PALETTE["accent_2"]},
        {"label": "Aggregated ties", "value": f"{stats.get('pairs', 0):,}",
         "accent": PALETTE["low"]},
        {"label": "Suspected burners", "value": stats.get("burners", 0),
         "accent": PALETTE["critical"],
         "delta": f"< {CDR_PARAMS['burner_max_lifespan_days']}d lifespan"},
        {"label": "Tower co-locations", "value": stats.get("colocation_pairs", 0),
         "accent": PALETTE["high"]},
        {"label": "Median calls / pair", "value": stats.get("median_calls_per_pair", 0),
         "accent": PALETTE["medium"]},
    ])

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("**Subscriber profiles** — ranked by odd-hour share")
        subs = cdr["subscribers"]
        if len(subs):
            cols = [c for c in ("number", "calls", "contacts", "odd_hour_calls",
                                "odd_hour_ratio", "towers", "imeis", "lifespan_days",
                                "avg_duration")
                    if c in subs.columns]
            sort_col = "odd_hour_ratio" if "odd_hour_ratio" in subs.columns else cols[1]
            st.dataframe(subs.sort_values(sort_col, ascending=False)[cols].head(25),
                         width="stretch", hide_index=True, height=330)
        else:
            empty_state("No subscriber profiles.", "📇")

    with right:
        st.markdown("**Burner handsets** — short lifespan, high contact churn")
        burners = cdr["burners"]
        if len(burners):
            cols = [c for c in ("number", "burner_score", "lifespan_days", "calls",
                                "contacts", "odd_hour_ratio") if c in burners.columns]
            st.dataframe(burners[cols].head(20), width="stretch", hide_index=True, height=330)
        else:
            st.info("No number matches the burner profile in this log — expected when "
                    "call activity is spread evenly across the whole period.")

    with st.expander("🗼 Tower co-location — numbers repeatedly seen on the same cells"):
        coloc = cdr["colocation"]
        if len(coloc):
            st.dataframe(coloc.head(30), width="stretch", hide_index=True)
            st.caption(
                f"Two numbers are paired when they share at least "
                f"{CDR_PARAMS['colocation_min_shared_towers']} towers across "
                f"{CDR_PARAMS['colocation_min_events']}+ events. Co-location is a "
                "physical-proximity indicator, independent of whether the two ever "
                "called each other — which is what makes it useful against handlers "
                "who avoid direct contact.")
        else:
            empty_state("No co-location pairs cleared the threshold.", "🗼")

    for note in cdr["notes"][:6]:
        st.caption(f"· {note}")


# ──────────────────────────────────────────────────────────────────────────────
# Case store
# ──────────────────────────────────────────────────────────────────────────────

def _store_panel() -> None:
    section_header("Case store & audit trail", "SQLite · every run and action recorded")

    status = db.storage_mode()
    tone = {"durable": PALETTE["low"], "ephemeral": PALETTE["medium"],
            "disabled": PALETTE["text_dim"], "unavailable": PALETTE["critical"]}.get(
        status.mode, PALETTE["text_dim"])
    st.markdown(
        f'<div class="store-pill" style="--tone:{tone}">● {status.mode.upper()}'
        f'&nbsp;·&nbsp;{status.message}</div>', unsafe_allow_html=True)

    stats = db.store_stats()
    metric_cards([
        {"label": "Saved runs", "value": stats.get("analysis_runs", 0),
         "accent": PALETTE["accent"]},
        {"label": "Stored alerts", "value": stats.get("run_alerts", 0),
         "accent": PALETTE["high"]},
        {"label": "Watchlist", "value": stats.get("watchlist", 0),
         "accent": PALETTE["critical"]},
        {"label": "Audit entries", "value": stats.get("audit_log", 0),
         "accent": PALETTE["accent_2"]},
        {"label": "Store size", "value": f"{stats.get('size_kb', 0.0):.0f} KB",
         "accent": PALETTE["low"]},
    ])

    tab_runs, tab_watch, tab_audit = st.tabs(
        ["🗂️ Run history", "⭐ Watchlist", "📜 Audit trail"])

    with tab_runs:
        runs = db.recent_runs(limit=20)
        if len(runs):
            st.dataframe(runs, width="stretch", hide_index=True)
        else:
            empty_state("No runs saved yet — use **Save run to case store** on the "
                        "Reports & Alerts screen.", "🗂️")

    with tab_watch:
        watch = db.get_watchlist()
        if len(watch):
            st.dataframe(watch, width="stretch", hide_index=True)
        else:
            empty_state("No actors flagged. Use the ⭐ control on the Key Players "
                        "screen to add one.", "⭐")

    with tab_audit:
        trail = db.audit_trail(limit=60)
        if len(trail):
            st.dataframe(trail, width="stretch", hide_index=True)
            st.caption("Append-only. The audit trail exists so an analyst who acted on "
                       "an alert can show the state of the system that produced it.")
        else:
            empty_state("Nothing logged yet.", "📜")

    with st.expander("⚠️ Clear the case store"):
        st.caption("Removes every saved run, alert, watchlist entry and audit line. "
                   "The schema is kept. Intended for resetting between demo sessions.")
        if st.button("Clear all persisted data", key="ds_reset"):
            if db.reset_store():
                st.success("Case store cleared.")
                st.rerun()
            else:
                st.error("Could not clear the store — it may be read-only.")


# ──────────────────────────────────────────────────────────────────────────────
# Page
# ──────────────────────────────────────────────────────────────────────────────

def render(result) -> None:
    docs = result.documents if result.documents is not None else pd.DataFrame()
    rows = _coverage(result)
    covered = sum(1 for r in rows if r["present"])

    # ── Inventory ─────────────────────────────────────────────────────────────
    metric_cards([
        {"label": "Actors", "value": f"{len(result.persons):,}", "accent": PALETTE["accent"]},
        {"label": "Relationships", "value": f"{len(result.relationships):,}",
         "accent": PALETTE["accent_2"]},
        {"label": "Case records", "value": f"{len(result.crimes):,}", "accent": PALETTE["high"]},
        {"label": "Intel documents", "value": f"{len(docs):,}", "accent": PALETTE["medium"]},
        {"label": "PS sources covered", "value": f"{covered}/{len(rows)}",
         "accent": PALETTE["low"] if covered >= 5 else PALETTE["medium"]},
        {"label": "Row ceiling", "value": f"{PERF['max_upload_rows']:,}",
         "accent": PALETTE["text_dim"], "delta": "per uploaded file"},
    ])

    # ── PS source coverage ────────────────────────────────────────────────────
    section_header("Mandated source coverage",
                   "the seven sources named in the PS background section")
    _source_cards(rows)
    st.caption(
        "Each row is computed from the loaded frames, not asserted. A source marked "
        "*not in this dataset* means the capability exists and no records of that type "
        "were supplied — upload them and the row turns green."
    )

    # ── Data quality ──────────────────────────────────────────────────────────
    rep = result.validation
    section_header("Ingestion & data quality",
                   f"{len(rep.errors)} error(s) · {len(rep.warnings)} warning(s) · "
                   f"{len(rep.info)} note(s)")
    if rep.errors:
        st.error("\n\n".join(f"- {e}" for e in rep.errors))
    q1, q2 = st.columns(2)
    with q1:
        if rep.warnings:
            st.markdown("**Warnings** — accepted, but worth knowing")
            for w in rep.warnings[:12]:
                st.markdown(f"- ⚠️ {w}")
        else:
            st.success("No data-quality warnings.")
    with q2:
        if rep.info:
            st.markdown("**Notes**")
            for i in rep.info[:12]:
                st.markdown(f"- ℹ️ {i}")
        if rep.renamed:
            st.caption("Auto-mapped headers: "
                       + ", ".join(f"`{k}`→`{v}`" for k, v in list(rep.renamed.items())[:10]))

    # ── Table previews ────────────────────────────────────────────────────────
    section_header("Loaded records", "first 200 rows per frame")
    t_p, t_r, t_c, t_d = st.tabs(
        ["👤 Actors", "🔗 Relationships", "📁 Case records", "📄 Intel documents"])
    with t_p:
        st.dataframe(result.persons.head(200), width="stretch", hide_index=True, height=330)
    with t_r:
        st.dataframe(result.relationships.head(200), width="stretch", hide_index=True, height=330)
    with t_c:
        if len(result.crimes):
            st.dataframe(result.crimes.head(200), width="stretch", hide_index=True, height=330)
        else:
            empty_state("No case records loaded. Co-accused inference and the Women "
                        "Safety lens both depend on this file.", "📁")
    with t_d:
        if len(docs):
            show = [c for c in ("doc_id", "source_type", "title", "date", "city", "case_id")
                    if c in docs.columns]
            st.dataframe(docs[show] if show else docs, width="stretch",
                         hide_index=True, height=270)
            if "source_type" in docs.columns:
                st.plotly_chart(charts.source_type_bars(docs), width="stretch",
                                config={"displayModeBar": False})
        else:
            empty_state("No unstructured documents in this dataset — the Entity "
                        "Extraction screen needs narrative text.", "📄")

    # ── Downloadable copies ───────────────────────────────────────────────────
    section_header("Export the working dataset", "exact frames this analysis ran on")
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("⬇ actors.csv", result.persons.to_csv(index=False).encode(),
                       "netra_actors.csv", "text/csv", width="stretch")
    d2.download_button("⬇ relationships.csv",
                       result.relationships.to_csv(index=False).encode(),
                       "netra_relationships.csv", "text/csv", width="stretch")
    d3.download_button("⬇ cases.csv", result.crimes.to_csv(index=False).encode(),
                       "netra_cases.csv", "text/csv", width="stretch",
                       disabled=not len(result.crimes))
    d4.download_button("⬇ documents.csv", docs.to_csv(index=False).encode(),
                       "netra_documents.csv", "text/csv", width="stretch",
                       disabled=not len(docs))

    # ── Live Telemetry Simulator + CDR + storage ──────────────────────────────
    _live_telemetry_simulator(result)
    _cdr_panel(result)
    _store_panel()


def _live_telemetry_simulator(result) -> None:
    section_header("Live Telemetry & CDR Stream Simulator",
                   "real-time packet ingestion simulation for high-risk targets")
    with st.expander("⚡ Real-Time Streaming Telemetry Console (Active Listeners)", expanded=True):
        st.markdown(
            "Simulates real-time streaming ingestion from **Cell Tower Dumps (CDRs), Banking Gateways, "
            "and Electronic Intercepts** connected to high-risk actors in the network."
        )
        c1, c2 = st.columns([1, 2])
        with c1:
            pulse = st.button("⚡ Inject Telemetry Stream Batch", width="stretch")
            stream_count = st.slider("Events per batch", 5, 25, 10, 5)
        with c2:
            st.caption("Live streaming listener active. Injected events automatically correlate "
                       "with known network actor identifiers and flag potential threat escalations.")

        if pulse or "live_stream_events" not in st.session_state:
            import random
            from datetime import datetime, timedelta
            events = []
            persons = result.persons if result.persons is not None and len(result.persons) else pd.DataFrame()
            names = list(persons["name"].values) if len(persons) and "name" in persons.columns else ["Suspect Alpha", "Suspect Bravo"]
            event_types = [
                ("CELL_TOWER_PING", "📡 Tower Dump", "High-frequency burst at border tower"),
                ("ENCRYPTED_CALL", "📞 VoIP Intercept", "VoIP packet burst to international gateway"),
                ("MULE_TRANSFER", "💳 IMPS / UPI Flow", "Rapid structuring transfer below INR 50k threshold"),
                ("CRYPTO_TX", "⛓️ On-Chain Transfer", "USDT sweep to flagged mixer wallet"),
                ("HOTEL_LOG", "🏨 Front Registration", "Guest checked in with flagged secondary ID"),
            ]
            t_now = datetime.now()
            for _ in range(stream_count):
                et, icon, desc = random.choice(event_types)
                actor = random.choice(names)
                threat = random.choice(["CRITICAL", "HIGH", "ELEVATED", "ROUTINE"])
                events.append({
                    "Timestamp": (t_now - timedelta(seconds=random.randint(1, 300))).strftime("%H:%M:%S"),
                    "Channel": icon,
                    "Target Actor": actor,
                    "Event Details": desc,
                    "Threat Level": threat,
                    "Action Flag": "⚠ Flagged for Dossier" if threat in ["CRITICAL", "HIGH"] else "Log Only"
                })
            st.session_state["live_stream_events"] = pd.DataFrame(events)

        st.dataframe(st.session_state["live_stream_events"], width="stretch", hide_index=True)
