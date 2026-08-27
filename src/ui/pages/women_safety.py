"""
Women Safety Division page — the owning department's own view of the network.

PS 26189 sits with NCRB's **Women Safety Division**, and a generic network-analysis
tool is not the same thing as one built for that mandate. This screen re-scores the
same graph through the division's lens:

* the **Women-Safety Threat Index (WSTI)** ranks actors on their exposure to
  women- and minor-victim offences, deliberately *independent* of the structural
  anomaly score — a well-behaved node in the graph can still be the division's
  highest priority, and conflating the two hides exactly that case;
* **trafficking corridors** surface the geographic routes actors move victims
  along, which no single-actor score can express;
* the **escalation list** promotes statutory aggravators over raw score, because
  POCSO carries a different and time-bound legal path.

Every WSTI carries its flags. A score with no stated basis is not actionable, and
an investigator asked to act on an unexplained number correctly refuses.
"""

from __future__ import annotations

import html as _html

import streamlit as st

from src.config import PALETTE, WS_PARAMS, WSTI_WEIGHTS
from src.ui.components import empty_state, metric_cards, section_header
from src.visualization import charts

_BAND_TONE = {
    "Severe": PALETTE["critical"],
    "High": PALETTE["high"],
    "Moderate": PALETTE["medium"],
    "Low": PALETTE["low"],
}


def _banner(stats: dict) -> None:
    st.markdown(f"""
    <div class="ws-banner">
      <div class="t">⚠ Women Safety Division · mandated caseload lens</div>
      <div class="d">
        {stats.get('cases', 0):,} of the loaded case records fall inside the division's
        mandate ({stats.get('case_share', 0.0):.1f}% of the total caseload), implicating
        {stats.get('actors', 0):,} actors. {stats.get('minor_victim_actors', 0)} are tied to an
        offence against a minor and {stats.get('cyber_actors', 0)} to a cyber-enabled offence.
        Most frequent offence: <b>{_html.escape(str(stats.get('top_crime', '—')))}</b>.
      </div>
    </div>""", unsafe_allow_html=True)


def _flags_html(flags) -> str:
    if isinstance(flags, str):
        flags = [f.strip() for f in flags.split("|") if f.strip()]
    return "".join(f'<span class="ws-flag">{_html.escape(str(f))}</span>'
                   for f in (flags or []))


def _as_list(value) -> list[str]:
    """Coerce a WSTI/corridor field to a list of items.

    ``src.analysis.women_safety`` emits these fields already joined — ``cities`` and
    ``crime_types`` with ``", "``, ``actor_ids`` and ``case_ids`` with ``"; "`` — while
    ``flags`` stays a real list. Slicing a joined string as though it were a list yields
    individual characters (``"Amritsar, Jaipur"[:4]`` → ``"Amri"`` → *"A, m, r, i"*), so
    the coercion has to happen before any truncation. Both delimiters are accepted so
    this keeps working if the producer changes which one it uses.
    """
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.split(";") if ";" in value else value.split(",")
        return [p.strip() for p in parts if p.strip()]
    try:
        return [str(v) for v in value]
    except TypeError:                                    # scalar (e.g. NaN)
        return [str(value)]


def _truncated(items: list[str], limit: int, sep: str = ", ") -> str:
    """Join at most ``limit`` items, marking that more exist."""
    if not items:
        return "—"
    return sep.join(items[:limit]) + ("…" if len(items) > limit else "")


def _escalation(result) -> None:
    esc = result.ws_escalation
    if not len(esc):
        empty_state("No actor cleared the escalation threshold.", "✅")
        return

    for _, row in esc.iterrows():
        tone = _BAND_TONE.get(str(row.get("band")), PALETTE["accent"])
        cities_txt = _truncated(_as_list(row.get("cities")), 4)
        crimes_txt = _truncated(_as_list(row.get("crime_types")), 3)
        st.markdown(f"""
        <div class="corridor-card" style="--tone:{tone}">
          <div style="display:flex;align-items:baseline;gap:0.6rem;flex-wrap:wrap;">
            <span style="font-weight:800;font-size:0.94rem;">
              #{int(row.get('escalation_rank', 0))} · {_html.escape(str(row.get('name', '')))}</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.74rem;
              color:{PALETTE['text_dim']};">{_html.escape(str(row.get('person_id', '')))}</span>
            <span style="margin-left:auto;font-weight:800;color:{tone};font-size:1.02rem;">
              WSTI {float(row.get('wsti', 0)):.1f}</span>
            <span style="font-size:0.7rem;color:{tone};font-weight:700;
              text-transform:uppercase;letter-spacing:0.06em;">
              {_html.escape(str(row.get('band', '')))}</span>
          </div>
          <div style="font-size:0.78rem;color:{PALETTE['text_dim']};margin:0.3rem 0 0.4rem;">
            {int(row.get('ws_cases', 0))} division case(s) · peak severity
            {int(row.get('peak_severity', 0))}/5 · {int(row.get('degree', 0))} network links ·
            {int(row.get('prior_offences', 0))} prior offence(s)<br>
            <b>Offences:</b> {_html.escape(crimes_txt)} &nbsp;·&nbsp;
            <b>Active in:</b> {_html.escape(cities_txt)}
          </div>
          <div>{_flags_html(row.get('flags'))}</div>
        </div>""", unsafe_allow_html=True)


def _corridors(result) -> None:
    cor = result.corridors
    if not len(cor):
        st.info(
            f"No corridor met the threshold — a route needs at least "
            f"{WS_PARAMS['corridor_min_cities']} distinct cities and "
            f"{WS_PARAMS['corridor_min_actors']} connected actors. A negative result here "
            "is a finding: the division's caseload in this dataset is locally contained "
            "rather than moving victims across districts.")
        return

    st.plotly_chart(charts.corridor_bars(result), width="stretch",
                    config={"displayModeBar": False})

    for _, row in cor.iterrows():
        tone = PALETTE["critical"] if row.get("minor_victim") else PALETTE["high"]
        actors = _as_list(row.get("actor_ids"))
        cases = _as_list(row.get("case_ids"))
        minor = ('<span class="ws-flag">Minor victim involved</span>'
                 if row.get("minor_victim") else "")
        st.markdown(f"""
        <div class="corridor-card" style="--tone:{tone}">
          <div style="display:flex;align-items:baseline;gap:0.6rem;flex-wrap:wrap;">
            <span style="font-weight:800;">{_html.escape(str(row.get('corridor_id', '')))}</span>
            <span style="margin-left:auto;font-weight:800;color:{tone};">
              risk {float(row.get('risk_score', 0)):.0f}</span>
          </div>
          <div class="corridor-route" style="margin:0.3rem 0;">
            {_html.escape(str(row.get('route', '')))}</div>
          <div style="font-size:0.78rem;color:{PALETTE['text_dim']};margin-bottom:0.35rem;">
            {int(row.get('n_cities', 0))} cities · {int(row.get('n_actors', 0))} actors ·
            peak severity {int(row.get('peak_severity', 0))}/5 ·
            {len(cases)} case(s) · actors: {_html.escape(_truncated(actors, 8))}
          </div>
          <div>{minor}</div>
        </div>""", unsafe_allow_html=True)

    if (cor["risk_score"] >= 99.5).any():
        st.caption(
            "A corridor saturating at risk 100 usually means the division caseload in "
            "this dataset forms a single connected component rather than several "
            "discrete routes. Treat the geography as the finding and the score as a "
            "ceiling artefact — the route list below the header is the part to act on.")


def _wsti_table(result) -> None:
    ws = result.wsti
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        bands = st.multiselect("Bands", ["Severe", "High", "Moderate", "Low"],
                               default=["Severe", "High"], key="ws_bands")
    with c2:
        floor = st.slider("Minimum WSTI", 0.0, 100.0, 0.0, 5.0, key="ws_floor")
    with c3:
        only = st.multiselect("Aggravators", ["Minor victim", "Cyber-enabled",
                                              "Repeat offender", "Multi-city"],
                              default=[], key="ws_aggr")

    view = ws.copy()
    if bands:
        view = view[view["band"].astype(str).isin(bands)]
    view = view[view["wsti"] >= floor]
    if "Minor victim" in only:
        view = view[view["minor_victim"]]
    if "Cyber-enabled" in only:
        view = view[view["cyber_enabled"]]
    if "Repeat offender" in only:
        view = view[view["ws_cases"] >= WS_PARAMS["repeat_offender_min_cases"]]
    if "Multi-city" in only:
        view = view[view["city_span"] >= 2]

    if not len(view):
        empty_state("No actor matches these filters.", "🔍")
        return

    show = view[["person_id", "name", "wsti", "band", "ws_cases", "peak_severity",
                 "minor_victim", "cyber_enabled", "city_span", "degree",
                 "prior_offences"]].copy()
    show.columns = ["ID", "Actor", "WSTI", "Band", "Division cases", "Peak severity",
                    "Minor victim", "Cyber", "Cities", "Links", "Priors"]
    st.dataframe(
        show, width="stretch", hide_index=True, height=400,
        column_config={
            "WSTI": st.column_config.ProgressColumn("WSTI", min_value=0.0,
                                                    max_value=100.0, format="%.1f"),
            "Minor victim": st.column_config.CheckboxColumn("Minor victim"),
            "Cyber": st.column_config.CheckboxColumn("Cyber"),
        })
    st.caption(f"{len(view):,} of {len(ws):,} actors inside the division's mandate.")

    # ``flags`` is the one genuinely list-valued column; every other multi-value field
    # arrives from the analysis layer pre-joined. Writing a Python list into a CSV cell
    # would emit ``['Repeat offender…', …]`` brackets and quotes, so it is flattened here.
    st.download_button(
        "⬇ Download WSTI ranking (CSV)",
        ws.assign(flags=ws["flags"].map(
            lambda f: " | ".join(map(str, f)) if isinstance(f, list) else f),
        ).to_csv(index=False).encode(),
        "netra_wsti_ranking.csv", "text/csv")


def render(result) -> None:
    stats = result.ws_stats or {}

    if not len(result.wsti):
        empty_state(
            "No case records in this dataset fall inside the Women Safety Division's "
            "mandate, so the WSTI cannot be computed. Load the sample dataset, or "
            "upload cases whose `crime_type` includes women- or minor-victim offences.",
            "⚠️")
        with st.expander("ℹ️ What this screen would show"):
            st.markdown(
                "The Women-Safety Threat Index scores each actor on division case "
                "involvement, peak offence severity, minor-victim aggravators, network "
                "reach and recidivism — then lists trafficking corridors and an "
                "escalation queue ordered so statutory aggravators outrank raw score.")
        return

    _banner(stats)

    metric_cards([
        {"label": "Division cases", "value": f"{stats.get('cases', 0):,}",
         "accent": "#f472b6", "delta": f"{stats.get('case_share', 0.0):.1f}% of caseload"},
        {"label": "Implicated actors", "value": f"{stats.get('actors', 0):,}",
         "accent": PALETTE["accent"]},
        {"label": "Severe band", "value": stats.get("severe", 0),
         "accent": PALETTE["critical"],
         "delta": f"{stats.get('high_or_worse', 0)} high or worse"},
        {"label": "Minor-victim actors", "value": stats.get("minor_victim_actors", 0),
         "accent": PALETTE["critical"], "delta": "POCSO escalation path"},
        {"label": "Cyber-enabled", "value": stats.get("cyber_actors", 0),
         "accent": PALETTE["accent_2"], "delta": "digital evidence at risk"},
        {"label": "Repeat offenders", "value": stats.get("repeat_offenders", 0),
         "accent": PALETTE["high"],
         "delta": f"≥{WS_PARAMS['repeat_offender_min_cases']} division cases"},
        {"label": "Corridors", "value": stats.get("corridors", 0),
         "accent": PALETTE["medium"]},
        {"label": "Peak WSTI", "value": f"{stats.get('peak_wsti', 0.0):.1f}",
         "accent": PALETTE["critical"]},
    ])

    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.plotly_chart(charts.wsti_ranking(result), width="stretch",
                        config={"displayModeBar": False})
    with c2:
        st.plotly_chart(charts.ws_band_donut(result), width="stretch",
                        config={"displayModeBar": False})

    t_esc, t_cor, t_case, t_rank = st.tabs([
        "🚨 Escalation queue", "🛣️ Trafficking corridors",
        "📊 Caseload profile", "📋 Full WSTI ranking"])

    with t_esc:
        section_header("Escalation queue",
                       "statutory aggravators promoted above raw score")
        _escalation(result)

    with t_cor:
        section_header("Trafficking corridors",
                       f"≥{WS_PARAMS['corridor_min_cities']} cities · "
                       f"≥{WS_PARAMS['corridor_min_actors']} connected actors")
        _corridors(result)

    with t_case:
        section_header("Division caseload by offence",
                       "cyber-enabled offences highlighted — the PS theme, evidenced")
        st.plotly_chart(charts.ws_crime_breakdown(result), width="stretch",
                        config={"displayModeBar": False})
        if len(result.ws_breakdown):
            show = result.ws_breakdown.copy()
            show.columns = ["Offence", "Cases", "Avg severity", "Cyber-enabled",
                            "Minor victim"]
            st.dataframe(show, width="stretch", hide_index=True,
                         column_config={
                             "Cyber-enabled": st.column_config.CheckboxColumn("Cyber-enabled"),
                             "Minor victim": st.column_config.CheckboxColumn("Minor victim")})
        with st.expander("📁 The mandated case records themselves"):
            if len(result.ws_cases):
                st.dataframe(result.ws_cases.head(300), width="stretch", hide_index=True,
                             height=330)
                st.download_button("⬇ Download division caseload (CSV)",
                                   result.ws_cases.to_csv(index=False).encode(),
                                   "netra_women_safety_cases.csv", "text/csv")
            else:
                empty_state("No mandated case records isolated.", "📁")

    with t_rank:
        section_header("Full WSTI ranking", "filterable · every actor in the mandate")
        _wsti_table(result)

    with st.expander("⚙️ How the WSTI is computed, and why it is separate"):
        weights = "\n".join(
            f"| {k.replace('_', ' ').capitalize()} | {v:.0%} |"
            for k, v in WSTI_WEIGHTS.items())
        st.markdown(f"""
**Composition.** Five normalised components, weighted:

| Component | Weight |
|---|---|
{weights}

Each is min-max normalised across the actors inside the mandate, so the index ranks
*within this caseload* rather than against an absolute national scale — the right
behaviour for a division working a specific set of files.

**Why it is not folded into the suspicion score.** The anomaly engine asks a
structural question: does this actor's position in the network look wrong? The WSTI
asks a content question: how deep is this actor into offences the division is
mandated to act on. The two genuinely disagree — a trafficker with an unremarkable
graph position scores low on suspicion and high on WSTI, and averaging them would
bury exactly the actor this division exists to catch. They are reported side by side
and left for the analyst to reconcile.

**Bands.** Severe ≥ {WS_PARAMS['high_threat_wsti']:.0f}, High 50–{WS_PARAMS['high_threat_wsti']:.0f},
Moderate 30–50, Low below 30. The escalation threshold is
{WS_PARAMS['high_threat_wsti']:.0f}, but the escalation *queue* is ordered with
minor-victim cases first regardless of score, because POCSO timelines are statutory
and do not wait for a higher number.

**Corridor detection.** A connected-component walk over actors in the mandate: a
component whose members' case cities span at least {WS_PARAMS['corridor_min_cities']}
distinct locations, with at least {WS_PARAMS['corridor_min_actors']} actors, is
reported as a movement corridor. Composite risk combines geographic span, actor
count, peak severity and minor-victim involvement.

**The limitation worth stating.** Corridors are inferred from case *cities*, which
record where an offence was registered — not the route travelled. A corridor is
therefore a hypothesis to check against CDR tower data and transport records, not a
mapped route. The system says so rather than implying a precision it does not have.
""")
