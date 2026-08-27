"""
Women Safety Division lens — the analytical layer for the division that owns PS 26189.

PS 26189 is filed by the **NCRB Women Safety Division**. A generic criminal-network
tool would answer the letter of the brief and miss its owner: the division's caseload
is trafficking, sexual exploitation, dowry and domestic-violence rings, and the fast-
growing cyber-enabled category (stalking, sextortion, obscene-content distribution)
that connects this problem statement to its *Blockchain & Cybersecurity* theme.

This module adds three division-specific capabilities on top of the general analysis:

1. **Women-Safety Threat Index (WSTI, 0–100)** — a per-actor score built only from
   offences within the division's mandate, so a high-influence narcotics kingpin does
   not crowd out a low-degree trafficking recruiter. Components and weights live in
   ``config.WSTI_WEIGHTS`` and are auditable.
2. **Trafficking-corridor detection** — victim movement is geographic. Actors whose
   cases span several cities, connected to each other, describe a *route*. Corridors
   are ranked by span, severity and the presence of minor victims.
3. **Repeat-offender and cyber-enabled cohorts** — the two escalation lists the
   division works from day to day.

Design note — why a separate index rather than another anomaly rule
------------------------------------------------------------------
The anomaly engine answers "who is structurally strange". That is the wrong question
for this division: a trafficking recruiter is *not* structurally anomalous — being
well-connected to many low-profile contacts is exactly normal for the role. WSTI is
therefore a *content* score (what offences, against whom) blended with a small
structural term, deliberately kept independent of the suspicion score so the two can
disagree. When they disagree, that is information, and the UI shows both.

All functions degrade to empty, correctly-typed frames when the input carries no
women-safety cases, so callers never special-case an all-narcotics dataset.
"""

from __future__ import annotations

import re

import networkx as nx
import pandas as pd

from src.config import (
    CRIME_SEVERITY,
    CYBER_WOMEN_SAFETY_CRIMES,
    MINOR_VICTIM_CRIMES,
    WOMEN_SAFETY_CRIMES,
    WS_PARAMS,
    WSTI_WEIGHTS,
)

WS_COLUMNS = [
    "person_id", "name", "wsti", "ws_cases", "ws_case_ids", "peak_severity",
    "crime_types", "minor_victim", "cyber_enabled", "cities", "city_span",
    "degree", "prior_offences", "band", "flags",
]

CORRIDOR_COLUMNS = [
    "corridor_id", "route", "n_cities", "n_actors", "actor_ids", "peak_severity",
    "minor_victim", "case_ids", "risk_score",
]

# ── Mandate matching vocabulary ───────────────────────────────────────────────
# Used only as a *fallback* when a label is not an exact WOMEN_SAFETY_CRIMES match,
# so an agency variant like "Human Trafficking (Women)" still reaches the division.

# Unambiguous: the label alone establishes the division's interest.
_WS_STRONG = (
    "women", "woman", "girl", "bride", "dowry", "domestic violence", "sexual",
    "stalking", "harassment", "sextortion", "obscene", "matrimonial", "rape",
    "molest", "pocso", "acid attack", "minor victim", "child",
)

# Ambiguous on its own. Narcotics, arms and wildlife are all "trafficked", so the
# bare word cannot decide the division's mandate — it must not pull a drugs case
# into a Women Safety caseload.
_WS_AMBIGUOUS = ("traffick",)

# Commodity qualifiers that disqualify an ambiguous match: these name a trafficked
# *thing*, not a trafficked person.
_WS_COMMODITY = (
    "narcotic", "drug", "arms", "firearm", "weapon", "ammunition", "wildlife",
    "timber", "fuel", "liquor", "counterfeit", "currency", "gold", "bullion",
    "antiqu", "cattle", "smuggl",
)


def _in_mandate(label: str) -> bool:
    """Whether a free-text crime label falls under the Women Safety Division.

    Exact vocabulary matches are handled by the caller; this is the fallback for
    agency-specific spellings. A strong indicator admits the label outright; an
    ambiguous one ("trafficking") admits it only when no commodity is named, which
    is what keeps "Narcotics Trafficking" out of a women-safety caseload.
    """
    text = label.lower()
    if any(k in text for k in _WS_STRONG):
        return True
    if any(k in text for k in _WS_AMBIGUOUS):
        return not any(c in text for c in _WS_COMMODITY)
    return False


def _minmax(s: pd.Series) -> pd.Series:
    """Scale to 0–1. A flat series maps to 0.0, not to NaN or to 1.0 —
    "no variation" must not read as "maximum threat"."""
    if s.empty:
        return s
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return pd.Series(0.0, index=s.index, dtype=float)
    return (s.astype(float) - lo) / (hi - lo)


def _severity_of(crime_type: str, fallback: int = 3) -> int:
    return int(CRIME_SEVERITY.get(str(crime_type), fallback))


def _joined(values: list[str], cap: int) -> str:
    """Join identifiers for display, stating plainly when the list was truncated.

    Every table here reports a *count* beside its *id list* — ``ws_cases`` next to
    ``ws_case_ids``, ``n_actors`` next to ``actor_ids``. A silently capped list makes the
    two columns contradict each other, and an analyst reads the short list as the complete
    roster, or as a defect. Naming the overflow keeps the cell readable without ever
    understating the evidence.

    The marker is prefixed with :data:`_OVERFLOW_PREFIX` so the one place that re-parses a
    joined column (:func:`trafficking_corridors`) can drop it rather than mistake it for an
    identifier.
    """
    if len(values) <= cap:
        return "; ".join(values)
    return "; ".join(values[:cap]) + f"; {_OVERFLOW_PREFIX} +{len(values) - cap} more"


def _split_ids(joined: str, separator: str = ";") -> list[str]:
    """Split a joined id column back into ids, discarding the overflow marker."""
    return [tok.strip() for tok in str(joined).split(separator)
            if tok.strip() and not tok.strip().startswith(_OVERFLOW_PREFIX)]


# Cell separators used across the accused-list column, matching the loader's contract.
_ID_SEPARATORS = re.compile(r"[;,|/]+")

# Leader for the "list was truncated" marker appended by :func:`_joined`.
_OVERFLOW_PREFIX = "…"


def _case_actor_ids(case: pd.Series) -> list[str]:
    """Resolve the accused on one case to a list of person IDs.

    ``load_dataset`` attaches a pre-parsed, roster-validated ``_ids`` column, and that
    is the preferred source. But this module is also reachable from call sites that
    hand ``run_pipeline`` raw frames directly — the bundled sample dataset and the
    tests both do — where only the delimited ``persons_involved`` text exists. Falling
    back to parsing it here mirrors what
    :func:`src.graph.builder.build_graph` already does, so the division's caseload can
    never silently come back empty just because the frame skipped the loader.
    """
    ids = case.get("_ids")
    if isinstance(ids, (list, tuple)):
        return [str(i) for i in ids]

    raw = case.get("persons_involved")
    if isinstance(raw, (list, tuple)):
        return [str(i).strip() for i in raw if str(i).strip()]
    if isinstance(raw, str) and raw.strip():
        return [tok.strip() for tok in _ID_SEPARATORS.split(raw) if tok.strip()]
    return []


def women_safety_cases(crimes: pd.DataFrame) -> pd.DataFrame:
    """Filter the crime frame to the division's mandate.

    Matching is exact against ``WOMEN_SAFETY_CRIMES`` with a substring fallback, so
    an agency label like "Human Trafficking (Women)" that is not in the vocabulary
    still lands in the division's caseload rather than being silently dropped.
    """
    if crimes is None or crimes.empty or "crime_type" not in crimes.columns:
        return pd.DataFrame(columns=list(crimes.columns) if crimes is not None else [])

    ctype = crimes["crime_type"].astype(str)
    exact = ctype.isin(WOMEN_SAFETY_CRIMES)
    # Substring fallback for agency-specific label variants — commodity-aware, so a
    # narcotics or arms trafficking case is never absorbed into this division.
    fuzzy = ctype.apply(_in_mandate)
    return crimes[exact | fuzzy].copy()


def compute_wsti(graph: nx.Graph,
                 crimes: pd.DataFrame,
                 centralities: pd.DataFrame | None = None) -> pd.DataFrame:
    """Women-Safety Threat Index for every actor touched by a division case.

    Args:
        graph: the built criminal network (supplies degree and actor attributes).
        crimes: canonical crime frame. Uses the pre-parsed ``_ids`` column that
            :func:`src.ingestion.loader.prepare_dataset` attaches when present, and
            otherwise parses ``persons_involved`` directly — see
            :func:`_case_actor_ids`.
        centralities: optional; supplies the structural reach component.

    Returns:
        One row per involved actor, sorted by WSTI descending. Actors with no
        division case are omitted entirely — this is a caseload, not a roster.
    """
    ws = women_safety_cases(crimes)
    if ws.empty or graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=WS_COLUMNS)

    per_actor: dict[str, dict] = {}

    for _, case in ws.iterrows():
        ids = _case_actor_ids(case)
        ctype = str(case.get("crime_type", ""))
        sev = int(case.get("severity") or _severity_of(ctype))
        city = str(case.get("city") or "").strip()
        case_id = str(case.get("case_id", "") or "").strip()
        is_minor = ctype in MINOR_VICTIM_CRIMES or "minor" in ctype.lower()
        is_cyber = ctype in CYBER_WOMEN_SAFETY_CRIMES

        for pid in ids:
            pid = str(pid)
            if pid not in graph:
                continue
            rec = per_actor.setdefault(pid, {
                "ws_cases": 0, "case_ids": [], "peak_severity": 0, "sev_sum": 0,
                "crime_types": set(), "cities": set(),
                "minor_victim": False, "cyber_enabled": False,
            })
            rec["ws_cases"] += 1
            if case_id:                      # a file with no id still counts, but never
                rec["case_ids"].append(case_id)   # contributes a blank to the id list
            rec["peak_severity"] = max(rec["peak_severity"], sev)
            rec["sev_sum"] += sev
            rec["crime_types"].add(ctype)
            if city:
                rec["cities"].add(city)
            rec["minor_victim"] = rec["minor_victim"] or is_minor
            rec["cyber_enabled"] = rec["cyber_enabled"] or is_cyber

    if not per_actor:
        return pd.DataFrame(columns=WS_COLUMNS)

    rows = []
    for pid, rec in per_actor.items():
        attrs = graph.nodes[pid]
        rows.append({
            "person_id": pid,
            "name": attrs.get("name", pid),
            "ws_cases": rec["ws_cases"],
            "ws_case_ids": _joined(rec["case_ids"], 8),
            "peak_severity": rec["peak_severity"],
            "mean_severity": rec["sev_sum"] / max(1, rec["ws_cases"]),
            "crime_types": ", ".join(sorted(rec["crime_types"])),
            "minor_victim": bool(rec["minor_victim"]),
            "cyber_enabled": bool(rec["cyber_enabled"]),
            "cities": ", ".join(sorted(rec["cities"])),
            "city_span": len(rec["cities"]),
            "degree": int(graph.degree(pid)),
            "prior_offences": int(attrs.get("prior_offences", 0) or 0),
        })
    df = pd.DataFrame(rows).set_index("person_id", drop=False)

    # ── Components, each scaled 0–1 over the involved cohort ──────────────────
    involvement = _minmax(df["ws_cases"].astype(float))
    severity = (df["peak_severity"].astype(float) / 5.0).clip(0, 1)
    minor = df["minor_victim"].astype(float)

    if centralities is not None and len(centralities) and "pagerank" in centralities.columns:
        reach_raw = centralities["pagerank"].reindex(df.index).fillna(0.0)
    else:
        reach_raw = df["degree"].astype(float)
    reach = _minmax(reach_raw)

    recidivism = _minmax(df["prior_offences"].astype(float))

    w = WSTI_WEIGHTS
    score = (w["case_involvement"] * involvement
             + w["severity"] * severity
             + w["minor_victim"] * minor
             + w["network_reach"] * reach
             + w["recidivism"] * recidivism)
    df["wsti"] = (score * 100).round(1)

    df["band"] = pd.cut(df["wsti"], bins=[-0.01, 30, 50, WS_PARAMS["high_threat_wsti"], 100],
                        labels=["Low", "Moderate", "High", "Severe"]).astype(str)
    df["flags"] = df.apply(_ws_flags, axis=1)

    return (df[WS_COLUMNS]
            .sort_values("wsti", ascending=False)
            .reset_index(drop=True))


def _ws_flags(row: pd.Series) -> list[str]:
    """Human-readable escalation reasons — the same explainability contract the
    anomaly engine holds to. A WSTI with no stated basis is not actionable."""
    flags: list[str] = []
    if row.get("minor_victim"):
        flags.append("Offence involving a minor — statutory aggravator")
    if int(row.get("ws_cases", 0)) >= int(WS_PARAMS["repeat_offender_min_cases"]):
        flags.append(f"Repeat offender — {int(row['ws_cases'])} division cases")
    if int(row.get("peak_severity", 0)) >= 5:
        flags.append("Involved in a maximum-severity offence")
    if int(row.get("city_span", 0)) >= 2:
        flags.append(f"Active across {int(row['city_span'])} cities — inter-district movement")
    if row.get("cyber_enabled"):
        flags.append("Cyber-enabled offence — digital evidence preservation required")
    if int(row.get("degree", 0)) >= 8:
        flags.append(f"Well-connected ({int(row['degree'])} links) — recruitment capability")
    if not flags:
        flags.append("Single lower-severity case; monitor only")
    return flags


def trafficking_corridors(graph: nx.Graph,
                          crimes: pd.DataFrame,
                          wsti: pd.DataFrame) -> pd.DataFrame:
    """Detect geographic movement corridors among connected division actors.

    Victim movement leaves a spatial signature that a purely topological analysis
    cannot see: a set of *connected* actors whose cases span several cities is a
    route, not a coincidence. We take the subgraph induced by division actors, walk
    its connected components, and report any component spanning at least
    ``corridor_min_cities`` cities with at least ``corridor_min_actors`` actors.

    Ranking blends span (how far the route reaches), peak severity, and the
    presence of minor victims — the three factors that decide whether a corridor
    warrants an inter-state operation.
    """
    if wsti is None or wsti.empty or graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=CORRIDOR_COLUMNS)

    actors = [p for p in wsti["person_id"].astype(str) if p in graph]
    if len(actors) < int(WS_PARAMS["corridor_min_actors"]):
        return pd.DataFrame(columns=CORRIDOR_COLUMNS)

    sub = graph.subgraph(actors)
    by_actor = wsti.set_index("person_id")

    min_cities = int(WS_PARAMS["corridor_min_cities"])
    min_actors = int(WS_PARAMS["corridor_min_actors"])

    rows = []
    for component in nx.connected_components(sub):
        members = sorted(component)
        if len(members) < min_actors:
            continue

        cities, cases, peak, minor = set(), set(), 0, False
        for pid in members:
            if pid not in by_actor.index:
                continue
            rec = by_actor.loc[pid]
            cities.update(_split_ids(rec.get("cities", ""), separator=","))
            cases.update(_split_ids(rec.get("ws_case_ids", "")))
            peak = max(peak, int(rec.get("peak_severity", 0)))
            minor = minor or bool(rec.get("minor_victim", False))

        if len(cities) < min_cities:
            continue

        ordered = sorted(cities)
        risk = (0.45 * min(1.0, len(cities) / 6.0)
                + 0.35 * (peak / 5.0)
                + 0.20 * (1.0 if minor else 0.0))
        rows.append({
            "corridor_id": "",          # assigned after ranking — see below
            "route": " → ".join(ordered),
            "n_cities": len(cities),
            "n_actors": len(members),
            "actor_ids": _joined(members, 12),
            "peak_severity": peak,
            "minor_victim": minor,
            "case_ids": _joined(sorted(cases), 10),
            "risk_score": round(risk * 100, 1),
        })

    if not rows:
        return pd.DataFrame(columns=CORRIDOR_COLUMNS)

    # Identify corridors by their risk rank, not by the order the components happened
    # to be enumerated in. Component enumeration order is not stable across processes,
    # which meant the same corridor was labelled CORR-02 on one run and CORR-01 on the
    # next — unusable in a tool whose output gets quoted in a dossier. Ranking by risk
    # (route as a deterministic tie-break) makes the identifier both reproducible and
    # meaningful: CORR-01 is always the corridor that most warrants an operation.
    out = (pd.DataFrame(rows, columns=CORRIDOR_COLUMNS)
           .sort_values(["risk_score", "route"], ascending=[False, True])
           .reset_index(drop=True))
    out["corridor_id"] = [f"CORR-{i:02d}" for i in range(1, len(out) + 1)]
    return out


def women_safety_summary(wsti: pd.DataFrame,
                         corridors: pd.DataFrame,
                         crimes: pd.DataFrame) -> dict:
    """Headline counters for the Women Safety dashboard panel."""
    ws_cases = women_safety_cases(crimes)
    if wsti is None or wsti.empty:
        return {
            "actors": 0, "cases": int(len(ws_cases)), "severe": 0, "high_or_worse": 0,
            "minor_victim_actors": 0, "cyber_actors": 0, "repeat_offenders": 0,
            "corridors": 0, "top_crime": "—", "peak_wsti": 0.0,
            "case_share": 0.0,
        }

    total_cases = int(len(crimes)) if crimes is not None else 0
    top_crime = "—"
    if not ws_cases.empty and "crime_type" in ws_cases.columns:
        top_crime = str(ws_cases["crime_type"].value_counts().idxmax())

    return {
        "actors": int(len(wsti)),
        "cases": int(len(ws_cases)),
        "case_share": round(100.0 * len(ws_cases) / total_cases, 1) if total_cases else 0.0,
        "severe": int((wsti["band"] == "Severe").sum()),
        "high_or_worse": int(wsti["band"].isin(["Severe", "High"]).sum()),
        "minor_victim_actors": int(wsti["minor_victim"].sum()),
        "cyber_actors": int(wsti["cyber_enabled"].sum()),
        "repeat_offenders": int((wsti["ws_cases"] >= WS_PARAMS["repeat_offender_min_cases"]).sum()),
        "corridors": 0 if corridors is None or corridors.empty else int(len(corridors)),
        "top_crime": top_crime,
        "peak_wsti": float(wsti["wsti"].max()),
    }


def crime_type_breakdown(crimes: pd.DataFrame) -> pd.DataFrame:
    """Division caseload by offence type, with the cyber-enabled subset marked.

    The cyber column is what lets the Women Safety panel speak to the PS theme
    (*Blockchain & Cybersecurity*) with data rather than assertion.
    """
    cols = ["crime_type", "cases", "avg_severity", "cyber_enabled", "minor_victim"]
    ws = women_safety_cases(crimes)
    if ws.empty or "crime_type" not in ws.columns:
        return pd.DataFrame(columns=cols)

    rows = []
    for ctype, grp in ws.groupby("crime_type"):
        sev = (grp["severity"].astype(float).mean() if "severity" in grp.columns
               else float(_severity_of(str(ctype))))
        rows.append({
            "crime_type": str(ctype),
            "cases": int(len(grp)),
            "avg_severity": round(float(sev), 2),
            "cyber_enabled": str(ctype) in CYBER_WOMEN_SAFETY_CRIMES,
            "minor_victim": str(ctype) in MINOR_VICTIM_CRIMES,
        })
    return (pd.DataFrame(rows, columns=cols)
            .sort_values("cases", ascending=False)
            .reset_index(drop=True))


def escalation_list(wsti: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """The actors the division should act on first.

    Ordered by WSTI but with statutory aggravators promoted: any actor tied to an
    offence against a minor outranks an equally-scored actor without one, because
    the legal escalation path (POCSO) is different and time-bound.
    """
    if wsti is None or wsti.empty:
        return pd.DataFrame(columns=WS_COLUMNS)
    ranked = wsti.sort_values(
        ["minor_victim", "wsti"], ascending=[False, False]).head(top_n).copy()
    ranked.insert(0, "escalation_rank", range(1, len(ranked) + 1))
    return ranked.reset_index(drop=True)
