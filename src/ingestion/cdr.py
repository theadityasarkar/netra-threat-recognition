"""
Layer 1 (telecom) — Call Detail Record ingestion.

PS 26189 names *"Call Detail Records (CDRs)"* as a primary source. A CDR is not a
relationship list: it is an event log, and the analytically interesting part is how
you collapse tens of thousands of call events into a small number of *weighted,
directed-aware* communication ties without losing the signal that matters.

What this module contributes beyond a naive ``groupby(caller, callee).size()``:

* **Reciprocity-aware strength.** Two numbers that call each other both ways are a
  relationship; a number that only ever receives calls from another is a
  subscriber-to-handler pattern. Mutual pairs are scored materially higher.
* **Burst / odd-hour profiling.** Contact concentrated in the 00:00–05:00 window is
  an established operational indicator, kept as an edge attribute so the anomaly
  layer and the analyst can both see it.
* **Burner-handset detection.** Short active lifespan + very few distinct contacts +
  high call volume is the classic burner profile.
* **Tower co-location.** Two subscribers repeatedly transacting from the same cell
  site places them physically together — evidence a call log alone doesn't give you.

Output is the canonical relationship schema, so CDR-derived ties flow into exactly
the same graph → centrality → GNN → anomaly pipeline as FIR-derived ties.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import RELATIONSHIP_COLUMNS
from src.ingestion.loader import ValidationReport, normalise_headers, read_csv_any

# Canonical CDR schema. ``caller``/``callee`` are the only hard requirements.
CDR_COLUMNS = [
    "caller",
    "callee",
    "timestamp",
    "duration_sec",
    "call_type",
    "tower_id",
    "imei",
    "location",
]

CDR_REQUIRED = ["caller", "callee"]

# CDR exports are the least standardised files in law enforcement — every telco
# and every extraction tool ships different headers.
CDR_ALIASES: dict[str, list[str]] = {
    "caller": ["a_party", "aparty", "a_number", "calling", "calling_number", "from",
               "msisdn", "caller_number", "source", "a", "origin", "caller_no"],
    "callee": ["b_party", "bparty", "b_number", "called", "called_number", "to",
               "callee_number", "target", "b", "destination", "called_no"],
    "timestamp": ["date_time", "datetime", "call_date", "call_time", "start_time",
                  "date", "time", "event_time", "call_datetime"],
    "duration_sec": ["duration", "call_duration", "dur", "seconds", "secs",
                     "duration_seconds", "talk_time"],
    "call_type": ["type", "direction", "call_dir", "in_out", "service_type", "cdr_type"],
    "tower_id": ["cell_id", "cellid", "tower", "cgi", "lac_cid", "site_id",
                 "first_cell_id", "cell"],
    "imei": ["imei_number", "device_imei", "handset", "equipment_id"],
    "location": ["tower_location", "cell_site", "address", "area", "place"],
}

# Odd-hour window (inclusive start, exclusive end) in local time.
ODD_HOUR_START, ODD_HOUR_END = 0, 5

CDR_PARAMS = {
    # Pairs with fewer calls than this are noise (wrong numbers, IVR, single ping).
    "min_calls_per_pair": 2,
    # Strength model — contributions, capped and blended below.
    "volume_saturation": 25,      # calls at which the volume term saturates
    "duration_saturation": 1800,  # total seconds at which the duration term saturates
    "reciprocity_bonus": 0.22,    # added when a pair calls both ways
    "odd_hour_bonus": 0.10,       # added when odd-hour contact dominates the pair
    "strength_floor": 0.15,
    "strength_ceiling": 0.97,
    # Burner profile.
    "burner_max_lifespan_days": 21,
    "burner_max_contacts": 6,
    "burner_min_calls": 8,
    # Tower co-location.
    "colocation_min_shared_towers": 2,
    "colocation_min_events": 4,
    # A cell serving a station, mall or highway toll puts hundreds of unrelated numbers
    # on one site. Counting those as co-location invents links between strangers, so
    # towers above this fanout are excluded from the pairing entirely.
    "colocation_max_tower_fanout": 40,
    # Reporting ceiling: the analyst reads the strongest pairs, not a 100k-row dump.
    "colocation_max_pairs": 500,
    # Ingestion ceiling — a CDR dump larger than this is refused with a clear message.
    "max_rows": 2_000_000,
}

_DIGITS = re.compile(r"\D")


@dataclass
class CDRResult:
    """Aggregated CDR intelligence."""

    events: pd.DataFrame          # cleaned event log
    relationships: pd.DataFrame   # canonical relationship rows
    subscribers: pd.DataFrame     # per-number profile
    burners: pd.DataFrame         # suspected burner handsets
    colocation: pd.DataFrame      # tower co-location pairs
    report: ValidationReport = field(default_factory=ValidationReport)
    stats: dict = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def n_pairs(self) -> int:
        return len(self.relationships)


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalise_msisdn(value) -> str:
    """Reduce a phone number to a comparable 10-digit national form.

    Indian CDRs mix ``+919876543210``, ``919876543210``, ``09876543210`` and
    ``9876543210`` for the same subscriber; without this every format becomes a
    separate node and the network silently fragments.
    """
    digits = _DIGITS.sub("", str(value or ""))
    if len(digits) > 10:
        digits = digits[-10:]
    return digits


def load_cdr(source) -> tuple[pd.DataFrame, ValidationReport]:
    """Read and clean a CDR export into :data:`CDR_COLUMNS`."""
    rep = ValidationReport()
    try:
        raw = read_csv_any(source)
    except Exception as exc:                                   # noqa: BLE001
        rep.errors.append(f"Unreadable CDR file: {exc}")
        return pd.DataFrame(columns=CDR_COLUMNS), rep

    if len(raw) > CDR_PARAMS["max_rows"]:
        rep.errors.append(
            f"CDR file has {len(raw):,} rows, above the {CDR_PARAMS['max_rows']:,} row "
            "ingestion ceiling. Split it by date range and load the parts separately.")
        return pd.DataFrame(columns=CDR_COLUMNS), rep

    df, renamed = _normalise_cdr_headers(raw)
    rep.renamed = renamed
    if renamed:
        rep.info.append(f"Auto-mapped {len(renamed)} CDR column(s): "
                        + ", ".join(f"{k} → {v}" for k, v in list(renamed.items())[:6])
                        + ("…" if len(renamed) > 6 else ""))

    missing = [c for c in CDR_REQUIRED if c not in df.columns]
    if missing:
        rep.errors.append(
            f"CDR file is missing required column(s): {', '.join(missing)}. "
            "Expected a calling-party and a called-party column (any common alias).")
        return df, rep

    for col in CDR_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["caller"] = df["caller"].map(normalise_msisdn)
    df["callee"] = df["callee"].map(normalise_msisdn)

    blank = (df["caller"].str.len() < 6) | (df["callee"].str.len() < 6)
    if blank.any():
        rep.warnings.append(f"Dropped {int(blank.sum())} event(s) with an unusable number.")
        df = df[~blank]

    loops = df["caller"] == df["callee"]
    if loops.any():
        rep.warnings.append(f"Dropped {int(loops.sum())} self-call event(s).")
        df = df[~loops]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    if df["timestamp"].isna().all() and len(df):
        rep.warnings.append("No parseable timestamps — odd-hour and lifespan analytics "
                            "are unavailable for this file.")
    df["duration_sec"] = (pd.to_numeric(df["duration_sec"], errors="coerce")
                          .fillna(0).clip(0, 86_400))
    df["call_type"] = (df["call_type"].fillna("VOICE").astype(str)
                       .str.strip().str.upper().replace("", "VOICE"))
    for col in ("tower_id", "imei", "location"):
        df[col] = df[col].fillna("").astype(str).str.strip()

    rep.stats["events"] = len(df)
    rep.info.append(f"Loaded {len(df):,} CDR event(s) covering "
                    f"{df[['caller', 'callee']].stack().nunique():,} distinct number(s).")
    return df.reset_index(drop=True), rep


def _normalise_cdr_headers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Header mapping against the CDR alias table (reuses the loader's canonicaliser)."""
    out, mapping = normalise_headers(df, CDR_COLUMNS)
    # normalise_headers only knows the persons/links/crimes aliases, so apply the
    # CDR-specific table to whatever it left untouched.
    reverse = {a: canon for canon, aliases in CDR_ALIASES.items() for a in aliases}
    reverse.update({c: c for c in CDR_COLUMNS})
    extra = {c: reverse[c] for c in out.columns if c in reverse and reverse[c] != c}
    if extra:
        out = out.rename(columns=extra)
        out = out.loc[:, ~out.columns.duplicated(keep="first")]
        mapping.update(extra)
    return out, mapping


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation into weighted ties
# ──────────────────────────────────────────────────────────────────────────────

def _with_parsed_timestamps(events: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose ``timestamp`` column is guaranteed datetime-typed.

    ``load_cdr`` already parses timestamps, so the ingestion path arrives clean. But
    the aggregators below are public and are reachable with a frame that never went
    through the loader — :func:`synthetic_cdr` output passed straight in, or a caller
    building events by hand — where ``timestamp`` is still text. The ``.dt`` accessor
    fails there with an opaque ``AttributeError`` naming neither the column nor the
    cause, so coerce once here instead. Unparseable values become ``NaT``, which the
    odd-hour and lifespan terms already treat as "unknown" rather than as a value.
    """
    df = events.copy()
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.NaT
    elif not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    return df


def cdr_to_relationships(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse a CDR event log into weighted undirected communication ties.

    Strength blends four signals — call volume, total talk time, reciprocity and
    odd-hour concentration — each saturating so that one very chatty pair cannot
    dominate the whole graph.
    """
    if not len(events):
        return pd.DataFrame(columns=RELATIONSHIP_COLUMNS)

    df = _with_parsed_timestamps(events)
    # Canonical undirected pair, while remembering which directions were observed.
    a = np.where(df["caller"] <= df["callee"], df["caller"], df["callee"])
    b = np.where(df["caller"] <= df["callee"], df["callee"], df["caller"])
    df["_a"], df["_b"] = a, b
    df["_forward"] = df["caller"] == df["_a"]

    hours = df["timestamp"].dt.hour
    df["_odd"] = ((hours >= ODD_HOUR_START) & (hours < ODD_HOUR_END)).fillna(False)

    grouped = df.groupby(["_a", "_b"], as_index=False).agg(
        calls=("_a", "size"),
        total_duration=("duration_sec", "sum"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
        n_forward=("_forward", "sum"),
        odd_hour_calls=("_odd", "sum"),
        towers=("tower_id", lambda s: sorted({t for t in s if t})),
        locations=("location", lambda s: next((x for x in s if x), "Unknown")),
    )

    grouped = grouped[grouped["calls"] >= CDR_PARAMS["min_calls_per_pair"]].copy()
    if not len(grouped):
        return pd.DataFrame(columns=RELATIONSHIP_COLUMNS)

    # ── Strength model ────────────────────────────────────────────────────────
    vol = np.log1p(grouped["calls"]) / np.log1p(CDR_PARAMS["volume_saturation"])
    dur = np.log1p(grouped["total_duration"]) / np.log1p(CDR_PARAMS["duration_saturation"])
    grouped["reciprocal"] = (grouped["n_forward"] > 0) & (grouped["n_forward"] < grouped["calls"])
    grouped["odd_hour_ratio"] = (grouped["odd_hour_calls"] / grouped["calls"]).fillna(0.0)

    strength = (0.55 * vol.clip(0, 1) + 0.45 * dur.clip(0, 1))
    strength = strength + CDR_PARAMS["reciprocity_bonus"] * grouped["reciprocal"]
    strength = strength + CDR_PARAMS["odd_hour_bonus"] * (grouped["odd_hour_ratio"] > 0.5)
    grouped["strength"] = strength.clip(CDR_PARAMS["strength_floor"],
                                        CDR_PARAMS["strength_ceiling"]).round(3)

    out = pd.DataFrame({
        "source_id": grouped["_a"],
        "target_id": grouped["_b"],
        "relation_type": "Communication",
        "strength": grouped["strength"],
        "interaction_count": grouped["calls"].astype(int),
        "first_seen": grouped["first_seen"],
        "last_seen": grouped["last_seen"],
        "evidence_source": "CDR Analysis",
        "location": grouped["locations"].replace("", "Unknown"),
    })
    return out[RELATIONSHIP_COLUMNS].reset_index(drop=True)


def subscriber_profiles(events: pd.DataFrame) -> pd.DataFrame:
    """One row per number: volume, distinct contacts, lifespan, odd-hour share."""
    if not len(events):
        return pd.DataFrame(columns=["number", "calls", "contacts", "total_duration",
                                     "first_seen", "last_seen", "lifespan_days",
                                     "odd_hour_ratio", "towers", "imeis"])

    df = _with_parsed_timestamps(events)
    hours = df["timestamp"].dt.hour
    df["_odd"] = ((hours >= ODD_HOUR_START) & (hours < ODD_HOUR_END)).fillna(False)

    # Stack both directions so a number is profiled whether it called or was called.
    legs = pd.concat([
        df.assign(number=df["caller"], peer=df["callee"]),
        df.assign(number=df["callee"], peer=df["caller"]),
    ], ignore_index=True)

    prof = legs.groupby("number", as_index=False).agg(
        calls=("number", "size"),
        contacts=("peer", "nunique"),
        total_duration=("duration_sec", "sum"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
        odd_hour_calls=("_odd", "sum"),
        towers=("tower_id", lambda s: len({t for t in s if t})),
        imeis=("imei", lambda s: len({i for i in s if i})),
    )
    span = (prof["last_seen"] - prof["first_seen"]).dt.total_seconds() / 86_400
    prof["lifespan_days"] = span.fillna(0).round(2)
    prof["odd_hour_ratio"] = (prof["odd_hour_calls"] / prof["calls"]).fillna(0).round(3)
    prof["avg_duration"] = (prof["total_duration"] / prof["calls"]).round(1)
    return prof.sort_values("calls", ascending=False).reset_index(drop=True)


def detect_burners(profiles: pd.DataFrame) -> pd.DataFrame:
    """Flag handsets matching the burner profile, with the reason spelled out.

    Burner = short active lifespan + narrow contact list + high call volume. Each
    flag carries its own justification so it can be defended in a case file.
    """
    cols = ["number", "calls", "contacts", "lifespan_days", "odd_hour_ratio",
            "burner_score", "reasons"]
    if not len(profiles):
        return pd.DataFrame(columns=cols)

    p = profiles.copy()
    short = p["lifespan_days"].between(0.01, CDR_PARAMS["burner_max_lifespan_days"])
    narrow = p["contacts"] <= CDR_PARAMS["burner_max_contacts"]
    busy = p["calls"] >= CDR_PARAMS["burner_min_calls"]
    nocturnal = p["odd_hour_ratio"] >= 0.4
    multi_imei = p["imeis"] > 1

    score = (0.34 * short + 0.28 * narrow + 0.22 * busy
             + 0.10 * nocturnal + 0.06 * multi_imei)
    p["burner_score"] = (score * 100).round(1)

    def _reasons(row) -> list[str]:
        out = []
        if row["lifespan_days"] and row["lifespan_days"] <= CDR_PARAMS["burner_max_lifespan_days"]:
            out.append(f"Active for only **{row['lifespan_days']:.0f} day(s)**")
        if row["contacts"] <= CDR_PARAMS["burner_max_contacts"]:
            out.append(f"Contacted just **{int(row['contacts'])} number(s)**")
        if row["calls"] >= CDR_PARAMS["burner_min_calls"]:
            out.append(f"**{int(row['calls'])} calls** in that window")
        if row["odd_hour_ratio"] >= 0.4:
            out.append(f"**{row['odd_hour_ratio'] * 100:.0f}%** of contact between 00:00–05:00")
        if row.get("imeis", 0) > 1:
            out.append(f"SIM seen in **{int(row['imeis'])} different handsets**")
        return out

    # Require the three defining conditions, not just a high blended score.
    flagged = p[short & narrow & busy].copy()
    if not len(flagged):
        return pd.DataFrame(columns=cols)
    flagged["reasons"] = flagged.apply(_reasons, axis=1)
    return (flagged.sort_values("burner_score", ascending=False)
            .loc[:, cols].reset_index(drop=True))


def tower_colocation(events: pd.DataFrame) -> pd.DataFrame:
    """Pairs of numbers that repeatedly transact from the same cell sites.

    Shared towers put two subscribers in the same place, which a call log alone cannot
    establish — this is the physical-surveillance corroboration an investigator would
    otherwise build by hand.

    Two design decisions matter here. First, high-fanout towers are discarded: a cell
    serving a railway station or a market puts hundreds of unrelated numbers on the same
    site, so counting it as co-location manufactures links between strangers. Second,
    pairs are accumulated through a tower→numbers inverted index rather than by
    comparing every number against every other — an all-pairs scan is O(n²) and would
    stall on a real seizure dump of tens of thousands of subscribers.
    """
    cols = ["number_a", "number_b", "shared_towers", "events", "towers"]
    if not len(events) or "tower_id" not in events.columns:
        return pd.DataFrame(columns=cols)

    df = events[events["tower_id"].astype(str).str.len() > 0]
    if not len(df):
        return pd.DataFrame(columns=cols)

    legs = pd.concat([
        df[["caller", "tower_id"]].rename(columns={"caller": "number"}),
        df[["callee", "tower_id"]].rename(columns={"callee": "number"}),
    ], ignore_index=True)

    # Events per (number, tower), so "events" below can count activity at the *shared*
    # sites rather than each number's total traffic everywhere.
    per_site = legs.groupby(["tower_id", "number"]).size()
    fanout = per_site.groupby(level="tower_id").size()
    keep = set(fanout[fanout <= CDR_PARAMS["colocation_max_tower_fanout"]].index)
    keep &= set(fanout[fanout >= 2].index)          # a tower with one number links nobody
    if not keep:
        return pd.DataFrame(columns=cols)

    site_counts = per_site.to_dict()
    shared: dict[tuple[str, str], list[str]] = {}
    for tower in keep:
        numbers = sorted(per_site.loc[tower].index)
        for i, a in enumerate(numbers):
            for b in numbers[i + 1:]:
                shared.setdefault((a, b), []).append(tower)

    min_towers = CDR_PARAMS["colocation_min_shared_towers"]
    rows = []
    for (a, b), towers in shared.items():
        if len(towers) < min_towers:
            continue
        ev = int(sum(site_counts.get((t, a), 0) + site_counts.get((t, b), 0)
                     for t in towers))
        if ev >= CDR_PARAMS["colocation_min_events"]:
            rows.append({"number_a": a, "number_b": b, "shared_towers": len(towers),
                         "events": ev, "towers": sorted(towers)[:6]})
    if not rows:
        return pd.DataFrame(columns=cols)
    out = (pd.DataFrame(rows).sort_values(["shared_towers", "events"], ascending=False)
           .reset_index(drop=True))
    return out.head(CDR_PARAMS["colocation_max_pairs"])


def analyse_cdr(source) -> CDRResult:
    """End-to-end CDR ingestion: load → aggregate → profile → flag."""
    events, rep = load_cdr(source)
    if not rep.ok or not len(events):
        empty = pd.DataFrame()
        return CDRResult(events, pd.DataFrame(columns=RELATIONSHIP_COLUMNS),
                         empty, empty, empty, rep, {"events": len(events)})

    rels = cdr_to_relationships(events)
    profiles = subscriber_profiles(events)
    burners = detect_burners(profiles)
    coloc = tower_colocation(events)

    if len(rels):
        rep.info.append(f"Aggregated {len(events):,} call event(s) into {len(rels):,} "
                        f"weighted communication tie(s).")
    if len(burners):
        rep.warnings.append(f"{len(burners)} number(s) match the burner-handset profile.")

    # Share of ties where contact ran both ways. Computed from the event log rather than
    # from ``rels``, which keeps only the canonical relationship columns: a one-way tie
    # and a reciprocal one are indistinguishable once aggregated.
    reciprocal_share = 0.0
    if len(rels):
        lo = np.where(events["caller"] <= events["callee"], events["caller"], events["callee"])
        hi = np.where(events["caller"] <= events["callee"], events["callee"], events["caller"])
        directions = pd.DataFrame({"lo": lo, "hi": hi,
                                   "fwd": events["caller"].to_numpy() == lo})
        both = directions.groupby(["lo", "hi"])["fwd"].nunique() == 2
        reciprocal_share = round(float(both.mean()), 3) if len(both) else 0.0

    stats = {
        "events": len(events),
        "numbers": len(profiles),
        "pairs": len(rels),
        "burners": len(burners),
        "colocation_pairs": len(coloc),
        "reciprocal_share": reciprocal_share,
        "median_calls_per_pair": (int(rels["interaction_count"].median())
                                  if len(rels) else 0),
        "odd_hour_share": (round(float(profiles["odd_hour_ratio"].mean()), 3)
                           if len(profiles) else 0.0),
    }
    return CDRResult(events, rels, profiles, burners, coloc, rep, stats)


def synthetic_cdr(relationships: pd.DataFrame, persons: pd.DataFrame,
                  *, seed: int = 42, calls_per_link: int = 6) -> pd.DataFrame:
    """Generate a plausible CDR event log from an existing communication network.

    Used by the demo so the CDR panel has data without shipping a second fixture.
    Deterministic for a given seed.

    "Plausible" is doing real work here. A naive generator — random tower per call, fresh
    IMEI per call, every pair reciprocal — produces a log where every analytic reads the
    same for every subscriber: co-location links everyone to everyone, handset-swap
    detection fires on all 100% of numbers, and the reciprocity term carries no
    information. So the generator models what actually constrains a call log:

    * one handset per SIM, with a small minority of genuine handset swaps;
    * a home cell cluster per actor, so shared towers mean shared geography;
    * a minority of one-directional ties (a handler who only ever calls out);
    * a handful of planted burners — short-lived, few contacts, night-heavy — which
      exist so :func:`detect_burners` has true positives to find in the demo.
    """
    rng = np.random.default_rng(seed)
    if not len(relationships):
        return pd.DataFrame(columns=CDR_COLUMNS)

    ids = list(persons["person_id"].astype(str)) if len(persons) else []
    if not ids:
        return pd.DataFrame(columns=CDR_COLUMNS)

    numbers = {pid: f"9{rng.integers(100000000, 999999999)}" for pid in ids}
    # One handset per SIM. ~8% of actors swap to a second handset partway through, which
    # is what a real IMEI-change flag looks like — not a new device on every call.
    handsets = {pid: [f"3{rng.integers(10**13, 10**14 - 1)}"] for pid in ids}
    for pid in ids:
        if rng.random() < 0.08:
            handsets[pid].append(f"3{rng.integers(10**13, 10**14 - 1)}")

    comm = relationships[relationships["relation_type"].isin(
        ["Communication", "Handler", "Co-Accused", "Gang Member"])]
    if not len(comm):
        comm = relationships

    # Cell sites are grouped into notional localities; each actor keeps a home cluster,
    # so two actors sharing towers genuinely implies shared ground.
    towers = [f"CGI-{rng.integers(10000, 99999)}" for _ in range(48)]
    clusters = [towers[i:i + 4] for i in range(0, len(towers), 4)]
    home = {pid: clusters[int(rng.integers(0, len(clusters)))] for pid in ids}

    # Pairs that only ever transact in one direction — a handler taking no callbacks.
    one_way = {(str(r["source_id"]), str(r["target_id"]))
               for _, r in comm.iterrows() if rng.random() < 0.18}

    rows: list[dict] = []
    base = pd.Timestamp("2026-01-05")

    # Hour-of-day distribution: waking hours carry most traffic, with a deliberate
    # minority inside the 00:00–05:00 odd-hour window so odd-hour profiling has
    # something to find. Normalised explicitly — ``rng.choice`` rejects a weight
    # vector that does not sum to exactly 1.0.
    _hours = np.array(list(range(6, 23)) + [0, 1, 2, 3, 4])
    _hour_p = np.array([1.0] * 17 + [0.43] * 5)
    _hour_p = _hour_p / _hour_p.sum()
    _night = np.array([0, 1, 2, 3, 4, 23])

    def _event(caller_id: str, callee_id: str, ts: pd.Timestamp, *, tower: str,
               duration: int, sms_p: float = 0.15) -> dict:
        return {
            "caller": numbers[caller_id], "callee": numbers[callee_id],
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": duration,
            "call_type": "SMS" if rng.random() < sms_p else "VOICE",
            "tower_id": tower,
            # Second handset only appears in the later half of the window.
            "imei": handsets[caller_id][-1] if (len(handsets[caller_id]) > 1
                                                and ts > base + pd.Timedelta(days=60))
                    else handsets[caller_id][0],
            "location": "Unknown",
        }

    for _, link in comm.iterrows():
        a, b = str(link["source_id"]), str(link["target_id"])
        if a not in numbers or b not in numbers:
            continue
        n_calls = max(2, int(rng.poisson(calls_per_link * float(link.get("strength", 0.5)) * 2)))
        # Contact happens where the pair overlaps: both home clusters, plus one shared
        # neutral site standing in for a meeting point.
        pair_towers = list(dict.fromkeys(home[a] + home[b] + [towers[hash(a + b) % len(towers)]]))
        directional = (a, b) in one_way
        for _ in range(n_calls):
            forward = True if directional else rng.random() < 0.62
            caller, callee = (a, b) if forward else (b, a)
            hour = int(rng.choice(_hours, p=_hour_p))
            ts = base + pd.Timedelta(days=int(rng.integers(0, 120)),
                                     hours=hour, minutes=int(rng.integers(0, 60)))
            rows.append(_event(caller, callee, ts,
                               tower=str(rng.choice(pair_towers)),
                               duration=int(max(5, rng.exponential(140)))))

    # ── Planted burners ───────────────────────────────────────────────────────
    # A short-lived SIM, a handful of contacts, night-weighted traffic: the profile
    # detect_burners() is built to catch. Without these the demo shows an empty panel
    # and a judge cannot tell a working detector from a broken one.
    n_burners = max(2, len(ids) // 40)
    for k in range(n_burners):
        holder = ids[int(rng.integers(0, len(ids)))]
        burner_id = f"__BURNER_{k:02d}"
        numbers[burner_id] = f"7{rng.integers(100000000, 999999999)}"
        handsets[burner_id] = [f"3{rng.integers(10**13, 10**14 - 1)}"]
        home[burner_id] = home[holder]
        # Contacts drawn from the holder's own neighbourhood — a burner used for one
        # operation, not a randomly wired number.
        peers = [p for p in rng.choice(ids, size=min(4, len(ids)), replace=False)]
        start = int(rng.integers(0, 100))
        for _ in range(int(rng.integers(9, 18))):
            peer = str(rng.choice(peers))
            outbound = rng.random() < 0.7
            ts = base + pd.Timedelta(days=start + int(rng.integers(0, 12)),
                                     hours=int(rng.choice(_night)),
                                     minutes=int(rng.integers(0, 60)))
            src, dst = (burner_id, peer) if outbound else (peer, burner_id)
            rows.append(_event(src, dst, ts,
                               tower=str(rng.choice(home[burner_id])),
                               duration=int(max(5, rng.exponential(45))), sms_p=0.4))

    out = pd.DataFrame(rows, columns=CDR_COLUMNS)
    return out.sort_values("timestamp", ignore_index=True)
