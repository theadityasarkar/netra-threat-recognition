"""
Layer 1 — CSV ingestion, header normalisation and validation.

Real agency exports never match a spec exactly, so this module is deliberately
forgiving: it lower-cases and de-punctuates headers, resolves them through an alias
table (``accused_id`` → ``person_id``), coerces types, fills sane defaults, and
returns a structured report of everything it had to fix. The report is surfaced in
the dashboard's Data Quality panel — nothing is silently repaired.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    COLUMN_ALIASES,
    CRIME_COLUMNS,
    CRIME_REQUIRED,
    CRIME_SEVERITY,
    DOCUMENT_COLUMNS,
    DOCUMENT_REQUIRED,
    PERSON_COLUMNS,
    PERSON_REQUIRED,
    RELATIONSHIP_COLUMNS,
    RELATIONSHIP_REQUIRED,
    RISK_CATEGORIES,
)

_ID_SPLIT = re.compile(r"[;,|/]+")


# ──────────────────────────────────────────────────────────────────────────────
# Validation report
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    """Everything the loader noticed while reading a dataset."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    stats: dict[str, int | float | str] = field(default_factory=dict)
    renamed: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def merge(self, other: "ValidationReport", prefix: str = "") -> None:
        p = f"[{prefix}] " if prefix else ""
        self.errors += [p + e for e in other.errors]
        self.warnings += [p + w for w in other.warnings]
        self.info += [p + i for i in other.info]
        self.renamed.update(other.renamed)
        for k, v in other.stats.items():
            self.stats[f"{prefix}_{k}" if prefix else k] = v


# ──────────────────────────────────────────────────────────────────────────────
# Header handling
# ──────────────────────────────────────────────────────────────────────────────

def _canon(col: str) -> str:
    """Reduce a header to a comparison key: lowercase, alphanumerics + underscore."""
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")


def _build_reverse_alias() -> dict[str, str]:
    rev: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        rev[_canon(canonical)] = canonical
        for a in aliases:
            rev.setdefault(_canon(a), canonical)
    return rev


_REVERSE_ALIAS = _build_reverse_alias()


def normalise_headers(df: pd.DataFrame, expected: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Map incoming headers onto the canonical schema where possible."""
    expected_canon = {_canon(c): c for c in expected}
    mapping: dict[str, str] = {}

    for col in df.columns:
        key = _canon(col)
        if key in expected_canon:
            target = expected_canon[key]
        elif key in _REVERSE_ALIAS and _REVERSE_ALIAS[key] in expected:
            target = _REVERSE_ALIAS[key]
        else:
            target = key                      # keep unknown columns, but tidy the name
        if target != col:
            mapping[str(col)] = target

    out = df.rename(columns=mapping)
    # Two source columns collapsing onto one canonical name: keep the first.
    out = out.loc[:, ~out.columns.duplicated(keep="first")]
    return out, mapping


# ──────────────────────────────────────────────────────────────────────────────
# Raw reading
# ──────────────────────────────────────────────────────────────────────────────

def read_csv_any(source) -> pd.DataFrame:
    """Read a CSV from a path, bytes, or a Streamlit ``UploadedFile``."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)

    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=enc, skipinitialspace=True,
                               dtype=str, keep_default_na=False, na_values=[""])
        except Exception as exc:                       # noqa: BLE001 — try next encoding
            last_err = exc
    raise ValueError(f"Could not parse CSV: {last_err}")


# ──────────────────────────────────────────────────────────────────────────────
# Per-table loaders
# ──────────────────────────────────────────────────────────────────────────────

def _num(series: pd.Series, default: float, lo: float | None = None, hi: float | None = None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").fillna(default)
    if lo is not None or hi is not None:
        out = out.clip(lower=lo, upper=hi)
    return out


def load_persons(source) -> tuple[pd.DataFrame, ValidationReport]:
    rep = ValidationReport()
    try:
        raw = read_csv_any(source)
    except Exception as exc:                            # noqa: BLE001
        rep.errors.append(f"Unreadable persons file: {exc}")
        return pd.DataFrame(columns=PERSON_COLUMNS), rep

    df, renamed = normalise_headers(raw, PERSON_COLUMNS)
    rep.renamed = renamed
    if renamed:
        rep.info.append(f"Auto-mapped {len(renamed)} column(s): "
                        + ", ".join(f"{k} → {v}" for k, v in list(renamed.items())[:6])
                        + ("…" if len(renamed) > 6 else ""))

    missing = [c for c in PERSON_REQUIRED if c not in df.columns]
    if missing:
        rep.errors.append(f"persons.csv is missing required column(s): {', '.join(missing)}")
        return df, rep

    for col in PERSON_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["person_id"] = df["person_id"].astype(str).str.strip()
    blank = df["person_id"].isin(["", "nan", "None"])
    if blank.any():
        rep.warnings.append(f"Dropped {int(blank.sum())} row(s) with a blank person_id.")
        df = df[~blank]

    dupes = df["person_id"].duplicated(keep="first")
    if dupes.any():
        rep.warnings.append(f"Removed {int(dupes.sum())} duplicate person_id row(s).")
        df = df[~dupes]

    df["name"] = df["name"].fillna("").astype(str).str.strip()
    unnamed = df["name"] == ""
    if unnamed.any():
        df.loc[unnamed, "name"] = "Unnamed " + df.loc[unnamed, "person_id"].astype(str)
        rep.warnings.append(f"{int(unnamed.sum())} record(s) had no name — substituted the ID.")

    df["alias"] = df["alias"].fillna("").astype(str)
    df["gender"] = df["gender"].fillna("U").astype(str).str.strip().str[:1].str.upper()
    df["age"] = _num(df["age"], 0, 0, 110).astype(int)
    df["city"] = df["city"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    df["state"] = df["state"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    df["occupation"] = df["occupation"].fillna("Unknown").astype(str).replace("", "Unknown")
    df["prior_offences"] = _num(df["prior_offences"], 0, 0, 500).astype(int)
    df["offence_types"] = df["offence_types"].fillna("").astype(str)
    df["first_arrest_year"] = _num(df["first_arrest_year"], 0, 0, 2100).astype(int)
    df["last_activity"] = pd.to_datetime(df["last_activity"], errors="coerce", format="mixed")
    df["status"] = df["status"].fillna("Active").astype(str).replace("", "Active")

    df["risk_category"] = df["risk_category"].fillna("").astype(str).str.strip().str.title()
    unknown_risk = ~df["risk_category"].isin(RISK_CATEGORIES)
    if unknown_risk.any():
        # Derive a band from prior offences rather than defaulting everyone to Low.
        derived = pd.cut(df.loc[unknown_risk, "prior_offences"],
                         bins=[-1, 0, 2, 5, 1000],
                         labels=["Low", "Medium", "High", "Critical"])
        df.loc[unknown_risk, "risk_category"] = derived.astype(str)
        rep.info.append(f"Derived risk_category for {int(unknown_risk.sum())} record(s) "
                        "from prior offence counts.")

    rep.stats["rows"] = len(df)
    return df.reset_index(drop=True), rep


def load_relationships(source, valid_ids: set[str] | None = None) -> tuple[pd.DataFrame, ValidationReport]:
    rep = ValidationReport()
    try:
        raw = read_csv_any(source)
    except Exception as exc:                            # noqa: BLE001
        rep.errors.append(f"Unreadable relationships file: {exc}")
        return pd.DataFrame(columns=RELATIONSHIP_COLUMNS), rep

    df, renamed = normalise_headers(raw, RELATIONSHIP_COLUMNS)
    rep.renamed = renamed
    if renamed:
        rep.info.append(f"Auto-mapped {len(renamed)} column(s) in relationships.")

    missing = [c for c in RELATIONSHIP_REQUIRED if c not in df.columns]
    if missing:
        rep.errors.append(f"relationships.csv is missing required column(s): {', '.join(missing)}")
        return df, rep

    for col in RELATIONSHIP_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["source_id"] = df["source_id"].astype(str).str.strip()
    df["target_id"] = df["target_id"].astype(str).str.strip()

    bad = df["source_id"].isin(["", "nan"]) | df["target_id"].isin(["", "nan"])
    if bad.any():
        rep.warnings.append(f"Dropped {int(bad.sum())} link(s) with a missing endpoint.")
        df = df[~bad]

    loops = df["source_id"] == df["target_id"]
    if loops.any():
        rep.warnings.append(f"Dropped {int(loops.sum())} self-link(s).")
        df = df[~loops]

    df["relation_type"] = (df["relation_type"].fillna("Associate").astype(str)
                           .str.strip().str.title().replace("", "Associate"))
    df["strength"] = _num(df["strength"], 0.5, 0.0, 1.0).round(3)
    df["interaction_count"] = _num(df["interaction_count"], 1, 0, 1_000_000).astype(int)
    df["first_seen"] = pd.to_datetime(df["first_seen"], errors="coerce", format="mixed")
    df["last_seen"] = pd.to_datetime(df["last_seen"], errors="coerce", format="mixed")
    df["evidence_source"] = (df["evidence_source"].fillna("Unspecified")
                             .astype(str).replace("", "Unspecified"))
    df["location"] = df["location"].fillna("Unknown").astype(str).replace("", "Unknown")

    if valid_ids is not None:
        orphan = ~(df["source_id"].isin(valid_ids) & df["target_id"].isin(valid_ids))
        if orphan.any():
            rep.warnings.append(
                f"{int(orphan.sum())} link(s) reference IDs absent from persons.csv — "
                "stub records were created so no evidence is lost.")

    # Collapse duplicate undirected pairs, keeping the strongest tie and summing volume.
    pair = pd.DataFrame({
        "a": np.where(df["source_id"] <= df["target_id"], df["source_id"], df["target_id"]),
        "b": np.where(df["source_id"] <= df["target_id"], df["target_id"], df["source_id"]),
    })
    df = df.assign(_a=pair["a"], _b=pair["b"])
    n_before = len(df)
    df = (df.sort_values("strength", ascending=False)
            .groupby(["_a", "_b"], as_index=False)
            .agg({**{c: "first" for c in RELATIONSHIP_COLUMNS},
                  "interaction_count": "sum"}))
    df["source_id"], df["target_id"] = df["_a"], df["_b"]
    df = df.drop(columns=["_a", "_b"])
    if n_before != len(df):
        rep.info.append(f"Merged {n_before - len(df)} duplicate link(s) between the same pair.")

    rep.stats["rows"] = len(df)
    return df.reset_index(drop=True), rep


def load_crime_records(source) -> tuple[pd.DataFrame, ValidationReport]:
    rep = ValidationReport()
    try:
        raw = read_csv_any(source)
    except Exception as exc:                            # noqa: BLE001
        rep.errors.append(f"Unreadable crime records file: {exc}")
        return pd.DataFrame(columns=CRIME_COLUMNS), rep

    df, renamed = normalise_headers(raw, CRIME_COLUMNS)
    rep.renamed = renamed

    missing = [c for c in CRIME_REQUIRED if c not in df.columns]
    if missing:
        rep.warnings.append(
            f"crime_records.csv is missing {', '.join(missing)} — co-accused edge "
            "inference will be skipped.")
        return pd.DataFrame(columns=CRIME_COLUMNS), rep

    for col in CRIME_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["case_id"] = df["case_id"].astype(str).str.strip()
    df["fir_number"] = df["fir_number"].fillna("").astype(str)
    df["crime_type"] = (df["crime_type"].fillna("Unclassified").astype(str)
                        .str.strip().replace("", "Unclassified"))
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df["city"] = df["city"].fillna("Unknown").astype(str).replace("", "Unknown")
    df["state"] = df["state"].fillna("Unknown").astype(str).replace("", "Unknown")
    df["status"] = df["status"].fillna("Unknown").astype(str).replace("", "Unknown")
    df["persons_involved"] = df["persons_involved"].fillna("").astype(str)

    sev = pd.to_numeric(df["severity"], errors="coerce")
    if sev.isna().any():
        fallback = df.loc[sev.isna(), "crime_type"].map(CRIME_SEVERITY).fillna(3)
        sev = sev.fillna(fallback)
        rep.info.append("Inferred severity for some cases from the offence-type table.")
    df["severity"] = sev.clip(1, 5).astype(int)

    empty = df["persons_involved"].str.strip() == ""
    if empty.any():
        rep.warnings.append(f"{int(empty.sum())} case(s) list no accused — excluded from "
                            "co-accused inference.")

    rep.stats["rows"] = len(df)
    return df.reset_index(drop=True), rep


def load_documents(source) -> tuple[pd.DataFrame, ValidationReport]:
    """Load an unstructured-intelligence corpus for the NLP layer.

    Accepts FIR narratives, surveillance notes, social-media intercepts and agency
    reports. Only ``text`` is required; ``doc_id`` is synthesised when absent and the
    provenance columns are filled with blanks, because a corpus with no metadata is
    still extractable — it just loses the trace back to a date and a case.
    """
    rep = ValidationReport()
    try:
        raw = read_csv_any(source)
    except Exception as exc:                            # noqa: BLE001
        rep.errors.append(f"Unreadable documents file: {exc}")
        return pd.DataFrame(columns=DOCUMENT_COLUMNS), rep

    df, renamed = normalise_headers(raw, DOCUMENT_COLUMNS)
    rep.renamed = renamed

    if "text" not in df.columns:
        # A single unnamed content column is the common shape of a pasted export;
        # adopt it rather than rejecting the file over a header we can infer.
        non_meta = [c for c in df.columns if c not in DOCUMENT_COLUMNS]
        if len(df.columns) == 1:
            df = df.rename(columns={df.columns[0]: "text"})
            rep.info.append(f"Adopted the only column `{raw.columns[0]}` as `text`.")
        elif non_meta:
            df = df.rename(columns={non_meta[0]: "text"})
            rep.info.append(f"Adopted `{non_meta[0]}` as the `text` column.")
        else:
            rep.errors.append("documents.csv has no `text` column and none could be inferred.")
            return pd.DataFrame(columns=DOCUMENT_COLUMNS), rep

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    blank = df["text"] == ""
    if blank.any():
        rep.warnings.append(f"{int(blank.sum())} document(s) had empty text and were dropped.")
        df = df[~blank]

    if "doc_id" not in df.columns or df["doc_id"].isna().all():
        df["doc_id"] = [f"D{i:04d}" for i in range(1, len(df) + 1)]
        rep.info.append("Generated sequential doc_id values.")

    for col in DOCUMENT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df["doc_id"] = df["doc_id"].astype(str).str.strip()

    rep.stats["rows"] = len(df)
    if len(df):
        rep.info.append(f"Loaded {len(df)} intelligence document(s) for entity extraction.")
    return df[DOCUMENT_COLUMNS].reset_index(drop=True), rep


# ──────────────────────────────────────────────────────────────────────────────
# Cross-table assembly
# ──────────────────────────────────────────────────────────────────────────────

def parse_id_list(value: str) -> list[str]:
    """Split a ``persons_involved`` cell into clean IDs."""
    if not isinstance(value, str) or not value.strip():
        return []
    return [tok.strip() for tok in _ID_SPLIT.split(value) if tok.strip()]


def prepare_dataset(
    persons: pd.DataFrame,
    relationships: pd.DataFrame,
    crimes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ValidationReport]:
    """Reconcile the three tables into a mutually consistent dataset.

    Creates stub person records for any ID that appears only in the relationship or
    crime tables — losing an edge because a roster is incomplete would lose evidence.
    """
    rep = ValidationReport()
    crimes = crimes if crimes is not None else pd.DataFrame(columns=CRIME_COLUMNS)

    known = set(persons["person_id"].astype(str))
    referenced: set[str] = set()
    if len(relationships):
        referenced |= set(relationships["source_id"].astype(str))
        referenced |= set(relationships["target_id"].astype(str))
    if len(crimes) and "persons_involved" in crimes.columns:
        for cell in crimes["persons_involved"]:
            referenced.update(parse_id_list(cell))

    stubs = sorted(referenced - known)
    if stubs:
        stub_df = pd.DataFrame({
            "person_id": stubs,
            "name": [f"Unidentified {s}" for s in stubs],
            "alias": "", "age": 0, "gender": "U", "city": "Unknown", "state": "Unknown",
            "occupation": "Unknown", "prior_offences": 0, "offence_types": "",
            "first_arrest_year": 0, "last_activity": pd.NaT, "status": "Active",
            "risk_category": "Low",
        })
        persons = pd.concat([persons, stub_df], ignore_index=True)
        rep.warnings.append(f"Created {len(stubs)} stub person record(s) for IDs seen only "
                            "in link/case data.")

    valid = set(persons["person_id"].astype(str))
    if len(crimes) and "persons_involved" in crimes.columns:
        crimes = crimes.copy()
        crimes["_ids"] = crimes["persons_involved"].apply(
            lambda v: [i for i in parse_id_list(v) if i in valid])

    rep.stats.update({
        "persons": len(persons),
        "relationships": len(relationships),
        "crime_records": len(crimes),
        "stubs_created": len(stubs),
    })
    rep.info.append(f"Dataset ready: {len(persons)} actors, {len(relationships)} recorded "
                    f"links, {len(crimes)} case files.")
    return persons.reset_index(drop=True), relationships.reset_index(drop=True), crimes, rep


def load_dataset(
    persons_src,
    relationships_src,
    crimes_src=None,
    documents_src=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, ValidationReport]:
    """End-to-end ingestion for the uploaded files.

    Returns ``(persons, relationships, crimes, documents, report)``. ``documents`` is
    an empty frame when no corpus was supplied, which is the common CSV-only case.
    """
    master = ValidationReport()
    empty_docs = pd.DataFrame(columns=DOCUMENT_COLUMNS)

    persons, p_rep = load_persons(persons_src)
    master.merge(p_rep, "persons")
    if not p_rep.ok:
        return persons, pd.DataFrame(columns=RELATIONSHIP_COLUMNS), \
               pd.DataFrame(columns=CRIME_COLUMNS), empty_docs, master

    rel, r_rep = load_relationships(relationships_src, set(persons["person_id"].astype(str)))
    master.merge(r_rep, "links")
    if not r_rep.ok:
        return persons, rel, pd.DataFrame(columns=CRIME_COLUMNS), empty_docs, master

    if crimes_src is not None:
        crimes, c_rep = load_crime_records(crimes_src)
        master.merge(c_rep, "cases")
    else:
        crimes = pd.DataFrame(columns=CRIME_COLUMNS)

    if documents_src is not None:
        documents, d_rep = load_documents(documents_src)
        master.merge(d_rep, "documents")
    else:
        documents = empty_docs

    persons, rel, crimes, a_rep = prepare_dataset(persons, rel, crimes)
    master.merge(a_rep)
    return persons, rel, crimes, documents, master
