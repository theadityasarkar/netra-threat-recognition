"""
SQLite case store — persistence for analysis runs, alerts and an audit trail.

Why SQLite and not "a real database"
------------------------------------
An investigative tool needs a defensible record of *what was analysed, when, by
whom, and what the system said* — an analyst who acted on an alert must be able to
show the state that produced it. That is an audit requirement, not a scaling one.
SQLite fits it exactly: zero-configuration, single file, ACID, present in the
Python standard library, and trivially backed up by copying one file. Postgres
would add an operational dependency and a service to keep alive for no benefit at
this data volume, and would make the free-tier deployment story worse.

Durability differs by host, and the tool is honest about it
-----------------------------------------------------------
* **Docker / VPS** — the database lives on a bind-mounted volume and is durable.
  This is the deployment where the audit trail is real.
* **Streamlit Community Cloud / Render free tier** — the container filesystem is
  ephemeral. Data survives page reloads within a session but is lost when the
  container recycles. :func:`storage_mode` reports which regime is active and the
  UI states it plainly rather than implying a durability that isn't there.

Every write is wrapped so a read-only or full filesystem degrades the app to
stateless operation with a warning instead of taking the analysis down with it —
persistence is a convenience here, never a prerequisite.

Schema
------
``analysis_runs``  one row per pipeline execution, with headline metrics
``run_alerts``     the alert queue as it stood for a given run
``watchlist``      analyst-flagged actors, persisted across runs
``audit_log``      append-only record of every consequential action
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import DB_ENABLED, DB_PATH, DB_TIMEOUT_SECONDS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id        TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    case_name     TEXT,
    data_source   TEXT,
    n_actors      INTEGER,
    n_links       INTEGER,
    n_modules     INTEGER,
    modularity    REAL,
    gnn_auc       REAL,
    n_critical    INTEGER,
    n_high        INTEGER,
    n_alerts      INTEGER,
    top_target    TEXT,
    parameters    TEXT,
    timings       TEXT
);

CREATE TABLE IF NOT EXISTS run_alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    severity      TEXT,
    category      TEXT,
    subject_id    TEXT,
    title         TEXT,
    detail        TEXT,
    score         REAL,
    action        TEXT,
    FOREIGN KEY (run_id) REFERENCES analysis_runs (run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS watchlist (
    person_id     TEXT PRIMARY KEY,
    name          TEXT,
    reason        TEXT,
    added_at      TEXT NOT NULL,
    added_by      TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    action        TEXT NOT NULL,
    detail        TEXT,
    actor         TEXT
);

CREATE INDEX IF NOT EXISTS idx_alerts_run  ON run_alerts (run_id);
CREATE INDEX IF NOT EXISTS idx_runs_created ON analysis_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_ts    ON audit_log (ts DESC);
"""

# One lock per process. SQLite serialises writers itself, but Streamlit reruns
# scripts on every interaction and can issue concurrent writes from separate
# threads; this keeps the error path quiet rather than surfacing "database is
# locked" to an analyst mid-demo.
_LOCK = threading.Lock()


@dataclass
class StorageStatus:
    """Whether persistence is available, and how durable it actually is."""

    enabled: bool
    durable: bool
    path: str
    mode: str
    message: str


def db_path() -> Path:
    """Resolved database location. ``NETRA_DB_PATH`` overrides the default."""
    return Path(os.environ.get("NETRA_DB_PATH", str(DB_PATH)))


def storage_mode() -> StorageStatus:
    """Report the active persistence regime without asserting false durability.

    Ephemeral hosts are detected by their platform environment variables. When we
    cannot tell, we assume ephemeral — over-claiming durability in an evidentiary
    tool is the worse error.
    """
    path = db_path()
    if not DB_ENABLED:
        return StorageStatus(False, False, str(path), "disabled",
                             "Persistence disabled in configuration — running stateless.")

    ephemeral_markers = ("STREAMLIT_SHARING_MODE", "STREAMLIT_RUNTIME_ENV",
                         "RENDER", "DYNO", "K_SERVICE")
    on_ephemeral = any(os.environ.get(k) for k in ephemeral_markers)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _connect() as conn:
            conn.execute("SELECT 1")
    except Exception as exc:                                   # noqa: BLE001
        return StorageStatus(False, False, str(path), "unavailable",
                             f"Case store unavailable ({type(exc).__name__}) — "
                             f"running stateless. Analysis is unaffected.")

    if on_ephemeral:
        return StorageStatus(True, False, str(path), "ephemeral",
                             "Managed host detected: the case store lives in the "
                             "container filesystem and is cleared when the container "
                             "recycles. Export reports for anything you need to keep.")
    return StorageStatus(True, True, str(path), "durable",
                         f"Case store is durable at {path}.")


@contextmanager
def _connect():
    """Yield a configured connection. WAL keeps readers unblocked during writes."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=DB_TIMEOUT_SECONDS,
                           check_same_thread=False)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> bool:
    """Create the schema if absent. Returns False if the store is unusable."""
    if not DB_ENABLED:
        return False
    try:
        with _LOCK, _connect() as conn:
            conn.executescript(_SCHEMA)
        return True
    except Exception:                                          # noqa: BLE001
        return False


def _safe(fn, default=None):
    """Run a store operation, swallowing storage faults.

    Persistence is a convenience layer: a locked or read-only database must never
    break an analysis that has already been computed in memory.
    """
    try:
        return fn()
    except Exception:                                          # noqa: BLE001
        return default


# ──────────────────────────────────────────────────────────────────────────────
# Writes
# ──────────────────────────────────────────────────────────────────────────────

def _new_run_id() -> str:
    """Mint a unique, human-sortable run identifier.

    A second-resolution stamp is not unique enough. Two analyses saved inside the same
    second collided on the id, and because ``analysis_runs`` was written with
    ``INSERT OR REPLACE`` — a DELETE plus an INSERT — the collision silently destroyed
    the earlier run *and* cascaded away its stored alert queue. That is reachable
    without any unusual timing: the pipeline is cached, so flipping between two
    previously-computed configurations persists two distinct runs back-to-back with no
    recompute in between. Millisecond precision keeps the id readable and chronologically
    sortable while making a collision effectively impossible.
    """
    now = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    return f"RUN-{stamp}-{int((now % 1) * 1000):03d}"


def save_run(result, *, case_name: str = "Untitled case",
             data_source: str = "sample", parameters: dict | None = None,
             alerts: list | None = None) -> str | None:
    """Persist one analysis run and its alert queue.

    Args:
        result: the :class:`~src.pipeline.AnalysisResult` to record.
        case_name: analyst-supplied label for the run.
        parameters: the sidebar configuration, stored as JSON for reproducibility —
            this is what makes a past finding re-derivable.
        alerts: :class:`~src.reporting.alerts.Alert` objects to store alongside.

    Returns:
        The generated ``run_id``, or ``None`` if the store is unavailable.
    """
    if not DB_ENABLED:
        return None
    run_id = _new_run_id()
    stats = getattr(result, "anomaly_stats", {}) or {}
    priority = getattr(result, "priority", None)
    top_target = ""
    if priority is not None and len(priority):
        row = priority.iloc[0]
        top_target = f"{row.get('name', '')} ({row.get('person_id', '')})".strip()

    def _write():
        with _LOCK, _connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                """INSERT INTO analysis_runs
                   (run_id, created_at, case_name, data_source, n_actors, n_links,
                    n_modules, modularity, gnn_auc, n_critical, n_high, n_alerts,
                    top_target, parameters, timings)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, time.strftime("%Y-%m-%d %H:%M:%S"), case_name, data_source,
                 int(result.n_actors), int(result.n_links), int(result.n_modules),
                 float(getattr(result, "modularity", 0.0) or 0.0),
                 float(getattr(getattr(result, "gnn", None), "auc", 0.0) or 0.0),
                 int(stats.get("critical", 0)), int(stats.get("high", 0)),
                 int(len(alerts or [])), top_target,
                 json.dumps(parameters or {}, default=str),
                 json.dumps(getattr(result, "timings", {}) or {}, default=str)))

            for a in (alerts or []):
                conn.execute(
                    """INSERT INTO run_alerts
                       (run_id, severity, category, subject_id, title, detail, score, action)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (run_id, getattr(a, "severity", ""), getattr(a, "category", ""),
                     getattr(a, "subject_id", ""), getattr(a, "title", ""),
                     getattr(a, "detail", ""), float(getattr(a, "score", 0.0) or 0.0),
                     getattr(a, "recommended_action", "")))
        return run_id

    saved = _safe(_write)
    if saved:
        log_action("analysis_saved", f"{case_name} · {result.n_actors} actors, "
                                     f"{result.n_links} links")
    return saved


def log_action(action: str, detail: str = "", actor: str = "analyst") -> None:
    """Append to the audit trail. Never raises — a lost log line must not
    interrupt the work being logged."""
    if not DB_ENABLED:
        return

    def _write():
        with _LOCK, _connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT INTO audit_log (ts, action, detail, actor) VALUES (?,?,?,?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"), action, detail, actor))

    _safe(_write)


def add_to_watchlist(person_id: str, name: str = "", reason: str = "",
                     added_by: str = "analyst") -> bool:
    """Flag an actor for follow-up. Idempotent on ``person_id``."""
    if not DB_ENABLED:
        return False

    def _write():
        with _LOCK, _connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                """INSERT OR REPLACE INTO watchlist
                   (person_id, name, reason, added_at, added_by) VALUES (?,?,?,?,?)""",
                (str(person_id), str(name), str(reason),
                 time.strftime("%Y-%m-%d %H:%M:%S"), added_by))
        return True

    ok = bool(_safe(_write, False))
    if ok:
        log_action("watchlist_add", f"{name} ({person_id}): {reason}")
    return ok


def remove_from_watchlist(person_id: str) -> bool:
    if not DB_ENABLED:
        return False

    def _write():
        with _LOCK, _connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute("DELETE FROM watchlist WHERE person_id = ?", (str(person_id),))
        return True

    ok = bool(_safe(_write, False))
    if ok:
        log_action("watchlist_remove", str(person_id))
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# Reads — all return empty, correctly-typed frames when the store is unavailable
# ──────────────────────────────────────────────────────────────────────────────

def _query(sql: str, params: tuple = (), columns: list[str] | None = None) -> pd.DataFrame:
    def _read():
        with _connect() as conn:
            conn.executescript(_SCHEMA)
            return pd.read_sql_query(sql, conn, params=params)

    out = _safe(_read)
    if out is None:
        return pd.DataFrame(columns=columns or [])
    return out


def recent_runs(limit: int = 20) -> pd.DataFrame:
    return _query(
        """SELECT run_id, created_at, case_name, data_source, n_actors, n_links,
                  n_modules, modularity, gnn_auc, n_critical, n_high, n_alerts, top_target
           FROM analysis_runs ORDER BY created_at DESC LIMIT ?""",
        (int(limit),),
        ["run_id", "created_at", "case_name", "data_source", "n_actors", "n_links",
         "n_modules", "modularity", "gnn_auc", "n_critical", "n_high", "n_alerts",
         "top_target"])


def run_alerts(run_id: str) -> pd.DataFrame:
    return _query(
        """SELECT severity, category, subject_id, title, detail, score, action
           FROM run_alerts WHERE run_id = ? ORDER BY score DESC""",
        (str(run_id),),
        ["severity", "category", "subject_id", "title", "detail", "score", "action"])


def get_watchlist() -> pd.DataFrame:
    return _query(
        "SELECT person_id, name, reason, added_at, added_by FROM watchlist "
        "ORDER BY added_at DESC",
        (), ["person_id", "name", "reason", "added_at", "added_by"])


def audit_trail(limit: int = 100) -> pd.DataFrame:
    return _query(
        "SELECT ts, action, detail, actor FROM audit_log ORDER BY ts DESC, id DESC LIMIT ?",
        (int(limit),), ["ts", "action", "detail", "actor"])


def store_stats() -> dict:
    """Row counts and file size for the storage panel."""
    def _read():
        with _connect() as conn:
            conn.executescript(_SCHEMA)
            out = {}
            for table in ("analysis_runs", "run_alerts", "watchlist", "audit_log"):
                out[table] = int(conn.execute(
                    f"SELECT COUNT(*) FROM {table}").fetchone()[0])   # noqa: S608
            return out

    stats = _safe(_read, {}) or {}
    path = db_path()
    stats["size_kb"] = round(path.stat().st_size / 1024, 1) if path.exists() else 0.0
    return stats


def reset_store() -> bool:
    """Drop every row, keeping the schema. Used by the demo-reset control."""
    def _write():
        with _LOCK, _connect() as conn:
            conn.executescript(_SCHEMA)
            for table in ("run_alerts", "analysis_runs", "watchlist", "audit_log"):
                conn.execute(f"DELETE FROM {table}")               # noqa: S608
        return True

    ok = bool(_safe(_write, False))
    if ok:
        log_action("store_reset", "All persisted runs and alerts cleared.")
    return ok
