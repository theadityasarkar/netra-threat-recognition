"""
Session-state management — one declared home for every mutable UI value.

Streamlit re-executes the whole script on every interaction, so anything that has
to survive a click must live in ``st.session_state``. Left to grow organically that
dictionary becomes an undocumented global namespace: keys written on one page and
read on another, with no record of what is allowed to exist. This module makes the
contract explicit.

* :data:`DEFAULTS` declares every key the app uses, with its initial value.
* :func:`init_state` seeds them once per session and is safe to call on every rerun.
* Accessors below are the only way pages touch state, so a key's readers and
  writers are greppable.

The watchlist is the one piece of state deliberately kept *outside* the session:
it belongs in the SQLite case store so an analyst's flags survive a browser
refresh. The helpers here mirror it into session state as a fast lookup set and
keep the two in step.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.config import PERF
from src.storage import database as db

# Every session-state key the app owns, with its initial value. Adding a key here
# is the whole registration step — no page should invent one inline.
DEFAULTS: dict[str, Any] = {
    # ── Run lifecycle ─────────────────────────────────────────────────────────
    "has_run": False,          # has an analysis completed at least once?
    "run_count": 0,            # incremented per completed pipeline execution
    "last_error": "",          # last pipeline failure, shown until superseded
    "case_name": "Operation Netra · Demo Case",
    "saved_run_id": "",        # run_id returned by the case store, if persisted
    # ── Navigation ────────────────────────────────────────────────────────────
    "active_actor": "",        # person_id currently pinned in the actor inspector
    # ── Analyst annotations (mirrored from the case store) ─────────────────────
    "watchlist": set(),        # set[str] of person_ids, cached for O(1) lookup
    "watchlist_loaded": False,
    # ── Presentation toggles ──────────────────────────────────────────────────
    "redact": False,           # mask identifiers for screen-sharing / demos
    "show_ps_panel": True,     # PS 26189 traceability matrix on the dashboard
}


def init_state() -> None:
    """Seed missing keys and initialise the case store. Idempotent.

    Called once at the top of ``app.py``. Absent keys are filled in; existing ones
    are never overwritten, so a rerun mid-session preserves everything the analyst
    has done. Mutable defaults are copied, otherwise every session in the process
    would share one set object.
    """
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = set(value) if isinstance(value, set) else value

    if "_db_ready" not in st.session_state:
        # Schema creation is cheap and idempotent; failure is non-fatal because the
        # store is a convenience and the analysis is computed in memory regardless.
        st.session_state["_db_ready"] = db.init_db()


def get(key: str, default: Any = None) -> Any:
    """Read a state value, falling back to the declared default."""
    return st.session_state.get(key, DEFAULTS.get(key, default))


def put(**values: Any) -> None:
    """Write one or more state values."""
    for key, value in values.items():
        st.session_state[key] = value


def mark_run_complete(*, error: str = "") -> None:
    """Record the outcome of a pipeline execution."""
    if error:
        st.session_state["last_error"] = error
        return
    st.session_state["has_run"] = True
    st.session_state["run_count"] = int(get("run_count", 0)) + 1
    st.session_state["last_error"] = ""


# ──────────────────────────────────────────────────────────────────────────────
# Watchlist — session cache over the durable case store
# ──────────────────────────────────────────────────────────────────────────────

def load_watchlist(force: bool = False) -> set[str]:
    """Return the watchlist id set, reading the case store at most once per session.

    Hitting SQLite on every rerun would put a query behind every mouse click for
    data that changes only when the analyst flags someone, so the set is cached and
    invalidated by the mutating helpers below.
    """
    if force or not get("watchlist_loaded"):
        frame = db.get_watchlist()
        ids = set(frame["person_id"].astype(str)) if len(frame) else set()
        put(watchlist=ids, watchlist_loaded=True)
    return set(get("watchlist", set()))


def is_watched(person_id: str) -> bool:
    return str(person_id) in load_watchlist()


def watch(person_id: str, name: str = "", reason: str = "") -> bool:
    """Flag an actor for follow-up, persisting through the case store."""
    ok = db.add_to_watchlist(str(person_id), name, reason)
    # Update the session cache even when persistence failed: the analyst's action
    # should still be visible for the rest of the session on an ephemeral host.
    st.session_state["watchlist"] = load_watchlist() | {str(person_id)}
    return ok


def unwatch(person_id: str) -> bool:
    ok = db.remove_from_watchlist(str(person_id))
    st.session_state["watchlist"] = load_watchlist() - {str(person_id)}
    return ok


def toggle_watch(person_id: str, name: str = "", reason: str = "") -> bool:
    """Flip an actor's watchlist membership. Returns the new state."""
    if is_watched(person_id):
        unwatch(person_id)
        return False
    watch(person_id, name, reason)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Cache decorator presets
# ──────────────────────────────────────────────────────────────────────────────

def cached_data(func=None, /, **kwargs):
    """``st.cache_data`` with the project's memory bounds already applied.

    An unbounded cache is a memory leak with a friendly name: a judge sweeping the
    parameter sliders would pin one full :class:`~src.pipeline.AnalysisResult` per
    combination, and the 1 GB Streamlit Community Cloud container would be OOM-killed
    mid-demo. ``max_entries`` and ``ttl`` come from :data:`~src.config.PERF` so the
    ceiling is tuned in one place.

    Usable both bare and called — ``@cached_data`` and ``@cached_data(ttl=60)`` behave
    identically to the equivalent ``st.cache_data`` forms. Supporting both matters
    because the bare form is what everyone reaches for by habit, and the factory-only
    version fails at import time with a message that does not name the real problem.
    """
    kwargs.setdefault("max_entries", PERF["cache_max_entries"])
    kwargs.setdefault("ttl", PERF["cache_ttl_seconds"])
    kwargs.setdefault("show_spinner", False)
    decorator = st.cache_data(**kwargs)
    return decorator(func) if func is not None else decorator
