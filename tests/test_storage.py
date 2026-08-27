"""Tests for the SQLite case store (``src/storage/database.py``).

The store carries the audit trail, so its contract is stricter than the analysis
layer's: a saved run must survive, the trail must be append-only, and — because
persistence is a convenience layer over an analysis that already exists in memory — a
broken or read-only database must degrade quietly instead of taking the app down.

Every test redirects ``NETRA_DB_PATH`` to a tmp file, so the developer's own case store
is never touched.
"""

from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from src.reporting.alerts import build_alerts
from src.storage import database as db


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fresh, isolated case store for one test."""
    monkeypatch.setenv("NETRA_DB_PATH", str(tmp_path / "netra.db"))
    monkeypatch.setenv("NETRA_DB_ENABLED", "1")
    assert db.init_db()
    return tmp_path / "netra.db"


@pytest.fixture
def alerts(result):
    return build_alerts(result)


# ──────────────────────────────────────────────────────────────────────────────
# Setup & status
# ──────────────────────────────────────────────────────────────────────────────

def test_init_creates_the_database_file(store):
    assert store.exists()
    assert db.db_path() == store


def test_storage_mode_reports_an_honest_regime(store):
    status = db.storage_mode()

    assert status.enabled
    assert status.mode in {"durable", "ephemeral", "disabled"}
    assert status.message, "the analyst is told nothing about durability"
    # Claiming durability we cannot guarantee is the error that matters here.
    assert isinstance(status.durable, bool)


def test_storage_mode_reports_ephemeral_on_a_known_ephemeral_host(store, monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    status = db.storage_mode()
    assert not status.durable
    assert status.mode == "ephemeral"


# ──────────────────────────────────────────────────────────────────────────────
# Saving runs
# ──────────────────────────────────────────────────────────────────────────────

def test_save_run_round_trips_the_headline_figures(store, result, alerts):
    run_id = db.save_run(result, case_name="Operation Test", data_source="sample",
                         parameters={"seed": 7, "gnn_epochs": 150}, alerts=alerts)

    assert run_id and run_id.startswith("RUN-")

    runs = db.recent_runs()
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["run_id"] == run_id
    assert row["case_name"] == "Operation Test"
    assert int(row["n_actors"]) == result.n_actors
    assert int(row["n_links"]) == result.n_links
    assert int(row["n_alerts"]) == len(alerts)

    stored = db.run_alerts(run_id)
    assert len(stored) == len(alerts)
    assert set(stored["severity"]) <= {"Critical", "High", "Medium", "Low"}


def test_two_runs_saved_in_the_same_second_both_survive(store, result, alerts):
    """Regression: the run id was second-resolution and rows were written with
    ``INSERT OR REPLACE``, so a same-second save silently deleted the earlier run and
    cascaded away its alert queue. Cached results persist back-to-back with no
    recompute between them, so this was reachable from the UI."""
    first = db.save_run(result, case_name="Case A", alerts=alerts)
    second = db.save_run(result, case_name="Case B", alerts=alerts)

    assert first and second
    assert first != second, "run ids collided"

    runs = db.recent_runs()
    assert len(runs) == 2, "a saved run was destroyed by the next save"
    assert sorted(runs["case_name"]) == ["Case A", "Case B"]
    # Each run keeps its own queue rather than inheriting or losing the other's.
    assert len(db.run_alerts(first)) == len(alerts)
    assert len(db.run_alerts(second)) == len(alerts)


def test_run_ids_sort_chronologically(store, result):
    """The id doubles as the sort key in the run history, so lexical order must match
    chronological order."""
    ids = [db.save_run(result, case_name=f"Case {i}") for i in range(3)]
    assert ids == sorted(ids)


def test_parameters_are_stored_for_reproducibility(store, result):
    """The sidebar configuration is what makes a past finding re-derivable, so it is
    persisted as JSON. ``recent_runs()`` doesn't project it, so read the column
    directly — this is the storage layer's own test."""
    params = {"seed": 42, "gnn_epochs": 200, "contamination": 0.08}
    run_id = db.save_run(result, case_name="Reproducible", parameters=params)

    with sqlite3.connect(str(store)) as conn:
        raw = conn.execute("SELECT parameters, timings FROM analysis_runs WHERE run_id = ?",
                           (run_id,)).fetchone()

    stored_params = json.loads(raw[0])
    assert stored_params["seed"] == 42
    assert stored_params["gnn_epochs"] == 200
    assert json.loads(raw[1]), "per-stage timings were not recorded"


def test_save_run_without_alerts_is_valid(store, result):
    run_id = db.save_run(result, case_name="No alerts")
    assert run_id
    assert db.run_alerts(run_id).empty


# ──────────────────────────────────────────────────────────────────────────────
# Audit trail
# ──────────────────────────────────────────────────────────────────────────────

def test_saving_a_run_writes_the_audit_trail(store, result):
    db.save_run(result, case_name="Audited")
    trail = db.audit_trail()

    assert len(trail) >= 1
    assert "analysis_saved" in set(trail["action"])


def test_audit_trail_is_append_only_and_newest_first(store):
    for i in range(5):
        db.log_action("test_action", f"detail {i}")

    trail = db.audit_trail()
    assert len(trail) == 5
    # Newest first — an analyst opening the panel wants the last thing that happened.
    assert list(trail["detail"])[0] == "detail 4"


def test_log_action_never_raises_even_when_the_store_is_broken(tmp_path, monkeypatch):
    """A lost log line must not interrupt the work being logged."""
    monkeypatch.setenv("NETRA_DB_PATH", str(tmp_path / "nonexistent" / "x" / "netra.db"))
    db.log_action("action", "detail")          # must not raise

    # A path that cannot be a database at all.
    bogus = tmp_path / "not-a-db"
    bogus.write_bytes(b"this is not sqlite")
    monkeypatch.setenv("NETRA_DB_PATH", str(bogus))
    db.log_action("action", "detail")          # must not raise
    assert db.audit_trail().empty or isinstance(db.audit_trail(), pd.DataFrame)


def test_reads_degrade_to_empty_frames_on_a_corrupt_store(tmp_path, monkeypatch):
    bogus = tmp_path / "corrupt.db"
    bogus.write_bytes(b"definitely not a sqlite header")
    monkeypatch.setenv("NETRA_DB_PATH", str(bogus))

    for frame in (db.recent_runs(), db.get_watchlist(), db.audit_trail(),
                  db.run_alerts("RUN-nope")):
        assert isinstance(frame, pd.DataFrame)
        assert frame.empty


# ──────────────────────────────────────────────────────────────────────────────
# Watchlist
# ──────────────────────────────────────────────────────────────────────────────

def test_watchlist_add_read_remove(store):
    db.add_to_watchlist("P0007", name="Aran Mehta", reason="Cross-module broker")

    wl = db.get_watchlist()
    assert len(wl) == 1
    assert wl.iloc[0]["person_id"] == "P0007"
    assert wl.iloc[0]["reason"] == "Cross-module broker"

    assert db.remove_from_watchlist("P0007")
    assert db.get_watchlist().empty


def test_watchlist_does_not_duplicate_the_same_actor(store):
    db.add_to_watchlist("P0007", name="Aran Mehta", reason="First reason")
    db.add_to_watchlist("P0007", name="Aran Mehta", reason="Updated reason")

    wl = db.get_watchlist()
    assert len(wl) == 1, "the same actor was watchlisted twice"


def test_removing_an_absent_actor_is_not_an_error(store):
    assert db.remove_from_watchlist("P9999") in (True, False)
    assert db.get_watchlist().empty


# ──────────────────────────────────────────────────────────────────────────────
# Stats & reset
# ──────────────────────────────────────────────────────────────────────────────

def test_store_stats_track_what_was_written(store, result, alerts):
    assert db.store_stats().get("analysis_runs", 0) == 0

    db.save_run(result, case_name="Counted", alerts=alerts)
    db.add_to_watchlist("P0007", name="Aran")

    stats = db.store_stats()
    assert stats["analysis_runs"] == 1
    assert stats["run_alerts"] == len(alerts)
    assert stats["watchlist"] == 1
    assert stats["audit_log"] >= 1
    assert stats["size_kb"] > 0


def test_reset_clears_the_store(store, result):
    db.save_run(result, case_name="Doomed")
    assert len(db.recent_runs()) == 1

    assert db.reset_store()
    assert db.recent_runs().empty


def test_everything_is_a_no_op_when_persistence_is_disabled(tmp_path, monkeypatch, result):
    """``NETRA_DB_ENABLED=0`` is the read-only-filesystem escape hatch. It must not
    raise anywhere — the analysis is already computed and must still be usable."""
    monkeypatch.setenv("NETRA_DB_PATH", str(tmp_path / "netra.db"))
    monkeypatch.setattr(db, "DB_ENABLED", False)

    assert db.init_db() is False
    assert db.save_run(result, case_name="Nowhere") is None
    db.log_action("action", "detail")
    assert db.storage_mode().enabled is False
    assert not (tmp_path / "netra.db").exists(), "a disabled store still created a file"
