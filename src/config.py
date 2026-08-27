"""
NETRA — central configuration.

Single source of truth for schemas, algorithm weights, thresholds and the visual
palette. Every other module imports from here so that a judge (or a reviewer) can
audit and tune the system's behaviour from one file.
"""

from __future__ import annotations

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Application identity
# ──────────────────────────────────────────────────────────────────────────────

APP_NAME = "NETRA"
APP_FULL_NAME = "Network Evaluation & Threat Recognition Analytics"
APP_TAGLINE = "AI-Powered Criminal Network Analysis System"
APP_ORG = "Ministry of Home Affairs"
APP_DEPARTMENT = "National Crime Records Bureau (NCRB) · Women Safety Division"
APP_EVENT = "Smart India Hackathon 2026"
APP_VERSION = "2.0.0"
APP_ICON = "🛡️"

# ── Official SIH 2026 problem-statement metadata ──────────────────────────────
# Source: SIH 2026 portal listing. Surfaced in the UI masthead, the "About" panel
# and the PDF dossier cover so every artefact is traceable to the brief.
PROBLEM_STATEMENT = {
    "id": "26189",
    "title": "AI-Powered Criminal Network Analysis System",
    "organization": "Ministry of Home Affairs",
    "department": "National Crime Records Bureau (NCRB), Women Safety Division",
    "category": "Software",
    "theme": "Blockchain & Cybersecurity",
    "deadline": "20 September 2026",
}

# The data sources named in the problem statement's Background section. Each entry
# maps to a concrete ingestion path in NETRA — see docs/DATA_SCHEMA.md.
MANDATED_SOURCES = [
    ("FIRs and police reports", "Structured CSV + unstructured narrative NLP"),
    ("Call Detail Records (CDRs)", "Dedicated CDR ingestion → communication edges"),
    ("Financial transaction records", "Financial relationship edges + mule detection"),
    ("Surveillance reports", "Free-text entity extraction"),
    ("Social media intelligence", "Free-text entity extraction"),
    ("Criminal history databases", "Person roster with prior-offence history"),
    ("Intelligence agency reports", "Free-text entity extraction"),
]

CLASSIFICATION_BANNER = "RESTRICTED · FOR AUTHORISED LAW-ENFORCEMENT USE ONLY · DEMO DATA"

# The six capabilities the published PS asks for, each mapped to where NETRA
# delivers it. Rendered as a live traceability matrix in the Dashboard tab so a
# judge can follow requirement → implementation → screen without reading code.
PS_REQUIREMENTS: list[dict[str, str]] = [
    {
        "requirement": "Collect and process data from multiple sources",
        "delivered_by": "Multi-source ingestion: FIR/CSV rosters, CDR call logs, "
                        "financial trails and free-text intelligence reports",
        "module": "src/ingestion/loader.py · src/ingestion/entities.py",
        "screen": "Data & Sources",
    },
    {
        "requirement": "Extract entities — people, locations, vehicles, phone numbers, organisations",
        "delivered_by": "NLP entity extraction over unstructured narratives, 12 entity "
                        "types with India-specific pattern recognition and gazetteers",
        "module": "src/ingestion/entities.py",
        "screen": "Entity Extraction",
    },
    {
        "requirement": "Build relationship maps showing how entities are connected",
        "delivered_by": "Multi-relational weighted graph with co-accused inference and "
                        "NLP co-occurrence linking",
        "module": "src/graph/builder.py",
        "screen": "Network Graph",
    },
    {
        "requirement": "Identify key individuals who play influential roles",
        "delivered_by": "10-measure centrality suite, Influence Index, operational-role "
                        "inference and fragmentation-based arrest ranking",
        "module": "src/analysis/key_players.py",
        "screen": "Key Players",
    },
    {
        "requirement": "Detect suspicious patterns and unusual activities",
        "delivered_by": "Hybrid ensemble — GraphSAGE GNN residuals + Isolation Forest + "
                        "8-rule expert engine, every score explained in plain language",
        "module": "src/analysis/anomaly.py",
        "screen": "Anomalies",
    },
    {
        "requirement": "Assist investigators with visual and analytical insights",
        "delivered_by": "Interactive graph console, Women-Safety threat lens, prioritised "
                        "alert queue and one-click PDF dossier",
        "module": "src/ui/ · src/analysis/women_safety.py · src/reporting/",
        "screen": "Women Safety · Reports & Alerts",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SAMPLE_DIR = DATA_DIR / "samples"
EXPORT_DIR = DATA_DIR / "exports"
DOCS_DIR = ROOT_DIR / "docs"

for _d in (SAMPLE_DIR, EXPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# CSV schema contract
# ──────────────────────────────────────────────────────────────────────────────

PERSON_COLUMNS = [
    "person_id",
    "name",
    "alias",
    "age",
    "gender",
    "city",
    "state",
    "occupation",
    "prior_offences",
    "offence_types",
    "first_arrest_year",
    "last_activity",
    "status",
    "risk_category",
]

RELATIONSHIP_COLUMNS = [
    "source_id",
    "target_id",
    "relation_type",
    "strength",
    "interaction_count",
    "first_seen",
    "last_seen",
    "evidence_source",
    "location",
]

CRIME_COLUMNS = [
    "case_id",
    "fir_number",
    "crime_type",
    "date",
    "city",
    "state",
    "severity",
    "status",
    "persons_involved",
]

PERSON_REQUIRED = ["person_id"]
RELATIONSHIP_REQUIRED = ["source_id", "target_id"]
CRIME_REQUIRED = ["case_id", "persons_involved"]

# Unstructured intelligence — FIR narratives, surveillance notes, social-media
# intercepts, agency reports. ``text`` is the only column the NLP layer truly needs;
# everything else is provenance that makes an extracted entity traceable back to a
# document, a date and a case, which is what turns a lead into something actionable.
DOCUMENT_COLUMNS = ["doc_id", "source_type", "title", "date", "city", "case_id", "text"]
DOCUMENT_REQUIRED = ["text"]

# Fuzzy header mapping — accept the column names real agencies actually ship.
COLUMN_ALIASES: dict[str, list[str]] = {
    "person_id": ["id", "personid", "person", "accused_id", "accusedid", "suspect_id", "uid", "code"],
    "name": ["full_name", "fullname", "person_name", "accused_name", "suspect_name"],
    "alias": ["aliases", "aka", "nickname", "alias_name", "known_as"],
    "age": ["years", "age_years", "current_age"],
    "gender": ["sex"],
    "city": ["district", "town", "location", "place"],
    "state": ["region", "province", "state_ut"],
    "occupation": ["profession", "job", "employment", "work"],
    "prior_offences": ["prior_offenses", "criminal_record_count", "record_count", "num_cases",
                       "cases", "offence_count", "priors"],
    "offence_types": ["offense_types", "crimes", "known_offences", "known_offenses",
                      "charges", "offence_type"],
    "first_arrest_year": ["first_arrest", "first_case_year", "entry_year"],
    "last_activity": ["last_seen", "last_active", "last_activity_date", "last_known_activity"],
    "status": ["current_status", "custody_status", "state_of_accused"],
    "risk_category": ["risk", "risk_level", "threat_level", "risk_band"],
    "source_id": ["source", "from", "from_id", "person_a", "a", "node1", "src"],
    "target_id": ["target", "to", "to_id", "person_b", "b", "node2", "dst"],
    "relation_type": ["relation", "relationship", "link_type", "type", "edge_type", "association"],
    "strength": ["weight", "confidence", "score", "link_strength", "tie_strength"],
    "interaction_count": ["interactions", "contacts", "call_count", "frequency", "num_contacts"],
    "first_seen": ["from_date", "start_date", "first_contact"],
    "evidence_source": ["source_type", "evidence", "basis", "derived_from", "intel_source"],
    "location": ["place", "area", "meeting_place"],
    "case_id": ["case", "caseid", "crime_id", "incident_id"],
    "fir_number": ["fir", "fir_no", "firno", "report_number"],
    "crime_type": ["offence", "offense", "crime", "category", "incident_type"],
    "date": ["crime_date", "incident_date", "occurred_on", "reported_on"],
    "severity": ["gravity", "seriousness", "severity_level", "priority"],
    "persons_involved": ["accused", "persons", "involved", "accused_ids", "participants",
                         "person_ids", "suspects"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Domain vocabularies
# ──────────────────────────────────────────────────────────────────────────────

RELATION_TYPES = [
    "Co-Accused",
    "Communication",
    "Financial",
    "Family",
    "Associate",
    "Gang Member",
    "Handler",
    "Courier Link",
    "Shelter Provider",
]

# Relative evidentiary weight of each link type when computing weighted influence.
RELATION_WEIGHTS: dict[str, float] = {
    "Co-Accused": 1.00,
    "Handler": 0.95,
    "Gang Member": 0.90,
    "Financial": 0.85,
    "Courier Link": 0.80,
    "Communication": 0.65,
    "Associate": 0.55,
    "Family": 0.50,
    "Shelter Provider": 0.45,
}

EVIDENCE_SOURCES = [
    "CDR Analysis",
    "FIR Linkage",
    "Physical Surveillance",
    "Financial Trail",
    "Informant Report",
    "Seizure Recovery",
    "Digital Forensics",
]

CRIME_TYPES = [
    "Narcotics Trafficking",
    "Arms Smuggling",
    "Extortion",
    "Human Trafficking",
    "Cyber Fraud",
    "Money Laundering",
    "Vehicle Theft Ring",
    "Counterfeit Currency",
    "Kidnapping",
    "Organised Robbery",
    "Illegal Betting",
    "Hawala Operation",
    # ── Women Safety Division mandate (PS 26189 sits under this division) ──────
    "Trafficking of Women",
    "Trafficking of Minors",
    "Commercial Sexual Exploitation",
    "Abduction of Women",
    "Dowry Harassment",
    "Domestic Violence Ring",
    "Cyber Stalking",
    "Online Sexual Harassment",
    "Sextortion Racket",
    "Obscene Content Distribution",
    "Matrimonial Fraud",
    "Acid Attack Conspiracy",
]

# Baseline severity (1–5) used by the synthetic generator and as a fallback.
CRIME_SEVERITY: dict[str, int] = {
    "Narcotics Trafficking": 5,
    "Arms Smuggling": 5,
    "Human Trafficking": 5,
    "Kidnapping": 5,
    "Money Laundering": 4,
    "Hawala Operation": 4,
    "Counterfeit Currency": 4,
    "Extortion": 4,
    "Organised Robbery": 3,
    "Cyber Fraud": 3,
    "Vehicle Theft Ring": 2,
    "Illegal Betting": 2,
    # Women Safety Division — organised offences against women and minors carry
    # the top severity band by NCRB convention; digital offences are graded by
    # the degree of coercion and organisation involved.
    "Trafficking of Women": 5,
    "Trafficking of Minors": 5,
    "Commercial Sexual Exploitation": 5,
    "Abduction of Women": 5,
    "Acid Attack Conspiracy": 5,
    "Sextortion Racket": 4,
    "Domestic Violence Ring": 4,
    "Dowry Harassment": 4,
    "Online Sexual Harassment": 3,
    "Cyber Stalking": 3,
    "Obscene Content Distribution": 3,
    "Matrimonial Fraud": 3,
}

# ──────────────────────────────────────────────────────────────────────────────
# Women Safety Division lens (PS 26189 · NCRB Women Safety Division)
#
# NETRA carries a dedicated analytical lens for the division that owns this
# problem statement. Any case whose crime type appears in ``WOMEN_SAFETY_CRIMES``
# contributes to a per-actor Women-Safety Threat Index (WSTI) and to the
# trafficking-corridor detector.
# ──────────────────────────────────────────────────────────────────────────────

WOMEN_SAFETY_CRIMES: set[str] = {
    "Trafficking of Women",
    "Trafficking of Minors",
    "Commercial Sexual Exploitation",
    "Abduction of Women",
    "Dowry Harassment",
    "Domestic Violence Ring",
    "Cyber Stalking",
    "Online Sexual Harassment",
    "Sextortion Racket",
    "Obscene Content Distribution",
    "Matrimonial Fraud",
    "Acid Attack Conspiracy",
    "Human Trafficking",  # legacy/general label also counts
}

# Offences committed against minors escalate the index — a statutory aggravator.
MINOR_VICTIM_CRIMES: set[str] = {
    "Trafficking of Minors",
    "Commercial Sexual Exploitation",
}

# Digitally-mediated offences against women. Isolated in their own bucket because
# the PS theme is *Blockchain & Cybersecurity* — the cyber-enabled subset is what
# ties this network analysis to that theme.
CYBER_WOMEN_SAFETY_CRIMES: set[str] = {
    "Cyber Stalking",
    "Online Sexual Harassment",
    "Sextortion Racket",
    "Obscene Content Distribution",
    "Matrimonial Fraud",
}

# Women-Safety Threat Index component weights (must sum to 1.0).
WSTI_WEIGHTS: dict[str, float] = {
    "case_involvement": 0.30,   # how many WS cases the actor appears in
    "severity": 0.24,           # peak severity of those cases
    "minor_victim": 0.18,       # statutory aggravator for offences against minors
    "network_reach": 0.16,      # structural ability to move victims or proceeds
    "recidivism": 0.12,         # prior-offence depth
}

WS_PARAMS = {
    "corridor_min_cities": 3,       # distinct cities to call a movement corridor
    "corridor_min_actors": 3,       # actors that must span those cities
    "repeat_offender_min_cases": 2,  # WS cases that make an actor a repeat offender
    "high_threat_wsti": 60.0,       # WSTI at or above which an actor is escalated
}

PERSON_STATUS = ["Active", "Under Surveillance", "In Judicial Custody", "Absconding", "On Bail", "Deceased"]

RISK_CATEGORIES = ["Critical", "High", "Medium", "Low"]

OPERATIONAL_ROLES = [
    "Kingpin",
    "Lieutenant",
    "Broker",
    "Financier",
    "Enforcer",
    "Foot Soldier",
    "Peripheral",
]

# ──────────────────────────────────────────────────────────────────────────────
# Key-player scoring — Influence Index weights (must sum to 1.0)
# ──────────────────────────────────────────────────────────────────────────────

INFLUENCE_WEIGHTS: dict[str, float] = {
    "pagerank": 0.26,
    "betweenness": 0.24,
    "weighted_degree": 0.18,
    "eigenvector": 0.16,
    "closeness": 0.10,
    "core_number": 0.06,
}

# ──────────────────────────────────────────────────────────────────────────────
# Anomaly detection — ensemble weights (must sum to 1.0)
# ──────────────────────────────────────────────────────────────────────────────

ANOMALY_WEIGHTS: dict[str, float] = {
    "gnn_residual": 0.30,
    "isolation_forest": 0.30,
    "rule_engine": 0.40,
}

# Saturation denominator for the rule-engine score. Set to the realistic maximum rule
# mass a single actor accumulates (~3 co-firing rules), NOT the theoretical sum of all
# eight — many rules are mutually exclusive (you cannot be both isolated and a hub).
RULE_SATURATION = 3.0

# Model-based signals (GNN, Isolation Forest) are down-weighted for barely-connected
# actors that trigger no expert rule: a lone node being a statistical outlier is
# expected, not intelligence-relevant. Rule-flagged actors are never gated.
ANOMALY_MODEL_GATE_FLOOR = 0.35

SUSPICION_BANDS = [
    (80, "Critical", "#ff3b5c"),
    (65, "High", "#ff8c42"),
    (45, "Medium", "#ffd23f"),
    (0, "Low", "#4ade80"),
]

# Expert-rule tuning knobs.
RULE_PARAMS = {
    "broker_betweenness_pct": 0.85,      # betweenness percentile to qualify as a broker
    "broker_degree_pct": 0.45,           # ...while degree stays below this percentile
    "bridge_min_communities": 3,         # distinct modules touched to count as a connector
    "mule_financial_ratio": 0.60,        # share of a node's links that are Financial
    "mule_max_offences": 1,              # ...with at most this many prior offences
    "isolate_max_degree": 2,             # degree at or below this = structurally isolated
    "isolate_min_severity": 4,           # ...but involved in crimes this severe
    "alias_min": 3,                      # number of recorded aliases that looks evasive
    "dormancy_years": 3,                 # gap after which renewed activity is a spike
    "dormancy_min_new_links": 5,         # fresh contacts needed to call it a burst
    "dormancy_new_link_ratio": 0.60,     # ...as a share of all the actor's contacts
    "hierarchy_gap": 0.35,               # centrality gap indicating rank inversion
    "gatekeeper_min_degree": 3,          # cut-vertex must have this many links to count
    "gatekeeper_min_cut": 3,             # ...and must gate a cell of at least this size
}

# ──────────────────────────────────────────────────────────────────────────────
# Light GNN hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────

GNN_CONFIG = {
    "embedding_dim": 32,
    "hidden_dim": 48,
    # 200 epochs is the measured sweet spot on the reference dataset: held-out
    # link-prediction AUC peaks near 0.96 and drifts down past ~250 as the encoder
    # starts memorising individual edges.
    "epochs": 200,
    "learning_rate": 0.02,
    "negative_samples": 5,
    "seed": 42,
    "l2": 1e-4,
}

# ──────────────────────────────────────────────────────────────────────────────
# Performance guards
# ──────────────────────────────────────────────────────────────────────────────

PERF = {
    "exact_centrality_max_nodes": 1500,   # above this, approximate betweenness/closeness
    "betweenness_sample_k": 320,          # pivots for approximate betweenness
    # Canvas cap. 500 is the measured ceiling for a smooth Plotly pan/zoom inside a
    # 1 GB Streamlit Community Cloud container; above it we keep the top-N nodes by
    # the active ranking metric and report exactly how many were withheld.
    "max_render_nodes": 500,
    "max_render_edges": 2500,
    # Hard ingestion ceilings. Exceeding these is refused with a clear message rather
    # than left to OOM-kill the container mid-analysis.
    "max_upload_rows": 200_000,
    "max_analysis_nodes": 20_000,
    # Streamlit cache bounds — cap retained entries so a judge sweeping the sliders
    # cannot fill the container's memory with stale pipeline results.
    "cache_max_entries": 3,
    "cache_ttl_seconds": 3600,
}

# ──────────────────────────────────────────────────────────────────────────────
# Visual palette
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg": "#0a0e17",
    "surface": "#121826",
    "surface_2": "#1a2233",
    "border": "#243044",
    "text": "#e6edf3",
    "text_dim": "#8b98a9",
    "accent": "#00d4ff",
    "accent_2": "#7c5cff",
    "critical": "#ff3b5c",
    "high": "#ff8c42",
    "medium": "#ffd23f",
    "low": "#4ade80",
    "grid": "#1e2735",
    "edge": "#2f3d55",
}

RISK_COLORS = {
    "Critical": PALETTE["critical"],
    "High": PALETTE["high"],
    "Medium": PALETTE["medium"],
    "Low": PALETTE["low"],
}

ROLE_COLORS = {
    "Kingpin": "#ff3b5c",
    "Lieutenant": "#ff8c42",
    "Broker": "#00d4ff",
    "Financier": "#7c5cff",
    "Enforcer": "#f472b6",
    "Foot Soldier": "#4ade80",
    "Peripheral": "#64748b",
}

# Qualitative sequence for communities — accessible in dark mode, distinguishable
# for the most common forms of colour-vision deficiency.
COMMUNITY_COLORS = [
    # Six brand hues first, then six chosen for *separation* rather than as shade
    # variants of the first six. The previous ordering repeated the same hues one shade
    # lighter from index 6 onward, so the default seven-module sample network drew
    # Module 7 in effectively the same cyan as Module 1 — and the module legend, which
    # the demo leans on to single out the planted cell, stopped meaning anything.
    # The colour wheel is already spent after six saturated hues, so the seventh is a
    # neutral platinum: the one value guaranteed not to read as any of the others.
    "#00d4ff",  # cyan
    "#ff8c42",  # orange
    "#7c5cff",  # violet
    "#4ade80",  # green
    "#ffd23f",  # yellow
    "#f472b6",  # pink
    "#e2e8f0",  # platinum
    "#14b8a6",  # teal
    "#e879f9",  # fuchsia
    "#a3e635",  # lime
    "#60a5fa",  # blue
    "#d6a06a",  # tan
    # Beyond twelve modules the legend is truncated anyway, so shade variants are fine.
    "#22d3ee", "#f59e0b", "#8b5cf6", "#10b981",
]

STATUS_COLORS = {
    "Active": "#ff8c42",
    "Under Surveillance": "#00d4ff",
    "In Judicial Custody": "#64748b",
    "Absconding": "#ff3b5c",
    "On Bail": "#ffd23f",
    "Deceased": "#475569",
}

# ──────────────────────────────────────────────────────────────────────────────
# NLP entity extraction (PS 26189: "extract important entities such as people,
# locations, vehicles, phone numbers, and organizations")
#
# Entity types are declared here with their display colour and the graph node
# kind they map to, so the extraction module, the entity table and the entity
# graph all agree without duplicating vocabulary.
# ──────────────────────────────────────────────────────────────────────────────

ENTITY_TYPES: list[dict[str, str]] = [
    {"key": "PERSON", "label": "Person", "icon": "👤", "colour": "#00d4ff"},
    {"key": "PHONE", "label": "Phone Number", "icon": "📞", "colour": "#4ade80"},
    {"key": "VEHICLE", "label": "Vehicle", "icon": "🚗", "colour": "#ff8c42"},
    {"key": "LOCATION", "label": "Location", "icon": "📍", "colour": "#f472b6"},
    {"key": "ORGANISATION", "label": "Organisation", "icon": "🏢", "colour": "#7c5cff"},
    {"key": "BANK_ACCOUNT", "label": "Bank Account", "icon": "🏦", "colour": "#ffd23f"},
    {"key": "CRYPTO_WALLET", "label": "Crypto Wallet", "icon": "⛓️", "colour": "#22d3ee"},
    {"key": "IMEI", "label": "Device IMEI", "icon": "📱", "colour": "#a78bfa"},
    {"key": "EMAIL", "label": "Email / Handle", "icon": "✉️", "colour": "#38bdf8"},
    {"key": "WEAPON", "label": "Weapon", "icon": "🔫", "colour": "#ff3b5c"},
    {"key": "CONTRABAND", "label": "Contraband", "icon": "📦", "colour": "#fb923c"},
    {"key": "CURRENCY", "label": "Currency Amount", "icon": "💵", "colour": "#34d399"},
]

ENTITY_COLORS: dict[str, str] = {e["key"]: e["colour"] for e in ENTITY_TYPES}
ENTITY_ICONS: dict[str, str] = {e["key"]: e["icon"] for e in ENTITY_TYPES}
ENTITY_LABELS: dict[str, str] = {e["key"]: e["label"] for e in ENTITY_TYPES}

NLP_PARAMS = {
    # Confidence floor below which an extracted mention is discarded.
    "min_confidence": 0.45,
    # Two entities mentioned in the same document are linked with this base strength;
    # co-occurrence is corroborative, not conclusive, so it starts below a recorded tie.
    "cooccurrence_strength": 0.40,
    # Each additional document that repeats a pair adds this much, capped below 1.0.
    "cooccurrence_bonus": 0.12,
    "cooccurrence_max": 0.88,
    # Documents longer than this are chunked so one narrative cannot dominate.
    "max_doc_chars": 40_000,
    # A single document mentioning more than this many entities is treated as a
    # bulk dump and excluded from co-occurrence linking (same logic as the
    # co-accused clique guard in the graph builder).
    "max_entities_per_doc": 25,
}

# ──────────────────────────────────────────────────────────────────────────────
# SQLite case store
#
# NETRA persists analysis runs, alerts and an append-only audit trail. On an
# ephemeral host (Streamlit Community Cloud) this survives only for the life of
# the container; on Docker/VPS the file is bind-mounted and durable. Override the
# location with the NETRA_DB_PATH environment variable.
# ──────────────────────────────────────────────────────────────────────────────

# SQLite case registry. Overridable with NETRA_DB_PATH so a container can mount a
# writable volume without rebuilding the image (see docker-compose.yml).
DB_PATH = Path(os.environ.get("NETRA_DB_PATH", DATA_DIR / "netra.db"))

# Set NETRA_DB_ENABLED=0 to run fully stateless (e.g. a read-only filesystem).
DB_ENABLED = os.environ.get("NETRA_DB_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
DB_TIMEOUT_SECONDS = 10.0
