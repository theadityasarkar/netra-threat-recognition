# 🛡️ NETRA — AI-Powered Criminal Network Analysis System

> **N**etwork **E**valuation & **T**hreat **R**ecognition **A**nalytics

<table>
<tr><td><b>Event</b></td><td>Smart India Hackathon 2026</td></tr>
<tr><td><b>Problem Statement</b></td><td><b>26189</b> — AI-Powered Criminal Network Analysis System</td></tr>
<tr><td><b>Organisation</b></td><td>Ministry of Home Affairs (MHA)</td></tr>
<tr><td><b>Department</b></td><td>National Crime Records Bureau (NCRB) · Women Safety Division</td></tr>
<tr><td><b>Category / Theme</b></td><td>Software · Blockchain &amp; Cybersecurity</td></tr>
<tr><td><b>Submission deadline</b></td><td>20 September 2026</td></tr>
</table>

NETRA converts multi-source criminal data — FIRs, Call Detail Records, financial trails,
surveillance reports, social-media intelligence, criminal-history databases and intelligence
agency reports — into an **interactive intelligence graph**, then applies graph theory, NLP
entity extraction and a lightweight **Graph Neural Network** to answer the questions an
investigator actually asks:

| Operational question | System capability | Screen |
|---|---|---|
| *What entities are hidden in this unstructured text?* | NLP extraction of 12 entity types (people, phones, vehicles, locations, orgs, accounts, wallets…) | **Entity Extraction** |
| *How is this network wired?* | Multi-relational weighted relationship graph + co-accused inference | **Network Graph** |
| *Who actually runs it?* | Key-Player detection — 10-measure centrality + Influence Index + role classification | **Key Players** |
| *What are we not seeing?* | Hybrid anomaly detection — GNN residuals ⊕ Isolation Forest ⊕ 8-rule expert engine | **Anomalies** |
| *Who threatens women and minors, and along which corridors?* | Women-Safety Threat Index (WSTI) + trafficking-corridor detection | **Women Safety** |
| *What do we act on today?* | Prioritised alert queue + one-click PDF / Markdown / CSV dossier | **Reports & Alerts** |

Built entirely on **free, open-source Python** — no paid APIs, no cloud lock-in, no licence
keys. Runs on a laptop and deploys free on Streamlit Community Cloud.

---

## ⚡ Quick Start (one command)

```bash
pip install -r requirements.txt && streamlit run app.py
```

The app opens at `http://localhost:8501` and **auto-loads a synthetic multi-syndicate
network** (actors, links, cases *and* intelligence narratives) — no CSV needed to start the
demo. It runs the full analysis on first load; adjust the sidebar and press **`▶ Run
Analysis`** to re-run.

> 🎬 **SIH demo walkthrough (script + timings): [`docs/PRESENTATION.md`](docs/PRESENTATION.md).**

Prefer `make`? &nbsp;`make install` · `make run` · `make test` · `make sample`.

---

## 🎯 How NETRA answers PS 26189, line by line

The problem statement names seven data sources and six capabilities. Each maps to a concrete
module — a live **requirement→implementation traceability matrix** is rendered on the
Dashboard so a judge can follow it without reading code.

**Seven mandated data sources → ingestion paths**

| PS data source | NETRA handling |
|---|---|
| FIRs and police reports | Structured CSV roster + unstructured narrative NLP |
| Call Detail Records (CDRs) | Dedicated CDR ingestion → communication edges (`src/ingestion/cdr.py`) |
| Financial transaction records | Financial relationship edges + money-mule detection |
| Surveillance reports | Free-text entity extraction |
| Social-media intelligence | Free-text entity extraction |
| Criminal-history databases | Person roster with prior-offence history |
| Intelligence-agency reports | Free-text entity extraction |

**Six mandated capabilities → screens** — collect/process multi-source data → *Data &
Sources*; extract entities → *Entity Extraction*; build relationship maps → *Network Graph*;
identify key individuals → *Key Players*; detect suspicious patterns → *Anomalies*; assist
investigators with visual/analytical insight → *Women Safety* + *Reports & Alerts*.

---

## 🏗️ Architecture — five layers

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  1. DATA INGESTION                                   src/ingestion/              │
│     CSV upload · schema validation · fuzzy column auto-mapping · CDR parsing ·   │
│     NLP entity extraction (entities.py) · seeded synthetic generator            │
└────────────────────────────────┬───────────────────────────────────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  2. GRAPH BUILDING                                   src/graph/                  │
│     NetworkX weighted multi-relational graph · co-accused inference · 24 node   │
│     features                                                                     │
└────────────────────────────────┬───────────────────────────────────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  3. AI ANALYSIS                                      src/analysis/               │
│     centrality · communities (Louvain) · GraphSAGE GNN (NumPy) · key_players ·  │
│     anomaly (hybrid) · women_safety (WSTI + trafficking corridors)              │
└────────────────────────────────┬───────────────────────────────────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  4. VISUALISATION & UI                     src/visualization/ + src/ui/          │
│     Plotly network canvas · 15+ analytical charts · 8 analyst screens · dark    │
│     "intelligence console" theme · bounded caching · session state              │
└────────────────────────────────┬───────────────────────────────────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  5. REPORTING & PERSISTENCE                 src/reporting/ + src/storage/         │
│     Severity-ranked alerts · ReportLab PDF dossier · Markdown brief · CSV/ZIP · │
│     SQLite case store (runs, alerts, watchlist, append-only audit trail)        │
└──────────────────────────────────────────────────────────────────────────────────┘
```

Full design rationale + algorithm notes: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## 📂 Folder structure

```
SIH/
├── app.py                          # Streamlit entry — routing, session state, caching, persistence
├── requirements.txt                # pinned, free/OSS only
├── Dockerfile                      # multi-stage production image (Python 3.12, non-root)
├── docker-compose.yml              # durable case-store volume + healthcheck
├── render.yaml · Procfile          # Render blueprint / PaaS process definition
├── runtime.txt · packages.txt      # Python pin / (intentionally no) apt packages
├── Makefile                        # install · run · test · docker · sample
├── LICENSE                         # MIT
├── .github/workflows/ci.yml        # pytest on 3.10 + 3.12
├── .streamlit/config.toml          # dark theme + server tuning
│
├── src/
│   ├── config.py                   # single source of truth: PS metadata, schemas, weights, palette
│   ├── pipeline.py                 # orchestrates layers 1→3, cached end-to-end, progress callback
│   │
│   ├── ingestion/
│   │   ├── loader.py               # CSV read, validation, fuzzy column mapping, load_documents
│   │   ├── entities.py             # NLP entity extraction — 12 types, India-specific patterns
│   │   ├── cdr.py                  # Call Detail Record analytics → communication edges
│   │   └── sample_data.py          # seeded synthetic syndicate + narrative generator
│   │
│   ├── graph/builder.py            # NetworkX graph + node feature matrix
│   │
│   ├── analysis/
│   │   ├── centrality.py           # 10 centrality measures (approx. for large graphs)
│   │   ├── communities.py          # Louvain + bridge/broker/cut-edge analysis
│   │   ├── gnn.py                  # 2-layer GraphSAGE encoder (pure NumPy, trained)
│   │   ├── key_players.py          # Influence Index + role inference + fragmentation impact
│   │   ├── anomaly.py              # hybrid anomaly detection + plain-language explanations
│   │   └── women_safety.py         # WSTI scoring + trafficking-corridor detection
│   │
│   ├── visualization/
│   │   ├── network_viz.py          # Plotly graph canvas + ego networks
│   │   └── charts.py               # analytical charts
│   │
│   ├── reporting/
│   │   ├── alerts.py               # 5-stream alert synthesis + prioritisation
│   │   └── report.py               # PDF / Markdown / CSV-ZIP exporters (PS-branded)
│   │
│   ├── storage/database.py         # SQLite case store — runs, alerts, watchlist, audit trail
│   │
│   └── ui/
│       ├── theme.py                # injected CSS, masthead, PS strip
│       ├── components.py           # metric cards, badges, alert cards
│       ├── state.py                # session-state contract + bounded cache decorator
│       └── pages/                  # 📊 overview · 🗂️ data_sources · 🔎 entities · 🕸️ network
│                                   # 👑 key_players · 🚨 anomalies · 🚺 women_safety · 📕 reports
│
├── data/  (samples/ · exports/)    ├── docs/  (ARCHITECTURE · DATA_SCHEMA · PRESENTATION)
├── scripts/generate_sample_data.py └── tests/  (ingestion · graph · analysis)
```

---

## 🧠 The AI layer, honestly described

Judges ask *"where is the AI?"* — the precise answer:

### 1. NLP entity extraction (`src/ingestion/entities.py`)
Pattern + gazetteer extraction over unstructured intelligence text, tuned for Indian data:
**12 entity types** — person, phone, vehicle (registration), location, organisation, bank
account, crypto wallet, IMEI, email/handle, weapon, contraband, currency amount. Each mention
is scored, de-duplicated via span arbitration, and shown **highlighted in its source
sentence** so an investigator can verify the lead against the narrative that produced it. A
transformer NER would add a ~500 MB dependency and GPU expectations for marginal gain on this
structured-entity task; the pattern approach is auditable, instant and free — see the "How
extraction works" panel in-app.

### 2. Light Graph Neural Network (`src/analysis/gnn.py`)
A **2-layer GraphSAGE-style encoder implemented from scratch in NumPy** (no PyTorch/TensorFlow
— keeps the install small and CPU-only).

- **Encoder:** `h⁽ˡ⁺¹⁾ᵥ = tanh( W⁽ˡ⁾ · [ h⁽ˡ⁾ᵥ ‖ mean_{u∈N(v)} h⁽ˡ⁾ᵤ ] )`
- **Training:** unsupervised **link prediction**, positive edges vs 5× degree-weighted
  negatives, logistic loss, **Adam**, analytic backprop, ~200 epochs. Held-out AUC ≈ 0.96.
- **Products:** role embeddings (2-D PCA), per-node **reconstruction residual** (high =
  structurally anomalous), and cosine **structural-twin search** (*"find every actor who
  behaves like this kingpin"* — the feature that finds successors after an arrest).

### 3. Hybrid anomaly detection (`src/analysis/anomaly.py`)
Hybrid because pure ML on a few hundred rows is not credible and pure rules are not "AI":

| Component | Weight | What it catches |
|---|---|---|
| GNN reconstruction residual | 30 % | actors whose connections defy the learnt network grammar |
| Isolation Forest (24 structural features) | 30 % | multivariate statistical outliers |
| Expert rule engine (8 rules) | 40 % | investigator doctrine (broker, money-mule, gatekeeper…) |

Every score ships with **human-readable reasons** — no black-box output reaches an officer.

### 4. Women-Safety Threat Index (`src/analysis/women_safety.py`)
The owning department is NCRB's Women Safety Division, so NETRA re-scores the network through
that mandate. The **WSTI** is a *content* score (independent of the structural anomaly band)
combining case involvement, peak severity, a **minor-victim statutory aggravator**, network
reach and recidivism, banded Low→Severe. A connected-component walk over women-safety cases
surfaces **trafficking corridors** (≥3 cities linked by ≥3 actors). A division-critical
offender can be invisible to generic centrality — this lens is what surfaces them.

---

## 📥 Data contract

Upload up to **four CSVs**. Only `persons` + `relationships` are required; headers are
auto-mapped from common variants (`id`/`person_id`/`accused_id` all resolve).

| File | Required? | Key columns |
|---|---|---|
| `persons.csv` | ✅ | `person_id, name, alias, age, gender, city, state, occupation, prior_offences, offence_types, first_arrest_year, last_activity, status, risk_category` |
| `relationships.csv` | ✅ | `source_id, target_id, relation_type, strength, interaction_count, first_seen, last_seen, evidence_source, location` |
| `crime_records.csv` | optional | `case_id, fir_number, crime_type, date, city, state, severity, status, persons_involved` |
| `documents.csv` | optional | `doc_id, source_type, title, date, city, case_id, text` — a `text` column alone is enough; powers Entity Extraction |

Full field semantics and validation rules: [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md).
Regenerate the sample set: `python scripts/generate_sample_data.py --persons 220 --seed 42`.

---

## 🖥️ Dashboard tour (8 screens)

| Tab | Purpose |
|---|---|
| **📊 Dashboard** | KPI strip, network snapshot, threat bands, distribution analytics, PS-26189 traceability matrix |
| **🗂️ Data & Sources** | The seven PS data sources mapped to what was actually loaded; data-quality report |
| **🔎 Entity Extraction** | Extracted entities highlighted in their source text; register, by-type gallery, actor resolution, front-business candidates |
| **🕸️ Network Graph** | Interactive Plotly canvas — layouts, colour by community/role/risk/anomaly, size by centrality, filters, ego-network drill-down |
| **👑 Key Players** | Influence Index leaderboard, role distribution, centrality radar, fragmentation-impact simulation, GNN structural-twin search |
| **🚨 Anomalies** | Suspicion leaderboard with per-actor explanations, anomaly scatter, suspicious-cluster cards, GNN embedding map |
| **🚺 Women Safety** | WSTI ranking, escalation queue, trafficking corridors, caseload profile — the owning-department view |
| **📕 Reports & Alerts** | Severity-ranked alert queue, PDF dossier, Markdown brief, CSV/ZIP bundle |

---

## 🧾 Tech stack & why each piece

`Python 3.10–3.14 (Docker pins 3.12)` — pinned exactly in `requirements.txt` to the set the
24-test suite passes on.

| Dependency | Role | Why this and not the heavier option |
|---|---|---|
| **Streamlit** | UI runtime + caching | Pure-Python dashboards with zero JS build; deploys free. No React/Flask split to host. |
| **NetworkX** | graph + centrality + Louvain | Reference-quality graph algorithms in pure Python; no native graph DB to operate. |
| **NumPy** | GraphSAGE encoder maths | The GNN is hand-written in NumPy → no **PyTorch/TensorFlow** (~2 GB, GPU expectations) for a CPU-only encoder. |
| **scikit-learn** | Isolation Forest, PCA, KMeans | Battle-tested unsupervised primitives; no deep-learning stack required. |
| **Plotly** | interactive network + charts | GPU-free client-side interactivity inside Streamlit; no D3/JS toolchain. |
| **ReportLab** | PDF dossier | Deterministic, dependency-light PDF; no headless-browser/LaTeX render farm. |
| **SQLite** (stdlib) | case store + audit trail | ACID, single-file, zero-config; a Postgres service would add ops burden for no benefit at this volume and would worsen the free-tier story. |

Everything is MIT / BSD / Apache-2.0. There is no paid API, cloud service or licence key
anywhere in the stack — which is exactly what makes the free deployment targets below possible.

---

## 🚀 Deploy free

### A. Streamlit Community Cloud (primary — recommended)
1. Push this folder to a **public GitHub repo**.
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo.
3. Main file path: `app.py` → **Deploy**.

`requirements.txt` and `packages.txt` are read automatically; pick Python 3.12 in *Advanced
settings*. No secrets, no database, no billing. Cold start ≈ 25 s, warm ≈ 1 s. The case store
runs in the container filesystem (ephemeral — the app states this honestly in the sidebar).

### B. livemy.app / Railway / Fly.io / Heroku (any `$PORT` PaaS)
These platforms inject a `$PORT` and run a process. NETRA already ships what they read:

- **`Procfile`** defines the web process (`streamlit run app.py --server.port=$PORT …`).
- **`runtime.txt`** pins Python 3.12.8.
- **`requirements.txt`** pins dependencies.

Steps: push to GitHub → connect the repo on the platform → it detects the Procfile and
requirements and builds. If the platform asks for a start command explicitly, use:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

### C. Render (free web service — Blueprint included)
`render.yaml` is a ready Blueprint. On [render.com](https://render.com): **New +** →
**Blueprint** → point at the repo. Render provisions the service, installs from
`requirements.txt`, health-checks `/_stcore/health`, and sets `NETRA_DB_PATH=/tmp/netra.db`
(free-tier disk is ephemeral). No dashboard clicks needed.

### D. Self-host with Docker (durable case store — VPS)
For a real, **durable** audit trail (the case store survives restarts), self-host the
container. The bind-mounted volume is what makes persistence real here.

```bash
docker compose up --build          # → http://localhost:8501, ./data mounted for the SQLite store
```

or without compose:

```bash
docker build -t netra:latest .
docker run --rm -p 8501:8501 -v "$PWD/data:/app/data" netra:latest
```

The image is multi-stage (no build toolchain shipped), runs as a **non-root** user, and
exposes Streamlit's `/_stcore/health` endpoint for orchestrator health checks. Behind Nginx,
proxy `/` to `127.0.0.1:8501` with WebSocket upgrade headers.

| Target | Cost | Case store durability |
|---|---|---|
| Streamlit Community Cloud | free | ephemeral |
| livemy.app / Railway / Fly.io / Heroku (free tiers) | free | ephemeral |
| Render free web service | free | ephemeral (`/tmp`) |
| Docker on a VPS | your VPS | **durable** (bind-mounted volume) |

---

## ✅ Tests & CI

```bash
python -m pytest -q          # 103 tests
```

Covers schema validation and document ingestion, graph-construction invariants, centrality
sanity bounds, GNN training convergence, and anomaly-score determinism. GitHub Actions
(`.github/workflows/ci.yml`) runs the suite on Python 3.10 and 3.12 for every push and PR.

---

## 🔐 Responsible use

A **decision-support** tool, not an automated accusation engine:

- **All sample data is synthetic** — names, FIR numbers and locations are procedurally
  generated and match no real individual.
- **No predictive policing.** Scores describe *observed graph structure* in data the agency
  already holds — not predictions of future criminality.
- **Explainability is mandatory.** Every alert and suspicion score carries the specific
  reasons that produced it, so an officer can audit and overrule it.
- **Human-in-the-loop by construction.** Output is a prioritised review queue; the system
  takes no action. Identifiers can be **redacted on screen** for briefings.
- **Auditable.** Every consequential action is written to an append-only audit trail in the
  case store.

---

## 📊 Performance

Measured on a laptop CPU, synthetic network ~234 nodes / 1,000 edges (warm, `st.cache_data`-cached):

| Stage | Time |
|---|---|
| Ingestion + validation + entity extraction | ~0.1 s |
| Graph build + 24 features | 0.21 s |
| Centrality suite | 0.72 s |
| Louvain communities | 0.27 s |
| GNN training (200 epochs) + role clustering | 1.12 s |
| Anomaly detection + Women-Safety lens | ~0.5 s |
| **Total warm pipeline** | **≈ 3 s** |
| Cached re-render (tab switch) | < 0.1 s |

Graphs above 1,500 nodes switch betweenness/closeness to k-sample approximation; the render
canvas is capped at 500 nodes (top-ranked kept) to hold interaction latency down inside a
1 GB free-tier container.

---

*Built for Smart India Hackathon 2026 · PS 26189 · Ministry of Home Affairs · NCRB Women Safety Division.*
