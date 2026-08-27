# 🛡️ NETRA — Complete Project Manual & Working Guide
> **N**etwork **E**valuation & **T**hreat **R**ecognition **A**nalytics  
> **Smart India Hackathon 2026 · Problem Statement 26189**  
> **Organisation:** Ministry of Home Affairs (MHA) · **Division:** NCRB Women Safety Division  

---

## 📑 Table of Contents
1. [Project Overview & Problem Statement](#1-project-overview--problem-statement)
2. [Core Architecture & How NETRA Works](#2-core-architecture--how-netra-works)
3. [Deep Dive into Algorithms & Mathematical Models](#3-deep-dive-into-algorithms--mathematical-models)
4. [New Enhancements Added](#4-new-enhancements-added)
5. [User Guide: How to Use Every Screen](#5-user-guide-how-to-use-every-screen)
6. [Testing & Verification](#6-testing--verification)
7. [Deployment Guide (Local, Docker, Cloud)](#7-deployment-guide-local-docker-cloud)

---

## 1. Project Overview & Problem Statement

### 🎯 What is NETRA?
**NETRA** is an AI-powered intelligence and criminal network analysis system designed for law enforcement agencies (LEAs), intelligence analysts, and police investigators.

It ingests multi-source structured and unstructured data — **Police FIRs, Call Detail Records (CDRs), Banking & Crypto Transactions, Surveillance Reports, Criminal Records, and Intel Briefs** — and automatically constructs a **multi-relational intelligence graph**. It then applies Graph Theory, pure NumPy **Graph Neural Networks (GraphSAGE)**, NLP Entity Extraction (English + Hindi), and Hybrid Anomaly Detection to uncover hidden kingpins, money mules, cut-outs, and human trafficking corridors.

### 🏛️ PS 26189 Requirements Matrix
| PS Mandate | How NETRA Solves It | Screen |
|---|---|---|
| **1. Multi-source Ingestion** | Automated ingestion & column mapping for 7 data sources + Live CDR Stream Simulator | `Data & Sources` |
| **2. Entity Extraction** | Multi-lingual NLP (English + Devanagari Hindi + Hinglish) for 12 entity types | `Entity Extraction` |
| **3. Relationship Mapping** | Weighted multi-relational NetworkX graph + co-accused inference | `Network Graph` |
| **4. Key Individual Identification** | 10 Centrality Metrics Suite + Composite Influence Index + Disruption Simulation | `Key Players` |
| **5. Suspicious Pattern Detection** | Hybrid ensemble (GraphSAGE GNN Residuals ⊕ Isolation Forest ⊕ 8-Rule Expert Engine) | `Anomalies` |
| **6. Women Safety & Investigative Insight** | Women-Safety Threat Index (WSTI), Trafficking Corridors, One-click PDF/Neo4j exports | `Women Safety`, `Reports & Alerts` |

---

## 2. Core Architecture & How NETRA Works

NETRA is designed with a **5-Layer Modular Architecture**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: DATA INGESTION & NLP (src/ingestion/)                               │
│  • CSV loaders with fuzzy header auto-mapping                                │
│  • Multi-lingual Entity Extraction (English, Hindi Devanagari, Hinglish)      │
│  • Call Detail Record (CDR) parser & Cell Tower Dump analyzer                 │
│  • Real-Time Telemetry & Live CDR Stream Simulator                           │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: MULTI-RELATIONAL GRAPH BUILDING (src/graph/)                        │
│  • NetworkX multi-edge directed graph builder                                 │
│  • Co-accused link inference from FIR case records                           │
│  • 24 structural node features computation (degree, triangles, ego-density)   │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: AI & GRAPH THEORY ANALYTICS (src/analysis/)                         │
│  • 10 Centrality metrics (PageRank, Betweenness, Eigenvector, Closeness, etc.)│
│  • Louvain Modularity Community Detection                                     │
│  • 2-Layer GraphSAGE GNN (Implemented in pure NumPy — No PyTorch needed!)     │
│  • Hybrid Anomaly Detection Engine (Residuals + Isolation Forest + Rules)    │
│  • Women-Safety Threat Index (WSTI) & Interstate Trafficking Route Inferrer   │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: VISUALIZATION & USER INTERFACE (src/visualization/ + src/ui/)       │
│  • Interactive 2D/3D Plotly WebGL network canvas                              │
│  • Live Text & Hindi FIR Interactive Analyzer Playground                      │
│  • 8 Analyst Screens with Dark Intelligence Console Theme                     │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 5: DECISION SUPPORT, STORAGE & EXPORT (src/reporting/ + src/storage/)  │
│  • Automated AI Executive Intelligence Briefing (MHA / NCRB Standard Format)  │
│  • ReportLab Automated PDF Dossier, Markdown Brief & CSV Bundles              │
│  • Open Graph Exports: Neo4j Cypher (.cql) & GraphML (.graphml)               │
│  • Durable SQLite Case Registry with SHA-256 Audit Trail                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Deep Dive into Algorithms & Mathematical Models

### 1. Pure NumPy GraphSAGE (GNN without PyTorch)
To ensure NETRA runs instantaneously on laptops and free cloud tiers without downloading a 2GB+ PyTorch binary, GraphSAGE is implemented with vectorized **NumPy** math:
- **Neighborhood Aggregation:** Mean aggregator pooling neighbor feature vectors:
  $$h_{\mathcal{N}(v)}^{(k)} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)}$$
- **Layer Update & Non-Linearity:**
  $$h_v^{(k)} = \text{ReLU}\left( W^{(k)} \cdot \left[ h_v^{(k-1)} \,\|\, h_{\mathcal{N}(v)}^{(k)} \right] + b^{(k)} \right)$$
- **Structural Embeddings & Residuals:** Used for self-supervised link prediction AUC and reconstruction anomaly residuals.

### 2. Composite Influence Index
Rather than relying on single degree count, NETRA blends 6 complementary centrality measures:
$$\text{Influence} = 0.25 \cdot \text{PageRank} + 0.25 \cdot \text{Betweenness} + 0.20 \cdot \text{WeightedDegree} + 0.15 \cdot \text{Eigenvector} + 0.10 \cdot \text{Closeness} + 0.05 \cdot \text{K-Core}$$

### 3. Women-Safety Threat Index (WSTI)
A dedicated formula specifically crafted for the NCRB Women Safety mandate:
$$\text{WSTI} = \text{Base Offence Severity} + \text{Minor Victim Weight} + \text{Syndicate Centrality Multiplier} + \text{Trafficking Corridor Proximity}$$

### 4. Hybrid Anomaly Detection Ensemble
1. **GNN Reconstruction Residual:** Identifies nodes whose connectivity contradicts their attributes.
2. **Isolation Forest:** Multi-dimensional outlier detection across 24 topological features.
3. **8-Rule Heuristic Engine:** Flags cut-outs, sleeper nodes, money mules, burner phones, and cross-border brokers.

---

## 4. New Enhancements Added

1. **🌐 Multi-lingual / Hindi (हिन्दी) & Hinglish NLP Support**:
   - Devanagari Person Cues (`आरोपी`, `अभियुक्त`, `पीड़िता`, `सरगना`, `दलाल`, `ठेकेदार`, etc.)
   - Devanagari Location Cues (`थाना`, `कोतवाली`, `चौकी`, `गाँव`, `मोहल्ला`, `नगर`, `जिला`)
   - Regional Weapon & Contraband Lexicons (`तमंचा`, `कट्टा`, `कारतूस`, `चरस`, `गांजा`, `अफीम`, `चिट्टा`, `नकली नोट`)
   - Hindi Currency Parsing (`₹ 50,000`, `2.5 लाख`, `1 करोड़`)
2. **⚡ Real-Time Text & Hindi FIR Playground**:
   - Interactive live analyzer on the *Entity Extraction* page where investigators can test custom text in real-time.
3. **📡 Real-Time Telemetry & Live CDR Stream Simulator**:
   - On the *Data & Sources* screen, simulates live incoming packet streams (CDRs, Tower Dumps, Wire Transfers) with visual threat badges.
4. **🤖 AI Executive Intelligence Briefing**:
   - Automated MHA / NCRB standard intelligence briefing with threat landscape, apex coordinator, critical disruption targets, and tactical directives.
5. **🕸️ Open Graph Exports (Neo4j Cypher & GraphML)**:
   - One-click `.cql` export for Neo4j and `.graphml` export for Gephi, Cytoscape, and i2 Analyst's Notebook.

---

## 5. User Guide: How to Use Every Screen

### 🚀 Starting the App
Run in terminal:
```bash
python -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 📱 Navigating the 8 Intelligence Screens:

#### 1. 📊 Dashboard
- Shows high-level KPIs: Total Actors, Relationships, Modularity, Critical Threat Alerts.
- Live **PS 26189 Traceability Matrix** demonstrating requirement-to-code mapping for hackathon judges.

#### 2. 🗂️ Data & Sources
- Displays status of the 7 mandated data sources.
- **⚡ Live Telemetry Simulator:** Click **`Inject Telemetry Stream Batch`** to simulate real-time CDR feeds.
- **CDR Analytics:** Cell tower dump distributions, call duration heatmaps, and night-time communication patterns.
- **Export Raw Data:** Download CSV copies of current actors, relationships, and cases.

#### 3. 🔎 Entity Extraction
- **Entity Register:** View 12 extracted entity types (PERSON, PHONE, VEHICLE, LOCATION, BANK_ACCOUNT, CRYPTO_WALLET, etc.).
- **Source Documents:** See documents with extracted entities color-highlighted in place.
- **⚡ Live Text & Hindi FIR Analyzer:** Paste custom Hindi/English FIR narratives and click to extract entities with zero delay.
- **Front Businesses:** Flags recruitment agencies, spas, and travel agencies suspected of human trafficking fronts.

#### 4. 🕸️ Network Graph
- **Interactive Graph Canvas:** Zoom, pan, drag nodes, and adjust gravity/physics.
- **Node Coloring:** Switch between Syndicate Communities, Risk Levels, or Node Types.
- **Node Dossier:** Click any node to view its phone numbers, vehicles, associates, and criminal background.

#### 5. 👑 Key Players
- **Influence Ranking:** Top network actors ranked by composite centrality.
- **Tactical Disruption Simulation:** Displays the **#1 High-Value Target** whose arrest will cause maximum structural fragmentation of the criminal network.
- **Role Taxonomy:** Classifies actors into Kingpins, Brokers, Lieutenants, Enforcers, and Mules.

#### 6. 🚨 Anomalies
- **Suspicion Scoreboard:** Identifies hidden actors operating under the radar.
- **Ensemble Breakdown:** View exact contributing scores (GNN Residuals vs Isolation Forest vs Rule Engine).
- **Explainability:** Transparent reasons for why every node was flagged.

#### 7. 🚺 Women Safety
- **WSTI Threat Leaderboard:** High-risk targets involved in crimes against women and POCSO offences.
- **Trafficking Corridors:** Inferred transit routes between source locations (e.g. Siliguri, Nepal borders) and destination metros (e.g. Mumbai, Delhi).
- **Vulnerability Profiling:** Protection prioritization for victims and minors.

#### 8. 📕 Reports & Alerts
- **🤖 AI Executive Intelligence Briefing:** View and copy the structured MHA / NCRB threat report.
- **Priority Alert Queue:** Filter alerts by Critical, High, Medium, or Category.
- **Export Suite:**
  - 📄 **Markdown Brief** (`.md`)
  - 📕 **Official PDF Dossier** (`.pdf` via ReportLab)
  - 🗜 **CSV Data Bundle** (`.zip`)
  - 🌐 **Neo4j Cypher Script** (`.cql`)
  - 🕸️ **GraphML XML** (`.graphml`)

---

## 6. Testing & Verification

NETRA includes an automated test suite verifying all modules:

```powershell
python -m pytest
```

### ✅ Test Suite Results:
- `tests/test_analysis.py` (Centrality, Louvain, GNN, Key Players, Anomalies) — **PASSED**
- `tests/test_cdr.py` (CDR parsing, tower analysis, night calls) — **PASSED**
- `tests/test_entities.py` (NER pattern extraction, normalisation, masking) — **PASSED**
- `tests/test_enhancements.py` (Hindi NER, Cypher, GraphML, AI Brief) — **PASSED**
- `tests/test_graph.py` (Graph construction, feature extraction, co-accused) — **PASSED**
- `tests/test_ingestion.py` (CSV loaders, fuzzy headers, validation) — **PASSED**
- `tests/test_storage.py` (SQLite case store, audit logs, persistence) — **PASSED**
- `tests/test_women_safety.py` (WSTI calculation, trafficking corridors) — **PASSED**

**Result:** `107 passed in 12.63s` (100% Pass Rate).

---

## 7. Deployment Guide (Local, Docker, Cloud)

### Option A: Local Execution (Fastest)
```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python -m streamlit run app.py
```

### Option B: Docker Container Deployment
```bash
# Build the production container
docker build -t netra:2.0.0 .

# Run container
docker run -d -p 8501:8501 --name netra-app netra:2.0.0
```

### Option C: Docker Compose (VPS / Server Self-Hosting)
```bash
docker compose up -d --build
```
Access at `http://<your-server-ip>:8501`. Persistent data is stored in the `netra-data` Docker volume.

### Option D: Cloud Deployment (Streamlit Cloud / Render / Railway)
- **Streamlit Community Cloud:** Connect GitHub repository, specify `app.py` as entry point. Zero configuration required.
- **Render / Railway:** Uses included `render.yaml` and `Procfile`. Port `$PORT` is dynamically bound.

---

### 🛡️ Built for India's Security & Women Safety · SIH 2026 PS 26189
