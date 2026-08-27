# NETRA — Architecture & Algorithm Notes

> Companion to the [README](../README.md). This document explains *how* each layer works
> and *why* it was built that way. Where the README answers "what does it do", this
> answers "what is the algorithm, and what were the design trade-offs".

NETRA is a five-layer pipeline. Data flows strictly downward; every layer is a pure
function of the layer above it, and nothing below Layer 4 imports Streamlit. That
separation is deliberate — it is what lets the whole analysis run in a notebook, a CLI
script, or the [test suite](../tests) without a browser.

```
Layer 1  Ingestion      CSV → validated, reconciled DataFrames        src/ingestion/
Layer 2  Graph build    DataFrames → weighted NetworkX graph          src/graph/
Layer 3  AI analysis    graph → centrality · communities · GNN ·      src/analysis/
                          key players · anomalies
Layer 4  Dashboard      AnalysisResult → 5 interactive tabs           src/ui/ + src/visualization/
Layer 5  Reporting      AnalysisResult → alerts · PDF · MD · CSV       src/reporting/
```

The connective tissue is [`src/pipeline.py`](../src/pipeline.py): `run_pipeline()` drives
Layers 2–3 and returns a single frozen [`AnalysisResult`](../src/pipeline.py) dataclass
that Layers 4 and 5 consume read-only. [`src/config.py`](../src/config.py) is the single
source of truth for every schema, weight, threshold and colour — a reviewer can retune the
system's behaviour from that one file without touching logic.

---

## Design principles

These five principles recur throughout the code and explain most of the non-obvious
decisions:

1. **Explainability is not optional.** Every suspicion score and every alert carries the
   specific human-readable reasons that produced it. A score an officer cannot justify in
   a surveillance request is worse than no score. `reasons` is a first-class output, never
   a debug field.
2. **Evidence is never silently dropped.** A malformed roster or an ID that appears only in
   a wiretap log is a data-quality problem, not a reason to lose a link. The loader creates
   *stub* records and *reports* every repair rather than discarding rows.
3. **Determinism.** Every random process is seeded (GNN init/sampling, Louvain, Isolation
   Forest, k-means, approximate betweenness). The same input always yields the same output,
   so a judge who reruns the demo sees the numbers this document quotes.
4. **Hybrid over pure.** Pure ML on a few hundred rows is not statistically credible; pure
   rules are not "AI". Anomaly detection therefore *votes* across a learnt model, a
   statistical model, and an expert rule engine, and reports the breakdown.
5. **Free and light.** No PyTorch, no cloud, no paid API. The GNN is ~200 lines of NumPy.
   The whole stack installs in under 100 MB and runs CPU-only on a laptop.

---

## Layer 1 — Ingestion (`src/ingestion/`)

**Files:** [`loader.py`](../src/ingestion/loader.py), [`sample_data.py`](../src/ingestion/sample_data.py)

### Header normalisation & validation

Real agency exports never match a spec exactly, so [`loader.py`](../src/ingestion/loader.py)
is deliberately forgiving:

- **Canonicalisation.** Every incoming header is lower-cased and de-punctuated
  (`_canon()`), then resolved through the `COLUMN_ALIASES` table in
  [`config.py`](../src/config.py). `accused_id`, `personid`, `uid` and `suspect_id` all
  map to `person_id`; `weight`, `confidence` and `tie_strength` all map to `strength`.
- **Type coercion with sane defaults.** `_num()` coerces numerics and clips them to legal
  ranges (age 0–110, strength 0–1, severity 1–5). Dates are parsed with `format="mixed"`.
  Missing `risk_category` is *derived* from prior-offence counts via `pd.cut` rather than
  defaulting everyone to "Low".
- **Structured reporting.** Everything the loader fixes is recorded in a
  `ValidationReport` (`errors` / `warnings` / `info` / `renamed`). The dashboard surfaces
  this verbatim in the Data Quality panel. Only a missing *required* column
  (`person_id`; `source_id`+`target_id`) is a hard error that stops the pipeline.

### Reconciliation & stub creation

`prepare_dataset()` reconciles the three tables. Any ID that appears in the relationship
or crime tables but not in `persons.csv` gets a **stub person record** (`Unidentified
P0999`) so the link survives. This is the evidence-preservation principle in code: an
incomplete roster degrades data quality, it does not delete intelligence.

Undirected duplicate links (A→B and B→A) are collapsed to a single canonical pair (sorted
endpoints), keeping the strongest tie and summing interaction volume.

### Synthetic data generator

[`sample_data.py`](../src/ingestion/sample_data.py) makes the app demo-ready with zero
setup. It is deterministic per seed and deliberately *plants* the exact structures the
analysis layer is supposed to find, so a judge can verify detection against ground truth:

| Planted structure | Detector it exercises | ID at seed 42 |
|---|---|---|
| Hierarchical syndicates (Kingpin→Lieutenant→Soldier) | role inference, community detection | modules 1–6 |
| Sparse inter-module brokers (high betweenness, low degree) | broker rule, betweenness centrality | P0221–P0223 |
| Gateway chokepoint (a 7-actor cell joined by exactly one liaison) | cut-vertex / gatekeeper rule, fragmentation | P0224 |
| Money mule (many financial links, clean record) | financial-mule rule, Isolation Forest | P0232 |
| Insulated high-severity actor (serious record, ~1 contact) | isolated-offender rule | P0233 |
| Dormant-then-reactivated actor | dormancy-spike rule | P0234 |

The ground-truth key is attached as `persons_df.attrs["planted_anomalies"]` for the demo
script — **no analysis module ever reads it**. Detection is entirely blind.

---

## Layer 2 — Graph building (`src/graph/builder.py`)

**File:** [`builder.py`](../src/graph/builder.py)

### Multi-relational graph assembly

`build_graph()` produces an undirected weighted `nx.Graph`. Two things happen:

1. **Recorded edges.** Each relationship row becomes an edge. The edge **weight** is
   *evidentiary weight* = tie strength × relation reliability, where reliability comes from
   `RELATION_WEIGHTS` (`Co-Accused` 1.00 → `Shelter Provider` 0.45). A strong financial tie
   therefore outweighs a weak family mention.
2. **Inferred co-accused edges.** If *k* people appear on one FIR, that is *k choose 2*
   links no analyst entered by hand — usually what makes a sparse dataset analysable. Two
   guards keep this honest:
   - Cases with more than `max_case_size` (12) accused are skipped — a 50-person round-up
     would inject a meaningless 1,225-edge clique.
   - Co-accused strength *rises with case severity* and *falls as the group grows*: two
     people on a murder FIR is a stronger signal than eight on a raid. Corroboration across
     multiple cases strengthens an existing tie rather than duplicating it.

At the default seed the sample network is **595 recorded + 405 inferred = 1,000 edges** —
co-accused inference nearly doubles the analysable structure.

### The 24-feature node matrix

`compute_node_features()` builds one row of **24 numeric features** per actor
(`FEATURE_COLUMNS`), spanning five families:

- **Structure** — degree, weighted degree, avg/max edge strength, log & avg interactions
- **Relation composition** — relation diversity, and the share of links that are financial /
  communication / co-accused / kinship / handler
- **Evidence** — number of distinct evidence sources corroborating the actor's links
- **Record** — prior offences, offence-type count, alias count, age, case count, max & mean
  case severity, dormancy years
- **Neighbourhood** — mean neighbour priors, mean neighbour degree, geographic spread

The *same matrix* feeds both the GNN and the Isolation Forest. That is intentional: an
anomaly found by the statistical model is describable in the same vocabulary as one found
by the neural model, so the explanation layer can name the two features (`+2.4σ`) that
drove a flag.

### Structural simulations

`fragmentation_impact()` answers *"who do we arrest first?"* by simulating node removal and
measuring the change in component count and largest-component share. `ego_network()` powers
the drill-down views. Both are O(V+E) and run only for shortlisted actors.

---

## Layer 3 — AI analysis (`src/analysis/`)

### 3a. Centrality suite (`centrality.py`)

Ten measures, because no single one identifies a "key player":

| Measure | What it captures | Why it matters operationally |
|---|---|---|
| degree / weighted degree | raw & evidence-weighted reach | easily inflated by chatter |
| **betweenness** | control of flow along shortest paths | finds *brokers* whose degree is low — the arrests that fragment a network |
| closeness | speed of reach to the whole network | per-component, scaled by component share |
| eigenvector | importance from being connected to important actors | detects lieutenants |
| PageRank | random-walk influence, robust on weighted graphs | primary influence signal |
| k-core | depth of embedding in the densest subgraph | separates core from periphery |
| clustering | local triangle density | flags tight cells |

Two subtleties that are easy to get wrong and are handled explicitly:

- **Weight-as-distance inversion.** NetworkX treats edge `weight` as a *cost* in shortest-path
  algorithms, but NETRA's weight is *affinity* — a strong tie is a *short* hop. Before
  computing betweenness/closeness the code writes an inverted `distance = 1/weight` attribute.
  Skip this and the brokers come out wrong.
- **Disconnected closeness.** Closeness is computed per connected component and scaled by the
  component's share of the graph, so a 3-node clique can't outrank a well-placed actor in the
  main organisation.

**Performance guard:** above `PERF["exact_centrality_max_nodes"]` (1,500) betweenness switches
to *k*-pivot sampling (`k=320`, seeded) to hold interaction latency under a second.

### 3b. Community detection (`communities.py`)

`detect_communities()` runs **Louvain modularity** (weighted, seeded), falling back to
greedy modularity then connected components so the pipeline never dies on an odd graph.
Modules are relabelled so the largest gets id 0 — stable colours in the UI. On the sample
network Louvain recovers the planted syndicates cleanly (**modularity Q ≈ 0.77**, well above
the ~0.3 that signals meaningful structure).

On top of the partition:

- `community_profiles()` scores each module's **threat** as a weighted blend of cohesion
  (density), offence severity, geographic reach, record depth, insulation and size (weights
  sum to 1.0, in `config.py` — auditable).
- `bridge_analysis()` finds actors linking multiple modules, ranked by how much of their
  connectivity is *external*.
- `critical_edges()` ranks inter-module links by edge betweenness and flags true cut-edges
  (bridges) — the surveillance targets that reveal how two organisations coordinate.

### 3c. Light Graph Neural Network (`gnn.py`)

A **2-layer GraphSAGE-style encoder** implemented from scratch in NumPy with analytic
backpropagation and an Adam optimiser. No deep-learning framework.

**Encoder** (per layer ℓ, node *v*):
```
a⁽ˡ⁾ᵥ = Σ_{u∈N(v)} ŵ_vu · h⁽ˡ⁾ᵤ            (degree-normalised weighted mean aggregate)
h⁽ˡ⁺¹⁾ᵥ = tanh( W⁽ˡ⁾ · [ h⁽ˡ⁾ᵥ ‖ a⁽ˡ⁾ᵥ ] + b⁽ˡ⁾ )   (concatenate self ‖ neighbourhood)
```

**Objective — unsupervised link prediction.** The model learns to score observed edges above
degree-matched negative samples with a logistic loss:
```
L = −Σ_{(u,v)∈E} log σ(zᵤ·z_v) − Σ_{(u,v')∈E⁻} log(1 − σ(zᵤ·z_v'))
```
Nothing is labelled, so nothing needs a ground truth an investigator does not have.

**Key implementation details:**
- **Degree-proportional negative sampling** (`deg^0.75`, the node2vec/word2vec exponent):
  hard negatives are hubs, so the model can't cheat by scoring "any two popular nodes are
  linked". A precomputed CDF + `searchsorted` alias sampler avoids rebuilding the
  distribution every epoch.
- **Honest evaluation.** 10% of edges are held out; reported ROC-AUC is on that held-out set
  (`≈ 0.97` on the sample), not a training score.
- **200 epochs** is the measured sweet spot in `GNN_CONFIG`: AUC peaks near 0.96 and drifts
  down past ~250 as the encoder starts memorising individual edges.
- **Vectorised backprop.** The scatter-add gradient uses per-column `np.bincount` (compiled)
  instead of `np.add.at` (Python-level loop) — roughly an order of magnitude faster at these
  shapes.

**Three products consumed downstream:**
1. `embeddings` — a 32-dim role vector per actor. Structurally similar actors cluster even
   if they never met; a PCA-to-2D projection drives the embedding map, and cosine
   nearest-neighbour search (`similar_actors()`) answers *"who behaves like this kingpin — who
   takes over if we arrest them?"*
2. `residuals` — per-node link-reconstruction error, normalised 0–1. High residual = the
   learnt grammar of the network cannot explain this actor's connections = structurally
   anomalous. This is the GNN's 30% vote in the anomaly ensemble. Pure isolates are pinned
   low (0.25) so an absence of links doesn't masquerade as a structural anomaly.
3. `embedding_kmeans()` — behavioural archetype clusters, distinct from Louvain: Louvain
   groups actors who are *connected*, this groups actors who *play the same role*.

### 3d. Key player detection (`key_players.py`)

The naive answer to "who runs this network?" — *most contacts* — is usually wrong: in a
compartmentalised organisation the highest-degree actor is a dispatcher, and the kingpin
keeps a small, high-value address book. So NETRA does three things:

1. **Influence Index (0–100).** A weighted blend of six *percentile-ranked* centrality
   measures (`INFLUENCE_WEIGHTS`: PageRank 0.26, betweenness 0.24, weighted degree 0.18,
   eigenvector 0.16, closeness 0.10, k-core 0.06). Percentile ranks, not raw min-max, so one
   runaway hub can't flatten everyone else's score. Rescaled so the strongest actor *in this
   dataset* anchors at 100 — analysts compare within a case file.
2. **Operational role inference.** Reads each actor's centrality *profile* (not magnitude) in
   a fixed decision order encoding investigative doctrine: Kingpin → Financier → Broker →
   Lieutenant → Enforcer → Foot Soldier → Peripheral. E.g. a *Broker* is defined by
   betweenness ranking ≥ 0.22 above its degree ranking — controlling flow far beyond what its
   contact count suggests.
3. **Disruption ranking.** Runs the fragmentation simulation for the top actors *by influence
   and by betweenness* (brokers have low influence but high disruption), then ranks by a
   `disruption_score` = 45% influence + 30% main-cluster drop + 25% new fragments, normalised
   against the strongest observed impact. `arrest_priority()` excludes actors already in
   custody or deceased — leaving them in the queue is the fastest way to lose an analyst's
   trust.

### 3e. Hybrid anomaly detection (`anomaly.py`)

Three detectors vote (`ANOMALY_WEIGHTS`, sum to 1.0):

| Component | Weight | Catches |
|---|---|---|
| GNN reconstruction residual | 30% | actors whose link pattern defies the learnt grammar |
| Isolation Forest (300 trees, 24 features) | 30% | multivariate outliers — odd *combinations*, not odd single values |
| Expert rule engine (8 rules) | 40% | doctrine-driven flags, each with a written justification |

**The eight rules (R1–R8):**

| # | Rule | Signal |
|---|---|---|
| R1 | Cross-Module Broker | high betweenness percentile, low degree percentile, links into another module |
| R2 | Multi-Module Connector | active links into ≥3 distinct modules |
| R3 | Suspected Financial Mule | ≥60% financial links, ≤1 prior offence, degree ≥3 |
| R4 | Insulated High-Severity Actor | very serious offences, ≤2 contacts |
| R5 | Identity Evasion | ≥3 recorded aliases |
| R6 | Reactivation After Dormancy | long history, then a burst of fresh contacts that dominate the actor's link set |
| R7 | Hierarchy Inversion | structurally senior to contacts with much heavier records |
| R8 | Structural Gatekeeper | a cut-vertex whose removal severs a cell of ≥3 actors |

R8 uses `nx.articulation_points` (O(V+E)) and then measures the *severed cell size* per
cut-vertex, keeping only those that gate a meaningful group — most cut-vertices are trivial
pendants.

**Score composition** encodes two more design decisions:
- **Rule saturation.** The rule score is a *saturating* sum (`RULE_SATURATION = 3.0`, the
  realistic max co-firing mass), so two strong rules outrank four weak ones and no actor maxes
  out on volume. Many rules are mutually exclusive (you can't be both isolated and a hub), so
  the denominator is *not* the theoretical sum of all eight.
- **Connectivity gate.** For actors that trigger *no* rule, the ML signals are damped in
  proportion to how embedded the actor is (`ANOMALY_MODEL_GATE_FLOOR = 0.35`). A lone node
  being a statistical outlier is expected, not intelligence. Rule-flagged actors are never
  gated — a rule is domain justification enough for the full model contribution.

The final 0–100 score maps to Critical / High / Medium / Low bands (`SUSPICION_BANDS`).
`suspicious_clusters()` then rolls anomalies up to the module level, scoring each module by
cohesion + flagged-actor density + peak individual suspicion.

**Validation on the sample:** the top-3 most-suspicious actors at seed 42 are *exactly* three
of the planted anomalies (dormancy spike, money mule, insulated offender), each labelled with
the correct rule — detection is blind to the ground-truth key.

### 3f. Entity extraction (`src/ingestion/entities.py`)

PS 26189 asks for entity extraction from *unstructured* data, so this stage reads the document
corpus rather than the CSV rosters. `extract_corpus()` returns an `ExtractionResult` carrying
three frames: **entities** (deduplicated, one row per real-world thing), **mentions** (one row
per occurrence, with its character span) and **links** (entity co-occurrence within a document).

| Decision | Why |
|---|---|
| Pattern + gazetteer, not a transformer NER | The high-value entities here are *structured* identifiers — MSISDNs, vehicle registrations, IMEIs, IFSC-bearing accounts, crypto addresses. Regex plus validation beats a learnt model on exactly these, is auditable line by line, runs in ~50 ms, and avoids a 500 MB dependency that would break free-tier deployment. |
| Twelve UPPERCASE types (`ENTITY_TYPES`) | One closed vocabulary shared by the extractor, the UI and the export, so a type can never drift between layers. |
| Span arbitration by pattern order | Patterns are declared **ordered by specificity** and a mention is refused if it overlaps a span an earlier pattern already claimed. So a *labelled* identifier beats a length inference (`a/c 918020045512345` stays a `BANK_ACCOUNT` instead of becoming an `IMEI`), and `IMEI` runs before `PHONE` because a 15-digit run contains 10-digit substrings that look like mobile numbers. |
| Luhn check on account numbers | A syntactically valid but impossible account number is a false lead an officer would waste a day on. |
| Normalisation key per type | `+91 98765-43210` and `9876543210` collapse to one entity; formatting variance never inflates the count. |
| Every mention keeps its span and context | The UI renders the entity **highlighted inside its source sentence**. An extracted lead an investigator cannot trace back to a narrative is not evidence. |
| `mask_value()` on every sensitive type | Phone, bank account, IMEI, crypto wallet and e-mail/handle are partially redacted on screen — the extraction page can be screen-shared in a briefing. Enough of the tail survives for an officer to match against a known identifier, and the **unmasked value stays in the frame** for export under authorisation. |
| Content-derived entity ids | `E-<TYPE>-<sha1[:8]>` of type + normalised value, so the same entity gets the same id on every machine and across runs — a cited entity id stays valid. |

`front_business_candidates()` layers one derived signal on top: organisations that co-occur with
financial entities across *multiple* documents — the shape a laundering front makes in text.

### 3g. Women-Safety threat lens (`src/analysis/women_safety.py`)

The problem statement is owned by the **NCRB Women Safety Division**, so the network is scored a
second time against that mandate. This is deliberately a *content* score, independent of the
structural anomaly band: a division-critical offender can be structurally unremarkable, and
generic centrality walks straight past them.

`women_safety_cases()` first narrows the caseload to the division's mandate — and explicitly
**excludes commodity trafficking**, which shares the word "trafficking" but not the mandate.
`compute_wsti()` then scores each involved actor 0–100 from five weighted components
(`WSTI_WEIGHTS`):

| Component | Source | Note |
|---|---|---|
| `case_involvement` | division cases per actor, min-max scaled over the cohort | scaled *within the division cohort*, not the whole network — otherwise every score collapses toward zero |
| `severity` | peak offence severity / 5 | peak, not mean: one maximum-severity offence is not diluted by lesser files |
| `minor_victim` | binary | the **statutory aggravator**, carried as a full weighted term rather than a soft bonus |
| `network_reach` | PageRank when centralities are available, degree otherwise | recruitment capability |
| `recidivism` | prior offences, min-max scaled | repeat offending |

Bands are Low / Moderate / High / **Severe** (`WS_PARAMS["high_threat_wsti"]` sets the top cut).
Cyber-enablement is deliberately *not* a weighted term — it changes the **evidence-handling
instruction**, not the threat level, so it surfaces as the flag "digital evidence preservation
required". Every score carries its `flags`: the plain-language reasons that produced it.

`trafficking_corridors()` adds the geographic lens topology cannot provide: take the subgraph
induced by division actors, walk its connected components, and report any component whose cases
span ≥3 cities with ≥3 actors. Three contract details matter for evidentiary use:

- **Corridor ids are the risk rank, not the enumeration order.** `connected_components()` order
  is not stable across processes, so the same corridor was once labelled `CORR-02` on one run and
  `CORR-01` on the next. Ids are now assigned after ranking by risk (route as tie-break), making
  `CORR-01` always the corridor most warranting an operation — and safe to quote in a dossier
  that will be re-derived later.
- **Truncated rosters say so.** `actor_ids` caps at 12 and `case_ids` at 10 for readability, but
  each appends `… +N more`, because they sit in the same row as `n_actors` / `ws_cases` and a
  silent cap reads as the complete roster.
- **The overflow marker never re-enters the data.** `ws_case_ids` is both displayed *and*
  re-parsed here, so `_split_ids()` drops the marker instead of mistaking it for a case id.

The lens feeds its own alert stream, so a minor-victim escalation reaches the queue even when the
actor is invisible to every structural detector.

---

## Layer 4 — Dashboard (`src/ui/` + `src/visualization/`)

[`app.py`](../app.py) is deliberately thin: it wires sidebar controls to the cached pipeline
and dispatches to five page modules. The heavy lifting stays in `src/` so the pipeline is
reusable and testable.

- **Caching.** `run_pipeline`, sample generation, and upload ingestion are each wrapped in
  `st.cache_data`. The first analysis runs the full pipeline; every subsequent tab switch or
  widget interaction is a cache hit (< 0.1 s). The PDF is separately cached by a cheap
  signature so repeated downloads don't rebuild it.
- **Never-empty default.** On first load the sample dataset auto-runs, so a judge sees a
  populated dashboard immediately — no "click here first" screen unless they choose upload.
- **Theme.** A dark "intelligence console" palette is injected as CSS
  ([`theme.py`](../src/ui/theme.py)); the whole palette lives in `config.py`. Table heat-maps
  are rendered with a pandas `Styler.apply` that returns rgba CSS strings — **no matplotlib
  dependency**, keeping the stack light.
- **Visualization.** [`network_viz.py`](../src/visualization/network_viz.py) builds the Plotly
  graph canvas and ego networks; [`charts.py`](../src/visualization/charts.py) builds the
  analytical charts. All Plotly — interactive, zero image-encoding round-trips.

The eight tabs — Dashboard, Data & Sources, Entity Extraction, Network Graph, Key Players,
Anomalies, Women Safety, Reports & Alerts — are declared in one place, the `PAGES` table in
[`app.py`](../app.py), and each renders inside its own error boundary so a failure on one screen
cannot take the app down. They are documented in the
[README dashboard tour](../README.md#-dashboard-tour) and the
[presentation script](PRESENTATION.md).

---

## Layer 5 — Reporting (`src/reporting/`)

**Files:** [`alerts.py`](../src/reporting/alerts.py), [`report.py`](../src/reporting/report.py)

`build_alerts()` synthesises a ranked, de-duplicated alert queue from five streams —
high-suspicion individuals, arrest-priority targets, suspicious clusters, critical
inter-module links, and **Women Safety** escalations from the WSTI lens. An **alert is a score plus a reason plus a recommended action plus the
evidence** an officer would cite. The recommended action is mapped from the *dominant rule*
(a gatekeeper → "arrest isolates that cell"; a mule → "refer to financial-intelligence unit").
Synthesising alerts here — not in the UI — guarantees the on-screen cards and the exported PDF
never disagree.

Three export formats, one source of truth (`AnalysisResult`):
- **Markdown brief** — self-contained, diff-able, and the fallback when ReportLab is absent.
- **PDF dossier** — a formatted ReportLab document (cover, severity strip, key-player and
  arrest tables, alert queue, methodology). Returns `None` gracefully if ReportLab isn't
  installed, so the app degrades to Markdown instead of crashing.
- **CSV/ZIP bundle** — every analysis frame plus the brief, for import into other tooling.
  List-typed cells (rules, reasons) are pipe-joined for CSV compatibility.

---

## Data & control flow summary

```
CSV / sample ─► load_dataset ─► (persons, relationships, crimes, documents, ValidationReport)
                                          │                      │
                                          │                      ▼
                                          │            extract_entities ─► ExtractionResult
                                          │              (12 entity types, NLP layer)
                                          ▼
                              build_graph ─► nx.Graph  ─►  compute_node_features ─► 24-feature matrix
                                          │                                              │
        ┌───────────────┬─────────────────┼──────────────────┬───────────────┐          │
        ▼               ▼                 ▼                  ▼               ▼          │
   centralities    communities        train_gnn        key_players      apply_rules ◄──┤
        │               │                 │                  │               │          │
        └───────────────┴────────┬────────┴──────────────────┴───────────────┘          │
                                 ▼                                                       │
                          detect_anomalies ◄──────── isolation_forest_scores ◄───────────┘
                                 │
                                 ├──► score_women_safety ─► WSTI + trafficking corridors
                                 ▼
                          AnalysisResult ──► dashboard tabs (Layer 4)
                                          ├─► build_alerts ──► PDF / MD / CSV (Layer 5)
                                          └─► db.save_run ──► SQLite case store + audit trail
```

`load_dataset` returns a **5-tuple** — the fourth element is the unstructured intelligence
corpus (empty frame when no `documents.csv` was supplied), which is what the NLP layer
consumes. Everything from `build_graph` onward is orchestrated by `run_pipeline()` and returned
as one `AnalysisResult`; `run_pipeline` also accepts a `progress_cb(fraction, message)` so a
CLI or notebook caller can report each of the nine stages live. The Streamlit app deliberately
does *not* wire it: `st.cache_data` replays the element writes a cached body made, and a write
aimed at a bar created outside that body raises `CacheReplayClosureError` on the first cache
hit. The app shows a spinner instead and reports the per-stage timings recorded in
`result.timings` afterwards. See [`pipeline.py`](../src/pipeline.py) for the exact call order.

---

## Testing

The [test suite](../tests) (`pytest tests/ -v`, 103 tests) locks the invariants this document
relies on: schema validation and stub creation, graph construction (node/edge counts, weight
bounds, co-accused inference), the 24-column feature matrix (no NaN/inf), centrality bounds
(PageRank sums to 1, normalised measures in [0,1]), community coverage and modularity > 0.2,
GNN convergence (AUC > 0.7, loss decreases, embedding shape), full-pipeline determinism
(identical embeddings and suspicion scores across two runs), anomaly explainability (every
flagged actor carries reasons), role membership, and arrest-priority exclusion of
incarcerated actors. A single small synthetic network is built once per session and shared
across tests to keep the suite fast.
