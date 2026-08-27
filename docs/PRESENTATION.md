# NETRA — SIH 2026 Presentation & Demo Script

> A **7-minute** judge-ready walkthrough for Smart India Hackathon 2026.
> Companion to the [README](../README.md) and [Architecture notes](ARCHITECTURE.md).
> Every number in this script is real, produced by the default configuration — run it once
> yourself before the demo so the screen matches the words.

**NETRA** — *Network Evaluation & Threat Recognition Analytics*.

| | |
|---|---|
| **Problem Statement** | **26189** — AI-Powered Criminal Network Analysis System |
| **Organisation** | Ministry of Home Affairs |
| **Department** | National Crime Records Bureau (NCRB) · **Women Safety Division** |
| **Category / Theme** | Software · Blockchain & Cybersecurity |

> **Say the department out loud in your opening.** Most teams will present a generic
> link-analysis tool. The PS is owned by the **Women Safety Division**, and NETRA has a
> dedicated screen scoring the network against that mandate. That is your differentiator —
> budget real time for it (⏱ 5:00–5:45 below).

---

## Before you walk in (2-minute setup)

```bash
streamlit run app.py
```

- The app **auto-loads the sample dataset and runs the full analysis on first load** — the
  dashboard is already populated when the browser opens. No "click to start" screen.
- Leave the sidebar at its **defaults**: source *Sample*, 220 persons, 6 syndicates, seed
  **42**, GNN on at 200 epochs. Every figure below assumes exactly this.
- Do one warm-up run so caching is primed — after the first analysis, every tab switch is
  instant (< 0.1 s).
- In the sidebar's **Presentation** expander, turn on **📌 Show PS-26189 traceability panel**.
  It renders a requirement→implementation matrix on the Dashboard, which answers "does this
  actually address the problem statement?" before a judge has to ask.
- Have the [Architecture notes](ARCHITECTURE.md) open in a second tab in case a judge asks
  "how does the GNN actually work".

> **On the numbers below:** the network *structure*, scores, and detections are identical on
> every machine (everything is seeded). The procedurally generated *names* are also
> deterministic at seed 42 but would change at another seed — so lead with the **planted
> pattern and the ID** (e.g. "the dormancy-spike actor, P0234"), and treat specific names as
> "at the default seed". This is also your strongest proof point: **you know the ground truth
> and the blind analysis recovers it.**

---

## The 60-second pitch (open with this)

> "Investigators drowning in FIRs, call records and surveillance notes can see individual
> crimes but not the *organisation* behind them. NETRA ingests all seven data sources the
> problem statement names, pulls the entities out of the unstructured text, reconstructs the
> network, and answers the four questions that actually drive an operation: **Who runs it?
> Who's hiding in it? Who threatens women and children? And who do we arrest first to break
> it?** It's built entirely on free, open tools — no cloud, no paid API — and every single
> flag it raises comes with a written justification an officer could put in a surveillance
> request. Let me show you."

---

## The demo — minute by minute

### ⏱ 0:00–0:45 — Dashboard: the network at a glance

**Land on the 📊 Dashboard tab.**

- Point to the headline metrics: **234 actors, 1,000 links, 7 detected modules, 105 case
  files.**
- The key line: *"Of those 1,000 links, only 595 were recorded by analysts — NETRA **inferred
  the other 405** from co-accused relationships in the case files. That's structure no human
  entered by hand."*
- Point to **modularity Q = 0.76**: *"That's a mathematical score for how cleanly the network
  splits into distinct organisations. Above ~0.3 means real structure; we're at 0.76 — these
  are genuinely separate syndicates, not one blob."*
- A quietly strong line: *"I generated this dataset with **six** syndicates. NETRA, working only
  from the link structure with no labels, found **seven** modules — and the seventh turns out to
  be a tightly-bound eight-person cell I planted separately. I'll come back to that."* (Payoff
  lands on the Anomalies tab at ⏱ 4:00.)
  Point at it while you say it: **Module 7 is the white cluster**, sitting just off the green
  Module 4 near the top of the graph. Every module gets a distinct colour, so the odd one out is
  the one you can name from across the room.
- Scroll to the **PS-26189 traceability matrix**: *"Every requirement in the problem
  statement, the module that implements it, and the screen it appears on. You can audit our
  coverage in ten seconds."*

### ⏱ 0:45–1:30 — Data & Sources + Entity Extraction: the seven sources, and the NLP

**Switch to 🗂️ Data & Sources**, then **🔎 Entity Extraction**. *Don't skip this — "extract
entities from unstructured data" is capability #2 in the PS and most teams will only do CSVs.*

- On **Data & Sources**: *"The PS names seven data sources. This panel maps each one to what
  was actually loaded — structured rosters and case files, plus **15 unstructured intelligence
  documents**: FIR narratives, surveillance reports, social-media intelligence and
  intelligence-agency reports."* Mention the **CDR panel** on the same screen: call records
  become weighted communication edges, with burner-handset detection and tower co-location.
- On **Entity Extraction**: *"From those 15 documents NETRA extracted **126 distinct entities
  across all 12 types** — 154 mentions, 249 co-occurrence links. People, phone numbers,
  vehicle registrations, locations, organisations, bank accounts, crypto wallets, IMEIs."*
- Now the credibility move — click one entity to expand it: *"And it is not a black box list.
  Every extracted entity is shown **highlighted inside the sentence that produced it**, with
  its document, date and case ID. An officer verifies the lead against the source narrative
  instead of trusting our model."*
- If you have a spare beat, point at **front-business candidates**: organisations
  co-occurring with financial entities across multiple documents.

### ⏱ 1:30–2:45 — Network Graph: see the organisations

**Switch to 🕸️ Network Graph.**

- The graph renders coloured by module — the syndicates are visually distinct clusters.
- *"Node size is influence, colour is which organisation they belong to. Notice these thin
  bridges between clusters"* — point to inter-module links — *"those are the people who
  connect two separate crime groups. In counter-network terms, those are gold."*
- Click into an **ego network** of one hub to show the drill-down: *"Every actor is one click
  from their local neighbourhood, their full record, and their **structural twins** — the
  actors whose learnt fingerprint most resembles theirs."*

> **Talking point — the AI is doing real work here:** *"Behind this view, a graph neural
> network has learned a 32-dimensional fingerprint of every actor's role — purely from the
> structure, with no labels. It scored **0.97 on held-out link prediction**, which means it
> genuinely learned how this network is wired, not just memorised it."*

> **Keep the twin panel in your pocket.** If a judge asks *"what happens after you arrest the
> #1 target?"*, come back to this tab: cosine nearest-neighbour search in the embedding space,
> excluding people already adjacent to them, returns exactly the actors positioned to inherit
> the role. That's a successor watchlist, generated automatically.

### ⏱ 2:45–4:00 — Key Players: who runs it, who do we arrest first

**Switch to 👑 Key Players.** *This is your strongest 75 seconds — spend them here.*

- *"Naive analysis says the boss is whoever has the most contacts. That's wrong — in a smart
  organisation the kingpin keeps a small, trusted circle and the loudest person is a
  dispatcher."*
- Point to the **role column**: NETRA infers Kingpin / Lieutenant / Broker / Financier /
  Enforcer / Foot Soldier from each actor's *centrality profile*, not their contact count. At
  seed 42: **18 Brokers, 34 Lieutenants, 10 Enforcers, 2 Financiers.**
- Now the payoff — the **Arrest Priority** list, ranked by **disruption score** (a simulation
  of what happens to the network when you remove each person):
  - **#1 target: a Broker, disruption 84.7** (P0134 — *Naveen Malhotra* at seed 42).
  - **#2 target: P0224, a Broker, disruption 78.7.**
- Deliver the ground-truth reveal: *"Here's the thing about P0224. When I generated this
  dataset, I deliberately planted a single 'gateway' actor — one person who is the **only**
  link holding a seven-member cell to the rest of the network. I never told the analysis that.
  It independently found that removing P0224 shatters the network, and ranked them the
  **#2 arrest priority**. That's the system identifying a structural choke point on its own."*
- If asked "what happens after the arrest?" — flip back to **Network Graph** and use the
  **structural-twin** panel: *"Cosine similarity in the learnt embedding space finds every actor
  who behaves structurally like the one you just removed. That's your successor watchlist."*

### ⏱ 4:00–5:00 — Anomalies: who's hiding in the network

**Switch to 🚨 Anomalies.** *This is the "wow" moment — the blind detection of planted threats.*

- *"This tab hunts for people who don't fit — using three independent detectors that vote: the
  neural network, a statistical outlier model, and an eight-rule expert engine. Watch the top
  three."*

| Rank | Suspicion | Band | Auto-detected flags | Ground truth (planted) |
|---|---|---|---|---|
| 1 | **80.2** | Critical | Cross-Module Broker · Reactivation After Dormancy | P0234 — **dormancy spike** |
| 2 | **79.3** | High | Multi-Module Connector · Suspected Financial Mule | P0232 — **money mule** |
| 3 | **78.1** | High | Insulated High-Severity Actor · Identity Evasion Pattern | P0233 — **isolated high-severity** |

- The reveal: *"When I built this dataset I planted exactly five kinds of hidden threat. The
  three highest-scoring anomalies the system found — completely blind, it never saw my
  answer key — are three of those five planted actors, and it labelled each one with the
  **correct** reason. The money mule is flagged as a money mule. The reactivated dormant actor
  is flagged for reactivation after dormancy."*
- Note the discipline of the output: **11 actors flagged out of 234** (1 Critical, 3 High,
  7 Medium). *"A tool that flags a third of the network is a tool nobody uses."*
- Click one flagged actor to expand the **reasons**: *"And it's never a black box. Every score
  breaks down into the exact rules and features that fired — this is what an officer needs to
  justify acting on it."*
- Scroll to **Suspicious clusters** and close the loop you opened on the Dashboard: *"Remember
  the seventh module NETRA found that I never planted as a syndicate? There it is — **eight
  actors, confined to a single state, signature patterns 'Cross-Module Broker' and 'Structural
  Gatekeeper', lead actor P0224**. The community detector isolated my planted cell, and the
  pattern engine named its gatekeeper — the same person the arrest ranking put at #2. Three
  independent parts of the system converging on one actor."*

> **If a judge is skeptical it's rigged:** *"Change the seed in the sidebar and rerun — the
> names and the exact network change completely, but the system still finds the planted
> patterns. The detection is structural, not memorised."* (Only do this live if you have a
> spare minute — it triggers a full recompute, ~5 seconds.)

### ⏱ 5:00–5:45 — Women Safety: the owning department's lens

**Switch to 🚺 Women Safety.** *Do not cut this section. This is what makes NETRA an NCRB
Women Safety Division tool rather than a generic graph demo.*

- Open with why the screen exists: *"This PS is owned by the Women Safety Division, so we
  re-score the entire network against that mandate. A division-critical offender can be
  structurally invisible — low centrality, no anomaly flag — and generic analysis will walk
  right past them."*
- The headline: **17 of 105 cases fall in the division's mandate — 16% of the caseload — and
  they involve 41 actors.** Of those: **5 at Severe threat, 13 High or worse, 6 linked to
  minor victims, 14 repeat offenders.** Dominant offence: **Human Trafficking.**
- Explain the **WSTI** in one sentence: *"The Women-Safety Threat Index blends five weighted
  components — case involvement, peak severity, a **statutory minor-victim aggravator**,
  network reach and recidivism — so it's a content score, deliberately independent of the
  structural anomaly band."* Top actor at seed 42: **P0136, WSTI 77, Severe, across 4 cities.**
- Then the finding that lands: **trafficking corridors.** *"We take only the division's actors,
  walk the connected components, and report any group whose cases span three or more cities.
  Corridor CORR-01 is **17 connected actors across 8 cities** — Amritsar, Coimbatore, Delhi,
  Jaipur, Kanpur, Mangaluru, Siliguri, Vijayawada — peak severity 5, **with minor victims**, and
  it scores a full 100 on corridor risk. That is not a coincidence, that's a route, and it's the
  case for an inter-state operation."*
- Close the loop: *"Eight of our alerts come from this lens alone — including two Critical
  minor-victim escalations that no other screen surfaces."*

### ⏱ 5:45–6:30 — Reports & Alerts: from analysis to action

**Switch to 📕 Reports & Alerts.**

- *"An analysis a judge can't hand to a field officer is a toy. NETRA closes that loop."*
- Point to the alert queue: **28 prioritised alerts — 8 Critical, 8 High, 12 Medium** — drawn
  from five streams (**11 Key Player, 8 Women Safety, 4 Anomaly, 4 Cluster, 1 critical Link**),
  each with a subject, a severity, the evidence, and a **recommended action** mapped to the
  finding (a gateway actor → 'target for arrest to fragment the cell'; a mule → 'refer to
  financial-intelligence unit').
- Click **📕 PDF dossier**: *"One click produces a formatted intelligence brief — cover page,
  executive summary, key-player and arrest tables, the alert queue, and the methodology."*
  (The two buttons beside it export a **📄 Markdown brief** and a **🗜 CSV data bundle** for
  feeding other tools.)
- Worth one sentence: *"Every saved run and every consequential action goes into an SQLite
  case store with an **append-only audit trail** — so a finding is re-derivable months later
  with the exact parameters that produced it."*

### ⏱ 6:30–7:00 — Close

> "So in about five seconds of compute, on a laptop, with free tools: NETRA turned four raw
> CSVs and fifteen intelligence narratives into a mapped criminal network, extracted the
> entities hidden in the text, named the leadership, ranked arrests by how much damage each
> one does to the organisation, blindly surfaced hidden operatives, and flagged an eight-city
> trafficking corridor for the Women Safety Division — each with a written justification.
> Everything is reproducible, everything is explainable, and it deploys free. That's NETRA:
> **Network Evaluation and Threat Recognition Analytics**, built for problem statement 26189,
> Ministry of Home Affairs. Thank you — I'm happy to take questions."

---

## Reference: verified demo numbers (seed 42, defaults)

Structural facts — identical on any machine:

| Metric | Value |
|---|---|
| Actors (nodes) | 234 |
| Links (edges) | 1,000 (595 recorded + 405 inferred) |
| Graph density | 0.037 |
| Connected components | 1 |
| Detected modules | 7 (from 6 planted syndicates + 1 planted cell) |
| Modularity Q | 0.764 |
| Suspicious clusters ranked | 7 (top 6 shown on the Anomalies tab) |
| Case files | 105 |
| Intelligence documents | 15 (4 FIR · 4 surveillance · 4 social-media · 3 agency) |
| Entities extracted | 126 across 12 types (154 mentions, 249 links, mean confidence 0.85) |
| GNN held-out AUC | 0.966 (200 epochs) |
| Actors flagged anomalous | 11 of 234 (1 Critical / 3 High / 7 Medium) |
| Women-safety cases | 17 of 105 (16.2%), 41 actors, 2 corridors |
| Alerts | 28 (8 Critical / 8 High / 12 Medium) |
| Warm re-run | ~5.5 s total · GNN training is 3.5 s of it |

**Planted ground truth (the answer key the analysis never sees):**

| Pattern | Planted IDs | Where it surfaces |
|---|---|---|
| Cross-module brokers | P0221, P0222, P0223 | high-betweenness brokers |
| Gateway chokepoint | P0224 | **#2 arrest priority**, disruption 78.7 · also **lead actor of Module 7**, the 8-actor cluster flagged "Structural Gatekeeper" |
| Money mule | P0232 | **#2 anomaly**, susp 79.3, flagged "Suspected Financial Mule" |
| Isolated high-severity | P0233 | **#3 anomaly**, susp 78.1 |
| Dormancy spike | P0234 | **#1 anomaly**, susp 80.2, flagged "Reactivation After Dormancy" |

**Women Safety lens (seed 42):**

| Metric | Value |
|---|---|
| Top WSTI actor | P0136 — 77.1, **Severe**, 4 cities |
| Severe band | 5 actors |
| High or worse | 13 actors |
| Minor-victim linked | 6 actors |
| Repeat offenders | 14 actors |
| Corridors | **CORR-01** — 17 actors / 8 cities / peak severity 5 / minor victims present / risk 100.0 · **CORR-02** — 12 actors / 3 cities / risk 57.5 |

> Corridor ids are the **risk rank**, so CORR-01 is always the corridor most warranting an
> operation — the label is reproducible across reruns and safe to quote in a dossier.

> If your screen disagrees with any of these, the sidebar is not at defaults. Reset to
> Sample / 220 / 6 / seed 42 / GNN on 200 epochs and rerun.

---

## Anticipated judge questions

**"Is this a real GNN or just centrality with a fancy name?"**
> Real. It's a 2-layer GraphSAGE encoder trained from scratch in NumPy — weighted neighbour
> aggregation, self-concatenation, tanh, trained by unsupervised link prediction with
> degree-proportional negative sampling and Adam. We hold out 10% of edges and report 0.97
> AUC on them. Centrality is a *separate* signal we combine with it; they're not the same
> thing. Details in [Architecture § 3c](ARCHITECTURE.md).

**"Where's the NLP? Extracting entities is in the problem statement."**
> The Entity Extraction tab. Pattern-plus-gazetteer extraction over unstructured intelligence
> text, tuned for Indian data — 12 entity types including vehicle registrations, IMEIs, bank
> accounts and crypto wallets, with Luhn validation on account numbers and span arbitration so
> overlapping claims resolve to one winner. We chose it over a transformer NER deliberately:
> for structured identifiers it is more accurate, it's auditable, it runs in 50 ms, and it
> doesn't add a 500 MB dependency and a GPU expectation to a free deployment.

**"Why should an officer trust a machine flag?"**
> Because it's never just a number. Every anomaly carries the specific rules and the two most
> deviant features that produced it, in plain language. The score is 40% explicit expert rules,
> so most flags trace to written investigative doctrine, not an opaque model. And every
> extracted entity is shown inside the sentence it came from.

**"Won't ML on a few hundred records be unreliable?"**
> That's exactly why we don't rely on one model. Three detectors vote — the neural network, an
> Isolation Forest, and an eight-rule expert engine — and we damp the ML signal for actors too
> sparsely connected to score reliably. The rules give us a credible floor even on small data.

**"What about scale — real datasets are huge."**
> The pipeline degrades gracefully: above 1,500 nodes betweenness switches to seeded sampling,
> the render canvas caps at 500 nodes, CDR ingestion has a row ceiling with an actionable
> message, and huge case files are capped before they inject meaningless cliques. It's CPU-only
> and caches aggressively, so interaction stays sub-second.

**"How is this specific to the Women Safety Division?"**
> The WSTI is a separate scoring pass over the division's own caseload, with a statutory
> minor-victim aggravator, and it feeds its own alert stream — 8 of our 28 alerts. The
> trafficking-corridor detector is geographic, not topological, so it finds routes that
> centrality cannot see. At seed 42 it surfaces an 8-city corridor with minor victims that no
> other screen flags.

**"How do we know the results aren't cherry-picked for this demo?"**
> Everything is seeded and reproducible — you get these exact numbers. And the sample data has
> a *planted* ground truth the analysis is blind to; the fact that the top detections match the
> plants is the proof it works, not a coincidence. Change the seed and it still finds them.

**"Is it free? Can MHA actually deploy this?"**
> Entirely free and open — Python, Streamlit, NetworkX, scikit-learn, NumPy, SQLite. No cloud,
> no licences, no external API. It deploys on Streamlit Community Cloud, Render or any
> `$PORT` platform at zero cost, and self-hosts as a Docker container on a VPS for a durable
> audit trail. It runs fully offline on a standard laptop, which matters for sensitive data
> that can't leave the building.

**"What's the data privacy story?"**
> All demo data is synthetic — no real individual is represented. In the sidebar there's a
> redaction toggle that masks identifiers on screen for briefings, extraction output masks
> phone and account numbers by default, and the store keeps an append-only audit trail of
> every consequential action. It's decision support with a human in the loop; the system never
> acts.

---

## Demo failure recovery

- **App won't start / import error:** `pip install -r requirements.txt`, then rerun. Python
  3.10+.
- **PDF button is greyed out and reads "📕 PDF (install reportlab)":** ReportLab isn't
  installed. Present the **📄 Markdown brief** instead — it carries the same content and is
  always available. Fix it before the demo with `pip install reportlab`, or just say the export
  layer degrades to Markdown rather than failing.
- **Numbers on screen differ from this script:** confirm the sidebar is at defaults (sample
  source, 220 / 6 / seed 42) and rerun. Any other seed changes names and structure.
- **A tab is slow:** it's the first (cold) computation; it's cached after. Do a warm-up run
  before presenting.
- **A tab shows an error card:** each screen renders inside its own error boundary, so one
  failure doesn't take down the app — move to the next tab and keep going. The card itself is
  honest engineering; say so.
- **The Women Safety tab looks empty:** the sample dataset always carries division cases at
  the defaults, so this means the sidebar filters are narrowed — reset them.
- **Judge wants to upload their own file:** switch source to *Upload* — see the
  [Data Schema](DATA_SCHEMA.md). If their headers differ, the Data Quality panel shows how
  NETRA auto-mapped them, which is itself a good feature to demo.
