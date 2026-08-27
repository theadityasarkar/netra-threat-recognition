# NETRA — Data Schema & Upload Contract

> Companion to the [README](../README.md) and [Architecture notes](ARCHITECTURE.md).
> This document is the exact contract for the four CSV files NETRA ingests. If your export
> conforms to this, the analysis runs end-to-end with no manual cleanup.

NETRA ingests **four CSV files**. Only one — persons — is strictly conceptual; in practice a
useful analysis needs at least the relationships **or** the crime-records file so there is
structure to build a graph from.

| File | Purpose | Required to run? |
|---|---|---|
| `persons.csv` | The roster — one row per person of interest | recommended (stubs are auto-created if omitted) |
| `relationships.csv` | Known ties between people | need this **or** crime records |
| `crime_records.csv` | FIR / case files, used to infer co-accused links | need this **or** relationships |
| `documents.csv` | Unstructured intelligence text for the NLP entity-extraction layer | optional (the Entity Extraction tab is empty without it) |

Working examples ship in [`data/samples/`](../data/samples/). The fastest way to understand
the format is to open those files alongside this document.

---

## Guiding principles (read this first)

Two behaviours make the loader forgiving; both are intentional and documented per field
below:

1. **Column auto-mapping.** You do **not** have to rename your columns. NETRA canonicalises
   headers (lower-cases, strips punctuation) and resolves common aliases — so `accused_id`,
   `Suspect ID`, and `uid` all become `person_id`. Every rename is reported back to you in the
   Data Quality panel. The accepted aliases are listed with each field.
2. **Evidence is never dropped.** A relationship pointing at a person who isn't in your roster
   does **not** get discarded — NETRA creates a *stub* record (`Unidentified P####`) so the
   link survives, and flags it as a data-quality warning. An incomplete roster degrades
   quality; it never loses intelligence.

Only two things are hard errors that stop the pipeline:

- `persons.csv` is provided but has no `person_id` column.
- `relationships.csv` is missing `source_id` or `target_id`.

Everything else is coerced, defaulted, or warned about.

---

## 1. `persons.csv` — the roster

One row per person. **14 recognised columns.** Only `person_id` is required.

| Column | Required | Type | Default if missing | Notes & accepted values |
|---|---|---|---|---|
| `person_id` | **Yes** | string | — | Unique key. Coerced to string, so `101` and `"101"` are the same actor. Must match the IDs used in the other two files. |
| `name` | no | string | `Unidentified P####` | Free text. |
| `age` | no | integer | blank | Clipped to **0–110**; out-of-range or non-numeric values are cleared. |
| `gender` | no | string | blank | Free text (e.g. M / F / Other). |
| `city` / `location` | no | string | blank | Used for the geographic-spread feature. |
| `state` | no | string | blank | Free text. |
| `occupation` | no | string | blank | Free text. |
| `prior_offences` | no | integer | `0` | Non-negative. Drives risk derivation and several anomaly rules. |
| `offence_types` | no | list | blank | Multi-value — separate with `;` `,` `\|` or `/` (e.g. `Narcotics; Extortion`). |
| `aliases` | no | list | blank | Multi-value, same separators. Alias **count** feeds the identity-evasion rule (R5). |
| `risk_category` | no | string | *derived* | `Low` / `Medium` / `High` / `Critical`. **If absent, NETRA derives it** from `prior_offences` via banding — it does not default everyone to Low. |
| `first_seen` | no | date | blank | Any common date format (parsed flexibly). |
| `last_active` | no | date | blank | Used with `first_seen` to compute **dormancy** (years inactive) for rule R6. |
| `status` | no | string | `Active` | Free text. The values `In Judicial Custody` and `Deceased` are **excluded from arrest-priority** — spell them exactly to get that behaviour. |

**`person_id` aliases:** `id`, `accused_id`, `suspect_id`, `uid`, `personid`, `person`
**`name` aliases:** `full_name`, `person_name`, `accused_name`
**`city` aliases:** `location`, `place`, `town`
**`prior_offences` aliases:** `priors`, `previous_offences`, `offence_count`

---

## 2. `relationships.csv` — known ties

One row per known tie between two people. **Undirected** — NETRA canonicalises `A→B` and
`B→A` into one link, keeping the strongest strength and summing interactions.
`source_id` and `target_id` are required.

| Column | Required | Type | Default if missing | Notes & accepted values |
|---|---|---|---|---|
| `source_id` | **Yes** | string | — | One endpoint. Must be (or become, via stub) a `person_id`. |
| `target_id` | **Yes** | string | — | Other endpoint. Self-loops (`source == target`) are dropped. |
| `relation_type` | no | string | `Associate` | Determines the tie's reliability weight — see the table below. Unknown types are accepted and treated at the default weight. |
| `strength` | no | float | `0.5` | Analyst confidence in the tie, **0.0–1.0**. Clipped into range. |
| `interactions` | no | integer | `1` | Count of observed contacts; feeds interaction-volume features. |
| `evidence_type` | no | string | blank | e.g. `Wiretap`, `Financial`, `Surveillance`. Distinct sources per actor = evidence-diversity feature. |
| `first_contact` | no | date | blank | Flexible date parse. |
| `last_contact` | no | date | blank | Flexible date parse. |
| `location` | no | string | blank | Free text. |

**`source_id` aliases:** `from`, `src`, `node1`, `source`, `person_a`, `a`
**`target_id` aliases:** `to`, `dst`, `node2`, `target`, `person_b`, `b`
**`strength` aliases:** `weight`, `confidence`, `tie_strength`, `score`
**`relation_type` aliases:** `type`, `relation`, `relationship`, `edge_type`

### Recognised relation types and their reliability weights

The graph edge weight is `strength × reliability`, so *what kind* of tie it is matters as much
as your confidence in it. Recognised types (from `RELATION_WEIGHTS` in
[`config.py`](../src/config.py)):

| Relation type | Reliability | | Relation type | Reliability |
|---|---|---|---|---|
| Co-Accused | 1.00 | | Handler | 0.80 |
| Financial | 0.90 | | Family | 0.75 |
| Gang Member | 0.90 | | Associate | 0.65 |
| Communication | 0.70 | | Shelter Provider | 0.45 |

Any type not in this table is still accepted — it simply uses the default reliability. Add or
retune types in `config.py`; nothing in the logic is hard-coded to these names.

---

## 3. `crime_records.csv` — case files

One row per case / FIR. NETRA uses these to **infer co-accused links**: every pair of people
on the same case becomes (or strengthens) an edge. `case_id` and `persons_involved` are
required.

| Column | Required | Type | Default if missing | Notes & accepted values |
|---|---|---|---|---|
| `case_id` | **Yes** | string | — | Unique case identifier. |
| `persons_involved` | **Yes** | list | — | The accused on this case — a **separator-delimited list of `person_id`s** (`P0007; P0012; P0034`). Separators: `;` `,` `\|` `/`. This is the field that drives co-accused inference. |
| `crime_type` | no | string | blank | e.g. `Narcotics`, `Arms`, `Cyber Fraud`. |
| `severity` | no | integer | `1` | **1–5** (5 = most serious). Clipped into range. Higher severity → stronger inferred co-accused ties. |
| `date` | no | date | blank | Flexible date parse. |
| `location` | no | string | blank | Free text. |
| `status` | no | string | blank | e.g. `Open`, `Chargesheeted`, `Convicted`. |
| `fir_number` | no | string | blank | Free text reference. |
| `description` | no | string | blank | Free text. |

**`case_id` aliases:** `case`, `fir_id`, `crime_id`, `record_id`
**`persons_involved` aliases:** `accused`, `persons`, `suspects`, `people`, `accused_ids`, `involved`
**`severity` aliases:** `seriousness`, `gravity`, `level`
**`crime_type` aliases:** `crime`, `offence`, `offense`, `category`

> **Co-accused inference guard:** cases with more than **12** accused are skipped for edge
> inference — a mass round-up would otherwise inject a huge meaningless clique. The case still
> appears in the data; it just doesn't manufacture 12-choose-2 links. See
> [Architecture § Layer 2](ARCHITECTURE.md#layer-2--graph-building-srcgraphbuilderpy).

---

## 4 · `documents.csv` — unstructured intelligence corpus (optional)

This is the file that feeds **Layer 1's NLP entity-extraction stage** and covers the four
free-text sources named in PS 26189 — FIR narratives, surveillance reports, social-media
intelligence and intelligence-agency reports. One row per document.

**Only `text` is required.** Every other column is provenance: supply what you have, and
NETRA fills the rest with blanks rather than rejecting the file.

| Column | Required | Type | Default if missing | Notes & accepted values |
|---|---|---|---|---|
| `text` | **Yes** | string | — | The document body — a narrative, an intercept, a surveillance note. Any length; extraction runs sentence-wise so long reports are fine. |
| `doc_id` | no | string | `D0001`… | Unique document reference. Auto-generated sequentially when absent. |
| `source_type` | no | string | blank | Provenance label, e.g. `FIR Narrative`, `Surveillance Report`, `Social Media Intel`, `Intel Agency Report`, `Financial Advisory`. Drives the source breakdown on the **Data & Sources** tab. |
| `title` | no | string | blank | Short human label shown above the highlighted text. |
| `date` | no | date | blank | Flexible date parse. |
| `city` | no | string | blank | Where the document originates — used when correlating extracted locations. |
| `case_id` | no | string | blank | Links the document back to a row in `crime_records.csv`, giving each extracted entity a case trace. |

**Loader leniency (deliberate, so a pasted export just works):**

- If there is **no `text` column**, NETRA adopts one: a single-column file has its only column
  renamed to `text`; otherwise the first column that isn't a known metadata name is adopted.
  Either way the substitution is reported in the Data Quality panel.
- Rows with **empty text are dropped** with a warning — they cannot yield entities.
- Only an unreadable file (or one where no text column can be inferred) is a hard error, and it
  is **non-fatal to the run**: the pipeline continues with no extraction rather than failing.

**Example**

```csv
doc_id,source_type,title,date,city,case_id,text
D0001,FIR Narrative,Complaint u/s 370 IPC,2025-11-04,Mumbai,FIR-001,"Complainant states that Aran Mehta (9876543210) transported four women from Siliguri to Mumbai in a Bolero bearing MH 02 AB 1234. Payment of Rs 4,50,000 was routed through account 1234567890123 at a Kurla branch."
D0002,Surveillance Report,Observation — Kurla transit point,2025-11-09,Mumbai,FIR-001,"Subject observed meeting an unidentified male outside Rahat Travels. IMEI 356938035643809 active at the location. Two minors reportedly present."
```

From those two rows NETRA extracts and cross-references a person, a phone number, a vehicle
registration, three locations, an organisation, a bank account, an IMEI, a currency amount and
a minor-victim indicator — each shown highlighted **in the sentence that produced it**, which
is what lets an investigator verify the lead instead of trusting the model.

> **Entity types extracted (12):** person, phone, vehicle, location, organisation, bank
> account, crypto wallet, IMEI, email/handle, weapon, contraband, currency amount. See
> [Architecture § Layer 1](ARCHITECTURE.md) for the scoring and span-arbitration rules.

---

## Minimal working example

The smallest input that produces a meaningful analysis — three tiny files:

**persons.csv**
```csv
person_id,name,age,prior_offences,offence_types,status
P1,Aran Mehta,44,6,Narcotics;Extortion,Active
P2,Bela Roy,31,2,Narcotics,Active
P3,Chirag Das,52,9,Arms;Narcotics,In Judicial Custody
P4,Divya Nair,28,0,,Active
```

**relationships.csv**
```csv
source_id,target_id,relation_type,strength,interactions
P1,P2,Gang Member,0.9,14
P1,P3,Co-Accused,0.8,5
P2,P4,Financial,0.7,9
```

**crime_records.csv**
```csv
case_id,persons_involved,crime_type,severity,date
FIR-001,P1;P2;P3,Narcotics,4,2025-02-11
FIR-002,P2;P4,Financial Fraud,3,2025-06-30
```

From these, NETRA builds a 4-node graph, infers the P1–P2, P1–P3, P2–P3 (FIR-001) and P2–P4
(FIR-002) co-accused ties on top of the recorded links, derives P4's risk from priors, and
excludes P3 from arrest priority (in custody).

---

## The Data Quality panel

Every ingestion produces a `ValidationReport` that the dashboard renders verbatim. It has four
streams:

- **Errors** (red) — hard problems that stopped the run (missing required column). Rare.
- **Warnings** (amber) — repairs you should know about: stub records created for unknown IDs,
  values clipped into range, orphan links preserved.
- **Info** — routine coercions and defaults applied.
- **Renamed** — every column NETRA auto-mapped, e.g. `weight → strength`, so you can confirm
  it interpreted your headers correctly.

A clean report means your CSVs matched the contract exactly. A report full of warnings does
**not** mean the analysis is invalid — it means NETRA repaired your data and is telling you
precisely what it did.

---

## Validation checklist

Before uploading, confirm:

- [ ] `persons.csv` has a `person_id` (or a recognised alias) column, and IDs are unique.
- [ ] `relationships.csv` has `source_id` and `target_id` (or aliases).
- [ ] `crime_records.csv` has `case_id` and `persons_involved` (or aliases).
- [ ] IDs are **consistent across all three files** (`P0007` everywhere, not `P7` in one and
      `P0007` in another) — otherwise NETRA treats them as different people and creates stubs.
- [ ] `persons_involved` cells are delimited with `;` `,` `\|` or `/`.
- [ ] `strength` is in 0–1 and `severity` is in 1–5 (out-of-range values are clipped, not
      rejected — but clip them yourself if the exact number matters).
- [ ] To exclude incarcerated/deceased actors from arrest priority, use the exact status
      strings `In Judicial Custody` / `Deceased`.
- [ ] `documents.csv` (if used) has a `text` column, and — to get case-traceable entities —
      a `case_id` that matches `crime_records.csv`.

If a file doesn't fit this shape, you don't need to reformat by hand — upload it and read the
Data Quality panel to see how NETRA interpreted it.
