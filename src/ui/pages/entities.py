"""
Entity Extraction page — the PS's second requirement, made auditable.

The problem statement asks the system to *"extract important entities such as people,
locations, vehicles, phone numbers, and organizations"*. This screen shows the
result of that extraction and, critically, shows it **next to the sentence it came
from**. An entity table on its own is an assertion; an entity table with a
one-click path back to the source narrative is evidence an investigator can act on.

Identifiers are masked on screen by default (see
:func:`~src.ingestion.entities.mask_value`) because this view is the one most likely
to be projected during a briefing. The unmasked values remain in the export.
"""

from __future__ import annotations

import html as _html

import pandas as pd
import streamlit as st

from src.config import ENTITY_COLORS, ENTITY_ICONS, ENTITY_LABELS, NLP_PARAMS, PALETTE
from src.ingestion.entities import extract_entities, front_business_candidates, mask_value
from src.ui.components import empty_state, metric_cards, section_header
from src.ui.state import get as state_get
from src.visualization import charts


def _pill(entity_type: str, text: str, extra: str = "") -> str:
    tone = ENTITY_COLORS.get(entity_type, PALETTE["accent"])
    icon = ENTITY_ICONS.get(entity_type, "•")
    suffix = f'<span style="opacity:0.65;">{_html.escape(extra)}</span>' if extra else ""
    return (f'<span class="ent-pill" style="--tone:{tone}">{icon} '
            f'{_html.escape(str(text))}{suffix}</span>')


def _highlight(text: str, mentions: pd.DataFrame) -> str:
    """Return the document with every extracted span wrapped in ``<mark>``.

    Escaping is done per-segment rather than up front: escaping first would change
    the string length and invalidate every character offset the extractor recorded.
    Overlapping spans are dropped defensively — the extractor arbitrates claims, but
    a stray overlap here would produce broken markup rather than a caught bug.
    """
    if not len(mentions):
        return _html.escape(text)

    spans = (mentions[["start", "end", "entity_type"]]
             .astype({"start": int, "end": int})
             .sort_values("start").to_dict("records"))

    out: list[str] = []
    cursor = 0
    for s in spans:
        if s["start"] < cursor or s["end"] > len(text):
            continue
        out.append(_html.escape(text[cursor:s["start"]]))
        tone = ENTITY_COLORS.get(s["entity_type"], PALETTE["accent"])
        fragment = _html.escape(text[s["start"]:s["end"]])
        out.append(f'<mark style="border-color:{tone};background:{tone}22;" '
                   f'title="{s["entity_type"]}">{fragment}</mark>')
        cursor = s["end"]
    out.append(_html.escape(text[cursor:]))
    return "".join(out)


def _entity_table(extraction, redact: bool) -> None:
    ents = extraction.entities
    types = sorted(ents["entity_type"].unique())

    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        picked = st.multiselect(
            "Entity types", types,
            default=[t for t in ("PERSON", "PHONE", "VEHICLE", "LOCATION", "ORGANISATION")
                     if t in types] or types,
            format_func=lambda t: f"{ENTITY_ICONS.get(t, '•')} {ENTITY_LABELS.get(t, t)}",
            key="ent_types")
    with c2:
        floor = st.slider("Minimum confidence", 0.40, 0.99,
                          float(NLP_PARAMS["min_confidence"]), 0.01, key="ent_conf")
    with c3:
        query = st.text_input("Search value", "", key="ent_query",
                              placeholder="name, number, plate…")

    view = ents[ents["entity_type"].isin(picked or types)]
    view = view[view["confidence"] >= floor]
    if query.strip():
        needle = query.strip().lower()
        view = view[view["value"].astype(str).str.lower().str.contains(needle, regex=False)]

    if not len(view):
        empty_state("No entities match these filters.", "🔍")
        return

    value_col = "display" if redact else "value"
    show = view[["entity_type", value_col, "n_mentions", "n_documents",
                 "confidence", "documents"]].copy()
    show.columns = ["Type", "Value", "Mentions", "Documents", "Confidence", "Seen in"]
    show["Type"] = show["Type"].map(lambda t: f"{ENTITY_ICONS.get(t, '•')} "
                                             f"{ENTITY_LABELS.get(t, t)}")
    show["Seen in"] = show["Seen in"].map(
        lambda ds: ", ".join(ds[:3]) + (f" +{len(ds) - 3}" if len(ds) > 3 else ""))

    st.dataframe(
        show, width="stretch", hide_index=True, height=420,
        column_config={"Confidence": st.column_config.ProgressColumn(
            "Confidence", min_value=0.0, max_value=1.0, format="%.2f")})
    st.caption(f"{len(view):,} of {len(ents):,} extracted entities shown."
               + ("  Identifiers are masked — untick **Redact identifiers** in the "
                  "sidebar to reveal them." if redact else ""))


def _gallery(extraction, redact: bool) -> None:
    """Entities grouped by type as coloured pills — the fastest visual scan."""
    ents = extraction.entities
    value_col = "display" if redact else "value"
    order = [t for t in ENTITY_LABELS if t in set(ents["entity_type"])]
    for entity_type in order:
        rows = ents[ents["entity_type"] == entity_type].head(24)
        if not len(rows):
            continue
        label = f"{ENTITY_ICONS.get(entity_type, '•')} {ENTITY_LABELS.get(entity_type, entity_type)}"
        total = int((ents["entity_type"] == entity_type).sum())
        pills = "".join(
            _pill(entity_type, r[value_col],
                  f" ×{int(r['n_mentions'])}" if int(r["n_mentions"]) > 1 else "")
            for _, r in rows.iterrows())
        more = (f'<span style="font-size:0.72rem;color:{PALETTE["text_dim"]};">'
                f'+{total - len(rows)} more</span>' if total > len(rows) else "")
        st.markdown(
            f'<div style="margin-bottom:0.7rem;">'
            f'<div style="font-size:0.74rem;color:{PALETTE["text_dim"]};'
            f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.25rem;">'
            f'{label} · {total}</div>{pills}{more}</div>', unsafe_allow_html=True)


def _document_inspector(result, extraction) -> None:
    docs = result.documents
    if not len(docs) or "text" not in docs.columns:
        empty_state("The source corpus is not attached to this run, so highlighted "
                    "text is unavailable.", "📄")
        return

    ids = docs["doc_id"].astype(str).tolist()
    titles = dict(zip(ids, docs.get("title", pd.Series(ids, index=docs.index)).astype(str)))
    sources = dict(zip(ids, docs.get("source_type", pd.Series("", index=docs.index)).astype(str)))

    picked = st.selectbox(
        "Source document", ids, key="ent_doc",
        format_func=lambda d: f"{d} · {sources.get(d, '')} · {titles.get(d, '')}"[:96])

    row = docs[docs["doc_id"].astype(str) == picked].iloc[0]
    text = str(row.get("text", ""))
    doc_mentions = extraction.mentions[extraction.mentions["doc_id"].astype(str) == picked]

    meta = " · ".join(str(row.get(k, "")) for k in ("source_type", "date", "city", "case_id")
                      if row.get(k, ""))
    st.markdown(f"""
    <div class="doc-card">
      <div class="doc-head">
        <span class="doc-title">{_html.escape(str(row.get('title', picked)))}</span>
        <span class="doc-meta">{_html.escape(meta)}</span>
        <span class="doc-meta" style="margin-left:auto;">
          {len(doc_mentions)} mention(s) · {doc_mentions['entity_id'].nunique()} entity(ies)</span>
      </div>
      <div class="doc-body">{_highlight(text, doc_mentions)}</div>
    </div>""", unsafe_allow_html=True)

    if len(doc_mentions):
        st.markdown("**Extracted from this document**")
        pills = "".join(_pill(m["entity_type"], m["raw"]) for _, m in
                        doc_mentions.sort_values("start").iterrows())
        st.markdown(pills, unsafe_allow_html=True)

        with st.expander("🔎 Mention-level detail with surrounding context"):
            cols = ["entity_type", "raw", "value", "confidence", "start", "end", "context"]
            st.dataframe(doc_mentions[cols].sort_values("start"),
                         width="stretch", hide_index=True)


def _graph_resolution(result, extraction) -> None:
    """Which extracted persons correspond to actors already in the network.

    This is the join that makes text extraction operationally useful: a name lifted
    from a surveillance note is only actionable once it is tied to a node that has
    centrality, a module and a case history behind it.
    """
    persons = extraction.of_type("PERSON")
    if not len(persons):
        empty_state("No person entities were extracted.", "👤")
        return

    by_name = {str(d.get("name", "")).strip().lower(): n
               for n, d in result.graph.nodes(data=True) if d.get("name")}
    rows = []
    for _, r in persons.iterrows():
        node = by_name.get(str(r["value"]).strip().lower())
        rows.append({
            "Extracted name": r["value"],
            "Resolved actor": result.label(node) if node else "— not in network —",
            "In network": bool(node),
            "Documents": int(r["n_documents"]),
            "Confidence": float(r["confidence"]),
            "Influence": (float(result.key_players.loc[node, "influence_score"])
                          if node and node in result.key_players.index else None),
            "Suspicion": (float(result.anomalies.loc[node, "suspicion_score"])
                          if node and node in result.anomalies.index else None),
        })
    frame = pd.DataFrame(rows).sort_values(
        ["In network", "Confidence"], ascending=[False, False])

    matched = int(frame["In network"].sum())
    metric_cards([
        {"label": "Person entities", "value": len(frame), "accent": PALETTE["accent"]},
        {"label": "Resolved to actors", "value": matched, "accent": PALETTE["low"],
         "delta": f"{matched / len(frame) * 100:.0f}% of extracted names"},
        {"label": "Unresolved", "value": len(frame) - matched, "accent": PALETTE["medium"],
         "delta": "candidate new nodes"},
    ])
    st.dataframe(frame, width="stretch", hide_index=True, height=330,
                 column_config={"In network": st.column_config.CheckboxColumn("In network")})
    st.caption(
        "Unresolved names are not errors — they are candidate additions to the roster, "
        "the normal output of reading intelligence the structured databases have not "
        "caught up with yet.")


def render(result) -> None:
    extraction = getattr(result, "extraction", None)
    if extraction is None or not extraction.n_entities:
        empty_state(
            "No unstructured intelligence in this run. Load the sample dataset, or "
            "upload a documents CSV with a `text` column, to exercise the NLP layer.",
            "📄")
        with st.expander("ℹ️ What this screen does"):
            st.markdown(
                "The PS requires entity extraction from FIR narratives, surveillance "
                "reports, social-media intelligence and intelligence agency reports. "
                "NETRA recognises 12 entity types with India-specific patterns — mobile "
                "series, RTO vehicle plates, IFSC codes, Luhn-checked IMEIs, crypto "
                "wallets — plus gazetteers of states, cities, organisation suffixes, "
                "weapons and contraband.")
        return

    redact = bool(state_get("redact", False))
    stats = extraction.stats

    metric_cards([
        {"label": "Documents", "value": stats.get("documents", 0), "accent": PALETTE["accent"]},
        {"label": "Distinct entities", "value": extraction.n_entities,
         "accent": PALETTE["accent_2"]},
        {"label": "Mentions", "value": extraction.n_mentions, "accent": PALETTE["low"]},
        {"label": "Entity types", "value": len(stats.get("by_type", {})),
         "accent": PALETTE["medium"], "delta": f"of {len(ENTITY_LABELS)} supported"},
        {"label": "Co-occurrence links", "value": stats.get("links", 0),
         "accent": PALETTE["high"]},
        {"label": "Mean confidence", "value": f"{stats.get('mean_confidence', 0):.2f}",
         "accent": PALETTE["accent"]},
    ])

    for w in extraction.warnings:
        st.warning(w)

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.plotly_chart(charts.entity_type_bars(extraction), width="stretch",
                        config={"displayModeBar": False})
    with c2:
        st.plotly_chart(charts.entity_confidence_hist(extraction), width="stretch",
                        config={"displayModeBar": False})

    t_table, t_docs, t_gallery, t_resolve, t_fronts, t_live = st.tabs([
        "📋 Entity register", "📄 Source documents", "🎨 By type",
        "🔗 Actor resolution", "🏢 Front businesses", "⚡ Live Text & Hindi FIR Analyzer"])

    with t_table:
        section_header("Entity register", "filterable · every row traceable to a document")
        _entity_table(extraction, redact)
        st.markdown("**Export**")
        e1, e2, e3 = st.columns(3)
        e1.download_button("⬇ entities.csv",
                           extraction.entities.to_csv(index=False).encode(),
                           "netra_entities.csv", "text/csv", width="stretch")
        e2.download_button("⬇ mentions.csv",
                           extraction.mentions.to_csv(index=False).encode(),
                           "netra_mentions.csv", "text/csv", width="stretch")
        e3.download_button("⬇ cooccurrence_links.csv",
                           extraction.links.to_csv(index=False).encode(),
                           "netra_entity_links.csv", "text/csv", width="stretch")
        st.caption("Exports carry unmasked values — treat them as case material.")

    with t_docs:
        section_header("Source documents", "extracted spans highlighted in place")
        _document_inspector(result, extraction)

    with t_gallery:
        section_header("Entities by type", "quick visual scan of the whole corpus")
        _gallery(extraction, redact)

    with t_resolve:
        section_header("Resolution against the network",
                       "extracted names matched to graph actors")
        _graph_resolution(result, extraction)

    with t_fronts:
        section_header("Front-business candidates",
                       "organisation names matching known cover archetypes")
        fronts = front_business_candidates(extraction)
        if len(fronts):
            st.dataframe(fronts, width="stretch", hide_index=True)
            st.caption(
                "Placement agencies, spas, bars, 'travels' and similar recur as covers "
                "in NCRB Women Safety casework, so a suffix match is surfaced rather "
                "than left in the general organisation list. A match is a lead to check, "
                "not a finding — most such businesses are legitimate.")
        else:
            empty_state("No organisation name matched a front-business suffix.", "🏢")

    with t_live:
        section_header("Live Text & Multi-Lingual FIR Analyzer",
                       "interactive extraction on English, Hindi (हिन्दी) & Hinglish narratives")
        _live_text_analyzer(redact)


def _live_text_analyzer(redact: bool) -> None:
    st.markdown(
        "Paste any raw FIR text, interrogation transcript, or intelligence report in "
        "**English, Hindi (हिन्दी), or Hinglish** to extract entities and connections in real-time."
    )

    c1, c2, c3 = st.columns(3)
    if c1.button("📝 Sample Hindi Police FIR", width="stretch"):
        st.session_state["live_fir_input"] = (
            "विशेष सूचना के आधार पर आरोपी रमेश कुमार और सरगना विजय शर्मा को थाना कोटवाली क्षेत्र में घेराबंदी कर पकड़ा गया। "
            "आरोपी वाहन DL 03 C 4521 में सवार थे। तलाशी में 1 तमंचा, 12 जिंदा कारतूस, ₹ 1,50,000 नकद और 400 ग्राम चरस बरामद हुई। "
            "पूछताछ में पीड़िता प्रिया सिंह को सिलीगुड़ी से लाकर अवैध देह व्यापार में धकेलने की पुष्टि हुई। "
            "संदिग्ध बैंक खाता 918020045512345 और मोबाइल 9876543210 की जांच जारी है।"
        )
    if c2.button("📝 Sample Trafficking Intercept", width="stretch"):
        st.session_state["live_fir_input"] = (
            "Complainant states that accused Aran Mehta, mobile +91 9811223344, transported three women "
            "from Siliguri Corridor to Mumbai in a Scorpio bearing MH 02 CD 9988. Account 445566778899 "
            "received advance of Rs 3,50,000 via IFSC HDFC0001234. Rahat Placement Agency Pvt Ltd "
            "is operating as the front recruitment business."
        )
    if c3.button("📝 Sample Cyber & Crypto Case", width="stretch"):
        st.session_state["live_fir_input"] = (
            "FIR registered under Section 66D IT Act and 370 IPC against suspect Rajesh Verma. "
            "Victim coerced into transferring Rs 2,00,000 to crypto wallet 0x71C8363433ab51c12a08701cdb21e8b8cce83612. "
            "IMEI 356938035643809 active in Gurugram near Cyber Hub."
        )

    val = st.session_state.get("live_fir_input", (
        "विशेष सूचना के आधार पर आरोपी रमेश कुमार और सरगना विजय शर्मा को थाना कोटवाली में पकड़ा गया। "
        "वाहन DL 04 AB 1234 से ₹ 2,50,000 नकद, 1 तमंचा और मोबाइल 9876543210 बरामद हुआ।"
    ))
    user_text = st.text_area("Input Intelligence Narrative / FIR Text", value=val, height=130, key="live_fir_input")

    if user_text.strip():
        mentions = extract_entities(user_text, doc_id="LIVE-INPUT")
        if not mentions:
            st.info("No entities detected in the provided text.")
            return

        m_df = pd.DataFrame([{
            "doc_id": m.doc_id,
            "entity_id": m.entity_id,
            "entity_type": m.entity_type,
            "raw": m.raw,
            "value": m.value,
            "start": m.start,
            "end": m.end,
            "confidence": m.confidence,
            "context": m.context,
        } for m in mentions])

        st.markdown("##### 🔍 Highlighted Intelligence Document")
        highlighted_html = _highlight(user_text, m_df)
        st.markdown(
            f'<div style="background:{PALETTE["surface"]};border:1px solid {PALETTE["border"]};'
            f'border-radius:8px;padding:1rem;line-height:1.8;font-size:0.95rem;">'
            f'{highlighted_html}</div>',
            unsafe_allow_html=True
        )

        st.markdown("##### 🏷️ Extracted Entity Breakdown")
        pills_html = []
        for _, r in m_df.iterrows():
            display_val = mask_value(r["entity_type"], r["value"]) if redact else r["value"]
            pills_html.append(_pill(r["entity_type"], display_val, f"({r['confidence']:.2f})"))
        st.markdown(f'<div style="display:flex;gap:0.4rem;flex-wrap:wrap;margin:0.5rem 0 1rem;">{"".join(pills_html)}</div>', unsafe_allow_html=True)

    with st.expander("⚙️ How extraction works, and what it does not do"):
        st.markdown(f"""
**Approach.** Deterministic pattern recognition plus curated gazetteers, arbitrated by
a span-claim resolver so a specific pattern always beats a generic one — an account
number labelled `a/c` is a bank account even when it happens to be 15 digits long,
which is also the length of an IMEI.

**Why not a transformer NER model.** `en_core_web_trf` is ~430 MB and would not fit
inside the 1 GB Streamlit Community Cloud container alongside the rest of the stack.
Beyond size: extraction feeds a dossier that has to be reproducible, and a generic
English NER model knows nothing about Indian RTO plate formats, IFSC codes, the 6–9
mobile series or IMEI Luhn checksums. The pattern signatures here are model-agnostic,
so a transformer can be added later as an additional recall tier without touching
anything downstream.

**The honest cost.** Recall on person names that carry no cue word — no honorific,
no role word, no `s/o` — is lower than a statistical model would achieve. That
trade is deliberate: a false name in a dossier costs an investigator more than a
missed one, because the missed one is still in the document the analyst is reading.

**Thresholds.** Mentions below {NLP_PARAMS['min_confidence']:.2f} confidence are
discarded. A document mentioning more than {NLP_PARAMS['max_entities_per_doc']}
entities is excluded from co-occurrence linking, the same guard the graph builder
applies to large case files, so a bulk dump cannot manufacture a dense clique.
""")
