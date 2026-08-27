"""
Pipeline orchestration — wires layers 1 → 3 into one call.

Kept free of Streamlit imports so it can be driven from a notebook, a CLI, or the
test suite. ``app.py`` wraps :func:`run_pipeline` in ``st.cache_data`` so a re-render
costs nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

from src.analysis.anomaly import (
    anomaly_summary,
    detect_anomalies,
    suspicious_clusters,
)
from src.analysis.centrality import centrality_correlations, compute_centralities
from src.analysis.communities import (
    bridge_analysis,
    community_profiles,
    critical_edges,
    detect_communities,
    modularity_score,
)
from src.analysis.gnn import GNNResult, embedding_kmeans, train_gnn
from src.analysis.key_players import arrest_priority, key_players_table, role_summary
from src.analysis.women_safety import (
    compute_wsti,
    crime_type_breakdown,
    escalation_list,
    trafficking_corridors,
    women_safety_cases,
    women_safety_summary,
)
from src.graph.builder import build_graph, compute_node_features, graph_summary
from src.ingestion.entities import ExtractionResult, extract_corpus
from src.ingestion.loader import ValidationReport


@dataclass
class AnalysisResult:
    """Everything the dashboard and the report generator need."""

    # Layer 1
    persons: pd.DataFrame
    relationships: pd.DataFrame
    crimes: pd.DataFrame
    validation: ValidationReport

    # Layer 2
    graph: nx.Graph
    features: pd.DataFrame
    summary: dict

    # Layer 3
    centralities: pd.DataFrame
    correlations: pd.DataFrame
    partition: dict[str, int]
    modularity: float
    profiles: pd.DataFrame
    bridges: pd.DataFrame
    cut_edges: pd.DataFrame
    gnn: GNNResult
    behaviour_clusters: pd.Series
    key_players: pd.DataFrame
    roles: pd.DataFrame
    priority: pd.DataFrame
    anomalies: pd.DataFrame
    clusters: pd.DataFrame
    anomaly_stats: dict

    # Layer 3 — Women Safety Division lens (PS 26189 owning department).
    # Defaulted so the dataclass stays constructible from older call sites.
    ws_cases: pd.DataFrame = field(default_factory=pd.DataFrame)
    wsti: pd.DataFrame = field(default_factory=pd.DataFrame)
    corridors: pd.DataFrame = field(default_factory=pd.DataFrame)
    ws_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame)
    ws_escalation: pd.DataFrame = field(default_factory=pd.DataFrame)
    ws_stats: dict = field(default_factory=dict)

    # Layer 1 (unstructured) — NLP entity extraction. ``None`` when the dataset
    # carried no narrative documents, which is the normal case for a CSV-only upload.
    extraction: ExtractionResult | None = None

    # The source corpus itself, retained so the Entity Extraction screen can show
    # each extracted mention highlighted *in its own sentence*. An entity table
    # without the surrounding text is unverifiable, and an investigator who cannot
    # check a lead against the narrative that produced it will not act on it.
    documents: pd.DataFrame = field(default_factory=pd.DataFrame)

    timings: dict[str, float] = field(default_factory=dict)
    generated_at: str = ""

    # ── Convenience accessors used across the UI ──────────────────────────────

    @property
    def n_actors(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_links(self) -> int:
        return self.graph.number_of_edges()

    @property
    def n_modules(self) -> int:
        return len(set(self.partition.values())) if self.partition else 0

    def actor(self, person_id: str) -> dict:
        """Merged profile for one actor across every analysis output."""
        if person_id not in self.graph:
            return {}
        rec = dict(self.graph.nodes[person_id])
        for frame, prefix in ((self.centralities, ""), (self.features, "feat_")):
            if person_id in frame.index:
                rec.update({f"{prefix}{k}": v for k, v in frame.loc[person_id].items()})
        if person_id in self.key_players.index:
            kp = self.key_players.loc[person_id]
            rec.update({"role": kp.get("role"), "influence_score": kp.get("influence_score"),
                        "influence_rank": kp.get("influence_rank"),
                        "disruption_score": kp.get("disruption_score")})
        if person_id in self.anomalies.index:
            an = self.anomalies.loc[person_id]
            rec.update({"suspicion_score": an.get("suspicion_score"), "band": an.get("band"),
                        "rules": an.get("rules"), "reasons": an.get("reasons")})
        rec["module"] = self.partition.get(person_id, -1)
        return rec

    def label(self, person_id: str) -> str:
        """``Name (P0042)`` — the display form used in every selector."""
        if person_id in self.graph:
            return f"{self.graph.nodes[person_id].get('name', person_id)} ({person_id})"
        return str(person_id)


def run_pipeline(
    persons: pd.DataFrame,
    relationships: pd.DataFrame,
    crimes: pd.DataFrame | None = None,
    validation: ValidationReport | None = None,
    *,
    documents: pd.DataFrame | dict[str, str] | None = None,
    infer_coaccused: bool = True,
    resolution: float = 1.0,
    contamination: float = 0.08,
    gnn_epochs: int = 200,
    enable_gnn: bool = True,
    exact_centrality: bool | None = None,
    reference_year: int = 2026,
    progress_cb=None,
) -> AnalysisResult:
    """Run ingestion-ready data through graph building and the full AI analysis.

    Args:
        documents: optional free-text intelligence — FIR narratives, surveillance
            notes, social-media intercepts — as ``{doc_id: text}`` or a frame with
            ``doc_id``/``text`` columns. Entities are extracted for the Entity
            Extraction screen; the co-occurrence links are *not* merged into the
            main graph automatically, because an inferred textual tie and a recorded
            relationship are different grades of evidence and the analyst decides
            whether to promote them.
        progress_cb: optional ``fn(fraction: float, message: str)`` for a status bar.
    """
    t: dict[str, float] = {}
    crimes = crimes if crimes is not None else pd.DataFrame()
    validation = validation or ValidationReport()

    def tick(frac: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(frac, msg)

    # ── Layer 1 (unstructured) — entity extraction ────────────────────────────
    extraction: ExtractionResult | None = None
    doc_frame = pd.DataFrame()
    if documents is not None and len(documents):
        tick(0.02, "Extracting entities from intelligence text…")
        t0 = time.perf_counter()
        try:
            extraction = extract_corpus(documents)
            doc_frame = (documents.copy() if isinstance(documents, pd.DataFrame)
                         else pd.DataFrame({"doc_id": list(documents.keys()),
                                            "text": list(documents.values())}))
        except Exception as exc:                                   # noqa: BLE001
            # A malformed document set must not sink the whole analysis; the
            # structured evidence is independently sufficient.
            validation.warnings.append(f"Entity extraction skipped: {exc}")
        t["extraction"] = time.perf_counter() - t0

    # ── Layer 2 — graph ───────────────────────────────────────────────────────
    tick(0.05, "Building network graph…")
    t0 = time.perf_counter()
    G = build_graph(persons, relationships, crimes, infer_coaccused=infer_coaccused)
    t["graph_build"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    features = compute_node_features(G, reference_year=reference_year)
    summary = graph_summary(G)
    t["features"] = time.perf_counter() - t0

    # ── Layer 3 — centrality ──────────────────────────────────────────────────
    tick(0.22, "Computing centrality measures…")
    t0 = time.perf_counter()
    centralities = compute_centralities(G, exact=exact_centrality)
    correlations = centrality_correlations(centralities)
    t["centrality"] = time.perf_counter() - t0

    # ── Layer 3 — communities ─────────────────────────────────────────────────
    tick(0.42, "Detecting operational modules…")
    t0 = time.perf_counter()
    partition = detect_communities(G, resolution=resolution)
    modularity = modularity_score(G, partition)
    profiles = community_profiles(G, partition, centralities)
    bridges = bridge_analysis(G, partition)
    cut_edges = critical_edges(G, partition)
    t["communities"] = time.perf_counter() - t0

    # ── Layer 3 — GNN ─────────────────────────────────────────────────────────
    tick(0.55, "Training graph neural network…")
    t0 = time.perf_counter()
    if enable_gnn and G.number_of_nodes() >= 8 and G.number_of_edges() >= 4:
        gnn = train_gnn(G, features, config={"epochs": int(gnn_epochs)})
        behaviour = embedding_kmeans(gnn, k=min(6, max(2, G.number_of_nodes() // 25)))
    else:
        import numpy as np
        gnn = GNNResult(list(G.nodes()), np.zeros((G.number_of_nodes(), 2)),
                        pd.Series(0.0, index=list(G.nodes())), [],
                        np.zeros((G.number_of_nodes(), 2)), 0, 0.0, 0.5)
        behaviour = pd.Series(0, index=list(G.nodes()), name="behaviour_cluster")
    t["gnn"] = time.perf_counter() - t0

    # ── Layer 3 — key players ─────────────────────────────────────────────────
    tick(0.74, "Ranking key players…")
    t0 = time.perf_counter()
    kp = key_players_table(G, centralities)
    if len(kp):
        kp["module"] = pd.Series(partition).reindex(kp.index)
    roles = role_summary(kp)
    priority = arrest_priority(kp, top_n=12)
    t["key_players"] = time.perf_counter() - t0

    # ── Layer 3 — anomalies ───────────────────────────────────────────────────
    tick(0.88, "Scoring anomalies and suspicious clusters…")
    t0 = time.perf_counter()
    anomalies = detect_anomalies(
        G, features, centralities, partition,
        gnn_residuals=gnn.residuals if enable_gnn else None,
        contamination=contamination, reference_year=reference_year)
    if len(anomalies) and len(kp):
        anomalies["role"] = kp["role"].reindex(anomalies.index)
        anomalies["influence_score"] = kp["influence_score"].reindex(anomalies.index)
    clusters = suspicious_clusters(G, partition, anomalies, profiles)
    stats = anomaly_summary(anomalies)
    t["anomaly"] = time.perf_counter() - t0

    # ── Layer 3 — Women Safety Division lens ──────────────────────────────────
    # PS 26189 is owned by NCRB's Women Safety Division, so the generic network
    # analysis is re-scored through that mandate: which actors touch women- and
    # minor-victim offences, and which geographic corridors they move victims along.
    tick(0.95, "Applying Women Safety threat lens…")
    t0 = time.perf_counter()
    ws_cases = women_safety_cases(crimes)
    wsti = compute_wsti(G, crimes, centralities)
    corridors = trafficking_corridors(G, crimes, wsti)
    ws_breakdown = crime_type_breakdown(crimes)
    ws_escalation = escalation_list(wsti, top_n=10)
    ws_stats = women_safety_summary(wsti, corridors, crimes)
    t["women_safety"] = time.perf_counter() - t0

    t["total"] = sum(v for k, v in t.items() if k != "total")
    tick(1.0, "Analysis complete")

    return AnalysisResult(
        persons=persons, relationships=relationships, crimes=crimes, validation=validation,
        graph=G, features=features, summary=summary,
        centralities=centralities, correlations=correlations,
        partition=partition, modularity=modularity, profiles=profiles,
        bridges=bridges, cut_edges=cut_edges,
        gnn=gnn, behaviour_clusters=behaviour,
        key_players=kp, roles=roles, priority=priority,
        anomalies=anomalies, clusters=clusters, anomaly_stats=stats,
        ws_cases=ws_cases, wsti=wsti, corridors=corridors,
        ws_breakdown=ws_breakdown, ws_escalation=ws_escalation, ws_stats=ws_stats,
        extraction=extraction,
        documents=doc_frame,
        timings=t,
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
