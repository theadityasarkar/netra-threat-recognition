"""
Layer 3c — light Graph Neural Network (pure NumPy).

A 2-layer **GraphSAGE-style encoder**, implemented from scratch with analytic
backpropagation and an Adam optimiser. No PyTorch, no TensorFlow: the whole install
stays under 100 MB, runs CPU-only, and trains a 220-node graph in well under a second.

**Encoder** (per layer ℓ, node v):

    a⁽ˡ⁾ᵛ = Σ_{u∈N(v)} ŵ_vu · h⁽ˡ⁾ᵤ          (degree-normalised weighted mean aggregate)
    h⁽ˡ⁺¹⁾ᵛ = tanh( W⁽ˡ⁾ · [ h⁽ˡ⁾ᵛ ‖ a⁽ˡ⁾ᵛ ] + b⁽ˡ⁾ )

**Objective** — unsupervised link prediction. The model must learn to score observed
links above degree-matched negative samples:

    L = −Σ_{(u,v)∈E} log σ(zᵤ·z_v) − Σ_{(u,v')∈E⁻} log(1 − σ(zᵤ·z_v'))

Nothing is labelled, so nothing needs a ground truth an investigator does not have.

**Three products used downstream**

1. ``embeddings`` — a 32-dim role vector per actor. Structurally similar actors land
   close together even if they have never met, which is what lets the dashboard answer
   *"who else behaves like this kingpin?"*.
2. ``residuals`` — per-node link-reconstruction error. A high residual means the
   learnt grammar of this network does not explain that person's connection pattern:
   a structurally anomalous actor. This is the GNN's contribution to the anomaly score.
3. ``similar_actors()`` — cosine nearest-neighbour search over the embedding space,
   for succession analysis after an arrest.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from src.config import GNN_CONFIG


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GNNResult:
    """Trained encoder outputs."""

    node_ids: list[str]
    embeddings: np.ndarray            # (n, embedding_dim)
    residuals: pd.Series              # per-node reconstruction error, 0–1 normalised
    loss_history: list[float]
    coords_2d: np.ndarray             # (n, 2) PCA projection for plotting
    epochs_run: int
    final_loss: float
    auc: float                        # held-out link-prediction ROC-AUC

    def embedding_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.embeddings,
            index=pd.Index(self.node_ids, name="person_id"),
            columns=[f"z{i}" for i in range(self.embeddings.shape[1])],
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable — the naive form overflows on the first few epochs.
    out = np.empty_like(x, dtype=float)
    pos, neg = x >= 0, x < 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def _standardise(X: np.ndarray) -> np.ndarray:
    """Z-score, then squash to keep heavy-tailed graph features in range."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return np.tanh((X - mu) / sd / 2.0)


def _normalised_adjacency(G: nx.Graph, node_ids: list[str]) -> np.ndarray:
    """Row-normalised weighted adjacency (mean aggregation, no self-loop)."""
    idx = {n: i for i, n in enumerate(node_ids)}
    n = len(node_ids)
    A = np.zeros((n, n), dtype=np.float32)
    for u, v, d in G.edges(data=True):
        if u in idx and v in idx:
            w = float(d.get("weight", 1.0))
            A[idx[u], idx[v]] = w
            A[idx[v], idx[u]] = w
    row = A.sum(axis=1, keepdims=True)
    row[row < 1e-8] = 1.0
    return A / row


def _scatter_add_rows(out: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> None:
    """In-place ``out[idx] += vals`` with duplicate indices accumulated.

    ``np.add.at`` is the idiomatic form but runs an un-buffered Python-level loop and
    dominated the training profile. A per-column ``bincount`` does the same scatter in
    compiled code and is roughly an order of magnitude faster at these shapes.
    """
    n, d = out.shape
    for k in range(d):
        out[:, k] += np.bincount(idx, weights=vals[:, k], minlength=n)


def _alias_sampler(probs: np.ndarray):
    """Return ``fn(rng, size) -> indices`` sampling from ``probs``.

    ``rng.choice(..., p=probs)` rebuilds and validates the CDF on every call, which is
    wasteful inside a training loop. Precomputing the CDF once and using
    ``searchsorted`` on uniforms gives identical semantics far more cheaply.
    """
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0

    def sample(rng: np.random.Generator, size: int) -> np.ndarray:
        return np.searchsorted(cdf, rng.random(size), side="right")

    return sample


class _Adam:
    """Minimal Adam optimiser over a list of parameter arrays."""

    def __init__(self, params: list[np.ndarray], lr: float,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _pca_2d(Z: np.ndarray, seed: int = 42) -> np.ndarray:
    """Two leading principal components — deterministic, no sklearn round-trip."""
    if Z.shape[0] < 2:
        return np.zeros((Z.shape[0], 2))
    Zc = Z - Z.mean(axis=0, keepdims=True)
    try:
        _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
        return Zc @ Vt[:2].T
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(seed)
        return Zc @ rng.normal(size=(Z.shape[1], 2))


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_gnn(
    G: nx.Graph,
    features: pd.DataFrame,
    config: dict | None = None,
    progress_cb=None,
) -> GNNResult:
    """Train the 2-layer GraphSAGE encoder on link prediction.

    Args:
        G: the criminal network graph.
        features: node feature matrix from ``graph.builder.compute_node_features``.
        config: overrides for :data:`src.config.GNN_CONFIG`.
        progress_cb: optional ``fn(epoch, total, loss)`` for a UI progress bar.

    Returns:
        :class:`GNNResult`.
    """
    cfg = {**GNN_CONFIG, **(config or {})}
    rng = np.random.default_rng(cfg["seed"])

    node_ids = list(G.nodes())
    n = len(node_ids)
    if n == 0:
        return GNNResult([], np.zeros((0, cfg["embedding_dim"])), pd.Series(dtype=float),
                         [], np.zeros((0, 2)), 0, 0.0, 0.5)

    idx = {node: i for i, node in enumerate(node_ids)}
    X = _standardise(features.reindex(node_ids).fillna(0.0).to_numpy(dtype=np.float64))
    A = _normalised_adjacency(G, node_ids).astype(np.float64)

    d_in = X.shape[1]
    d_hid, d_out = cfg["hidden_dim"], cfg["embedding_dim"]

    # He-style init on the concatenated [self ‖ aggregate] input of each layer.
    W1 = rng.normal(0, np.sqrt(2.0 / (2 * d_in)), size=(2 * d_in, d_hid))
    b1 = np.zeros(d_hid)
    W2 = rng.normal(0, np.sqrt(2.0 / (2 * d_hid)), size=(2 * d_hid, d_out))
    b2 = np.zeros(d_out)
    params = [W1, b1, W2, b2]

    # ── Edge sets ─────────────────────────────────────────────────────────────
    edges = np.array([[idx[u], idx[v]] for u, v in G.edges()
                      if u in idx and v in idx], dtype=np.int64)
    if len(edges) == 0:
        Z = np.tanh(np.concatenate([X, A @ X], axis=1) @ W1 + b1)
        return GNNResult(node_ids, Z[:, :d_out], pd.Series(0.0, index=node_ids),
                         [], _pca_2d(Z[:, :d_out]), 0, 0.0, 0.5)

    # Hold out 10 % of links to report an honest ROC-AUC rather than a training score.
    perm = rng.permutation(len(edges))
    n_test = max(1, int(0.10 * len(edges))) if len(edges) >= 20 else 0
    test_edges = edges[perm[:n_test]] if n_test else np.empty((0, 2), dtype=np.int64)
    train_edges = edges[perm[n_test:]]

    # Degree-proportional negative sampling: hard negatives are hubs, so the model
    # cannot cheat by scoring "any two popular nodes are linked".
    deg = np.asarray([max(1, G.degree(node)) for node in node_ids], dtype=np.float64)
    neg_p = (deg ** 0.75) / (deg ** 0.75).sum()
    sample_negatives = _alias_sampler(neg_p)

    existing = {(int(a), int(b)) for a, b in edges}
    existing |= {(int(b), int(a)) for a, b in edges}

    n_neg = int(cfg["negative_samples"]) * len(train_edges)
    l2 = float(cfg["l2"])
    opt = _Adam(params, float(cfg["learning_rate"]))
    loss_history: list[float] = []
    epochs = int(cfg["epochs"])

    def _forward(Xin: np.ndarray):
        """Two SAGE layers. Returns embeddings plus the cache needed for backprop."""
        agg1 = A @ Xin
        H1_in = np.concatenate([Xin, agg1], axis=1)
        H1_pre = H1_in @ W1 + b1
        H1 = np.tanh(H1_pre)

        agg2 = A @ H1
        H2_in = np.concatenate([H1, agg2], axis=1)
        H2_pre = H2_in @ W2 + b2
        Z = np.tanh(H2_pre)
        return Z, (H1_in, H1_pre, H1, H2_in, H2_pre)

    for epoch in range(epochs):
        Z, (H1_in, H1_pre, H1, H2_in, H2_pre) = _forward(X)

        neg_src = sample_negatives(rng, n_neg)
        neg_dst = sample_negatives(rng, n_neg)
        keep = neg_src != neg_dst
        neg_src, neg_dst = neg_src[keep], neg_dst[keep]

        pu, pv = train_edges[:, 0], train_edges[:, 1]
        pos_score = np.einsum("ij,ij->i", Z[pu], Z[pv])
        neg_score = np.einsum("ij,ij->i", Z[neg_src], Z[neg_dst])

        pos_sig, neg_sig = _sigmoid(pos_score), _sigmoid(neg_score)
        eps = 1e-9
        loss = (-np.log(pos_sig + eps).mean()
                - np.log(1.0 - neg_sig + eps).mean()
                + l2 * (np.sum(W1 ** 2) + np.sum(W2 ** 2)))
        loss_history.append(float(loss))

        # ── Backward: dL/dZ from both terms ───────────────────────────────────
        dZ = np.zeros_like(Z)
        gp = ((pos_sig - 1.0) / len(pu))[:, None]
        _scatter_add_rows(dZ, pu, gp * Z[pv])
        _scatter_add_rows(dZ, pv, gp * Z[pu])
        if len(neg_src):
            gn = (neg_sig / len(neg_src))[:, None]
            _scatter_add_rows(dZ, neg_src, gn * Z[neg_dst])
            _scatter_add_rows(dZ, neg_dst, gn * Z[neg_src])

        # Layer 2
        dH2_pre = dZ * (1.0 - np.tanh(H2_pre) ** 2)
        gW2 = H2_in.T @ dH2_pre + 2 * l2 * W2
        gb2 = dH2_pre.sum(axis=0)
        dH2_in = dH2_pre @ W2.T
        # [self ‖ aggregate] split: the aggregate branch back-propagates through Aᵀ.
        dH1 = dH2_in[:, :d_hid] + A.T @ dH2_in[:, d_hid:]

        # Layer 1
        dH1_pre = dH1 * (1.0 - np.tanh(H1_pre) ** 2)
        gW1 = H1_in.T @ dH1_pre + 2 * l2 * W1
        gb1 = dH1_pre.sum(axis=0)

        opt.step(params, [gW1, gb1, gW2, gb2])
        W1, b1, W2, b2 = params

        if progress_cb is not None and (epoch % 25 == 0 or epoch == epochs - 1):
            progress_cb(epoch + 1, epochs, float(loss))

    Z, _ = _forward(X)

    # ── Held-out link-prediction AUC ──────────────────────────────────────────
    auc = 0.5
    if len(test_edges):
        neg_test: list[tuple[int, int]] = []
        guard = 0
        while len(neg_test) < len(test_edges) and guard < 40 * len(test_edges) + 200:
            a, b = int(rng.integers(n)), int(rng.integers(n))
            guard += 1
            if a != b and (a, b) not in existing:
                neg_test.append((a, b))
        if neg_test:
            nt = np.array(neg_test)
            ps = np.einsum("ij,ij->i", Z[test_edges[:, 0]], Z[test_edges[:, 1]])
            ns = np.einsum("ij,ij->i", Z[nt[:, 0]], Z[nt[:, 1]])
            # ROC-AUC = P(positive scored above negative), ties counted as half.
            wins = (ps[:, None] > ns[None, :]).sum() + 0.5 * (ps[:, None] == ns[None, :]).sum()
            auc = float(wins / (len(ps) * len(ns)))

    residuals = _reconstruction_residuals(G, Z, node_ids, rng)

    return GNNResult(
        node_ids=node_ids,
        embeddings=Z,
        residuals=residuals,
        loss_history=loss_history,
        coords_2d=_pca_2d(Z, cfg["seed"]),
        epochs_run=epochs,
        final_loss=loss_history[-1] if loss_history else 0.0,
        auc=auc,
    )


def _reconstruction_residuals(
    G: nx.Graph,
    Z: np.ndarray,
    node_ids: list[str],
    rng: np.random.Generator,
) -> pd.Series:
    """Per-node link-reconstruction error, normalised to 0–1.

    For each actor we ask two questions the trained model should answer well:
      • do its *real* links score high?      (missed-link error)
      • do *random* non-links score low?     (false-affinity error)

    An actor whose neighbourhood the model cannot reproduce is structurally
    unusual — exactly the kind of actor an analyst should look at by hand.
    """
    idx = {node: i for i, node in enumerate(node_ids)}
    n = len(node_ids)
    if n == 0:
        return pd.Series(dtype=float)

    errors = np.zeros(n)
    for node in node_ids:
        i = idx[node]
        nbrs = [idx[m] for m in G.neighbors(node) if m in idx]
        if not nbrs:
            # A pure isolate carries no *network* reconstruction signal — the encoder
            # is not "surprised" by an absence of links. Keep it low so isolated nodes
            # don't masquerade as structural anomalies; the rule engine flags the
            # genuinely interesting isolates (serious record + no contacts) separately.
            errors[i] = 0.25
            continue

        pos = _sigmoid(Z[nbrs] @ Z[i])
        missed = float(np.mean(1.0 - pos))

        n_neg = min(30, max(5, len(nbrs) * 3))
        cand = rng.integers(0, n, size=n_neg * 2)
        nbr_set = set(nbrs) | {i}
        cand = np.array([c for c in cand if c not in nbr_set][:n_neg])
        if len(cand):
            false_affinity = float(np.mean(_sigmoid(Z[cand] @ Z[i])))
        else:
            false_affinity = 0.0

        errors[i] = 0.7 * missed + 0.3 * false_affinity

    s = pd.Series(errors, index=node_ids, name="gnn_residual")
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(n), index=node_ids, name="gnn_residual")
    return ((s - lo) / (hi - lo)).rename("gnn_residual")


# ──────────────────────────────────────────────────────────────────────────────
# Downstream utilities
# ──────────────────────────────────────────────────────────────────────────────

def similar_actors(
    result: GNNResult,
    person_id: str,
    top_n: int = 8,
    exclude_neighbours_of: nx.Graph | None = None,
) -> pd.DataFrame:
    """Cosine nearest neighbours in embedding space — structural twins.

    Operationally: *if this actor is arrested, who is positioned to take over?*
    Passing ``exclude_neighbours_of`` drops direct contacts, which surfaces the
    non-obvious matches rather than restating the person's own address book.
    """
    if person_id not in result.node_ids or len(result.node_ids) < 2:
        return pd.DataFrame()

    Z = result.embeddings
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    Zn = Z / norms

    i = result.node_ids.index(person_id)
    sims = Zn @ Zn[i]

    exclude = {person_id}
    if exclude_neighbours_of is not None and person_id in exclude_neighbours_of:
        exclude |= set(exclude_neighbours_of.neighbors(person_id))

    rows = [
        {"person_id": pid, "similarity": round(float(sims[j]), 4)}
        for j, pid in enumerate(result.node_ids) if pid not in exclude
    ]
    df = pd.DataFrame(rows).sort_values("similarity", ascending=False)
    return df.head(top_n).reset_index(drop=True)


def embedding_kmeans(result: GNNResult, k: int = 5, seed: int = 42) -> pd.Series:
    """Cluster the embedding space into behavioural archetypes.

    Distinct from Louvain communities: Louvain groups actors who are *connected*,
    this groups actors who *play the same role*, even in different organisations.
    """
    if len(result.node_ids) < k or k < 2:
        return pd.Series(0, index=result.node_ids, name="behaviour_cluster")
    try:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(result.embeddings)
    except Exception:                                       # noqa: BLE001 — sklearn optional
        labels = np.zeros(len(result.node_ids), dtype=int)
    return pd.Series(labels, index=result.node_ids, name="behaviour_cluster")
