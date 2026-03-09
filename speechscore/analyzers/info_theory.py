"""
SpeechScore 2.0 — Information-Theoretic Coherence  ⭐ NOVEL  (V2-3)

Replaces the naive Pearson-correlation-based coherence module with
**Mutual Information (MI)** and **Transfer Entropy (TE)** to capture
*nonlinear* cross-modal dependencies between speech channels — and
determine the *direction* of information flow.

Novelty argument
----------------
Cross-modal coherence in speech assessment has previously been measured
using linear correlation (Pearson r between text sentiment and vocal
energy).  This misses:

  1. **Nonlinear dependencies** — a speaker might modulate energy
     quadratically with sentiment intensity, which Pearson r = 0 for
     even though the relationship is perfect.
  2. **Directional information flow** — does prosody *predict* content
     shifts, or does content drive prosody?  Linear correlation is
     symmetric and cannot answer this.
  3. **Conditional dependencies** — does channel A tell you something
     about channel B *beyond what the past of B already tells you*?

We introduce:

  - **MI(X; Y)** — mutual information between speech channels,
    capturing *all* statistical dependencies (linear + nonlinear).
    Estimated via k-nearest-neighbour method (Kraskov et al. 2004).
  - **TE(X → Y)** — transfer entropy from X to Y, measuring the
    directed information flow.  TE(X→Y) > TE(Y→X) indicates X
    influences Y more than Y influences X.
  - **Normalised MI** — MI / min(H(X), H(Y)), giving a [0, 1]
    measure of information-theoretic coupling strength.

Prior art:
  - Schreiber (2000) introduced Transfer Entropy for coupled systems.
  - Kraskov et al. (2004) proposed KNN-based MI estimation.
  - Fusaroli & Tylén (2016) used CRQA for conversational coupling.
  - Nobody has applied MI + TE to within-speaker prosodic-semantic
    coherence for communication assessment.

Cross-modal channel pairs analysed:
  1. Energy ↔ Sentiment — vocal force vs emotional content
  2. Pitch variability ↔ Speech rate — prosodic coupling
  3. Pause frequency ↔ Sentence complexity — cognitive planning alignment

Mathematical foundation
-----------------------
**Mutual Information** (Shannon):
  MI(X; Y) = H(X) + H(Y) − H(X, Y)
           = Σ p(x,y) log [p(x,y) / (p(x) p(y))]

Estimated via KNN (Kraskov Algorithm 1):
  MI ≈ ψ(k) − ⟨ψ(nₓ + 1) + ψ(n_y + 1)⟩ + ψ(N)

**Transfer Entropy** (Schreiber 2000):
  TE(X→Y) = H(Yₜ₊₁ | Yₜ) − H(Yₜ₊₁ | Yₜ, Xₜ)
           = MI(Yₜ₊₁ ; Xₜ | Yₜ)

This is the information that *past X* provides about *future Y*
beyond what *past Y* already provides.

References
----------
  - Shannon CE (1948). "A mathematical theory of communication."
  - Schreiber T (2000). "Measuring information transfer."
    Physical Review Letters, 85(2).
  - Kraskov A, Stögbauer H, Grassberger P (2004). "Estimating
    mutual information." Physical Review E, 69(6).
  - Cover TM, Thomas JA (2006). Elements of Information Theory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import digamma
from scipy.spatial import KDTree

from speechscore.models.schemas import WindowMetrics, TranscriptionResult

logger = logging.getLogger(__name__)

# ── Algorithm parameters ────────────────────────────────────────
_DEFAULT_K = 3           # k for KNN-based MI estimation (Kraskov 2004)
_MIN_SERIES_LEN = 10     # minimum data points for reliable MI
_TE_LAG = 1              # time lag for transfer entropy


# ── Data classes ────────────────────────────────────────────────

@dataclass
class ChannelPairInfo:
    """Information-theoretic metrics for a pair of speech channels."""
    channel_x: str = ""
    channel_y: str = ""
    mutual_information: float = 0.0       # MI(X; Y) in nats
    normalised_mi: float = 0.0            # MI / min(H(X), H(Y)) ∈ [0, 1]
    transfer_entropy_x_to_y: float = 0.0  # TE(X → Y) in nats
    transfer_entropy_y_to_x: float = 0.0  # TE(Y → X) in nats
    dominant_direction: str = "none"      # "x→y" | "y→x" | "bidirectional" | "none"
    coupling_strength: str = "none"       # "strong" | "moderate" | "weak" | "none"
    series_length: int = 0


@dataclass
class InfoTheoreticCoherenceResult:
    """Complete information-theoretic coherence analysis."""
    channel_pairs: list[ChannelPairInfo] = field(default_factory=list)

    # Composite scores (0–100)
    nonlinear_coherence: float = 50.0     # from normalised MI
    directional_flow: float = 50.0        # from TE asymmetry
    composite_it_coherence: float = 50.0  # weighted combination

    interpretation: str = ""
    k_neighbours: int = _DEFAULT_K


# ────────────────────────────────────────────────────────────────
# Core algorithms
# ────────────────────────────────────────────────────────────────

def _add_noise(x: np.ndarray, scale: float = 1e-10) -> np.ndarray:
    """Add tiny noise to break ties in KNN distance computation."""
    return x + np.random.default_rng(42).normal(0, scale, size=x.shape)


def knn_entropy(x: np.ndarray, k: int = _DEFAULT_K) -> float:
    """
    Estimate differential entropy H(X) using KNN distances.

    Uses the Kozachenko-Leonenko estimator:
      H(X) ≈ ψ(N) − ψ(k) + d·(1/N) Σᵢ log(2εᵢ)
    where εᵢ = distance to k-th nearest neighbour, d = dimensionality.

    Parameters
    ----------
    x : (N, d) array or (N,) 1-D array
    k : number of nearest neighbours

    Returns
    -------
    float: estimated entropy in nats
    """
    x = np.atleast_2d(x)
    if x.shape[0] == 1:
        x = x.T  # ensure (N, d) shape

    N, d = x.shape
    if N <= k:
        return 0.0

    x = _add_noise(x)
    tree = KDTree(x)
    # Query k+1 neighbours (first is the point itself at distance 0)
    dists, _ = tree.query(x, k=k + 1)
    # Distance to k-th neighbour (index k, since index 0 is self)
    eps = dists[:, k]

    # Avoid log(0) for identical points
    eps = np.maximum(eps, 1e-15)

    # Kozachenko-Leonenko estimator
    h = digamma(N) - digamma(k) + d * np.mean(np.log(2 * eps))
    return float(h)


def knn_mutual_information(x: np.ndarray, y: np.ndarray,
                           k: int = _DEFAULT_K) -> float:
    """
    Estimate Mutual Information MI(X; Y) using KNN method.

    Uses Kraskov Algorithm 1 (KSG estimator):
      MI(X;Y) ≈ ψ(k) − ⟨ψ(nₓ + 1) + ψ(n_y + 1)⟩ + ψ(N)

    where nₓ, n_y are the number of points within the ε-ball in
    the marginal spaces, and ε is the distance to the k-th neighbour
    in the joint space.

    Parameters
    ----------
    x : (N,) array for channel X
    y : (N,) array for channel Y
    k : number of neighbours

    Returns
    -------
    float: estimated MI in nats (≥ 0)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    # Truncate to the shorter series
    n = min(len(x), len(y))
    x = x[:n].reshape(-1, 1)
    y = y[:n].reshape(-1, 1)
    N = n

    if N <= k + 1:
        return 0.0

    x = _add_noise(x)
    y = _add_noise(y)

    # Joint space
    xy = np.hstack([x, y])
    tree_xy = KDTree(xy)

    # Find k-th neighbour distance in joint space (Chebyshev/max norm)
    # KSG uses Chebyshev distance
    dists_xy, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = dists_xy[:, k]  # k-th neighbour distance for each point

    # Count neighbours in marginal spaces within eps
    tree_x = KDTree(x)
    tree_y = KDTree(y)

    nx = np.zeros(N, dtype=int)
    ny = np.zeros(N, dtype=int)

    for i in range(N):
        # Count points within eps in each marginal (excluding self)
        nx[i] = len(tree_x.query_ball_point(x[i], eps[i], p=np.inf)) - 1
        ny[i] = len(tree_y.query_ball_point(y[i], eps[i], p=np.inf)) - 1

    # KSG Algorithm 1
    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(N)

    return float(max(0.0, mi))  # MI ≥ 0 by definition


def transfer_entropy(source: np.ndarray, target: np.ndarray,
                     lag: int = _TE_LAG, k: int = _DEFAULT_K) -> float:
    """
    Estimate Transfer Entropy TE(source → target).

    TE(X→Y) = MI(Y_{t+1} ; X_t | Y_t)
            = H(Y_{t+1}, Y_t) + H(X_t, Y_t) − H(Y_t) − H(Y_{t+1}, Y_t, X_t)

    This measures how much the past of X reduces uncertainty about
    the future of Y, beyond what the past of Y already tells us.

    Parameters
    ----------
    source : (N,) array — the "cause" channel
    target : (N,) array — the "effect" channel
    lag    : time lag for conditioning
    k      : number of neighbours for KNN estimation

    Returns
    -------
    float: TE in nats (≥ 0)
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    N = len(source)
    if N < lag + k + 3:
        return 0.0

    # Create lagged variables
    # Y_{t+1} = target[lag:]
    # Y_t     = target[:-lag]
    # X_t     = source[:-lag]
    y_future = target[lag:].reshape(-1, 1)
    y_past = target[:-lag].reshape(-1, 1)
    x_past = source[:-lag].reshape(-1, 1)

    n = len(y_future)
    if n < k + 2:
        return 0.0

    # TE = H(Y_{t+1}, Y_t) + H(X_t, Y_t) - H(Y_t) - H(Y_{t+1}, Y_t, X_t)
    y_future = _add_noise(y_future)
    y_past = _add_noise(y_past)
    x_past = _add_noise(x_past)

    h_yfut_ypast = knn_entropy(np.hstack([y_future, y_past]), k=k)
    h_xpast_ypast = knn_entropy(np.hstack([x_past, y_past]), k=k)
    h_ypast = knn_entropy(y_past, k=k)
    h_all = knn_entropy(np.hstack([y_future, y_past, x_past]), k=k)

    te = h_yfut_ypast + h_xpast_ypast - h_ypast - h_all
    return float(max(0.0, te))


# ────────────────────────────────────────────────────────────────
# VADER sentiment extraction (per-window)
# ────────────────────────────────────────────────────────────────

def _extract_per_window_sentiment(windows: list[WindowMetrics]
                                  ) -> list[float]:
    """
    Compute VADER compound sentiment for each window's transcript.

    Returns list of sentiment values ∈ [−1, +1].
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        sentiments = []
        for w in windows:
            text = w.transcript.strip() if w.transcript else ""
            if text:
                sentiments.append(analyzer.polarity_scores(text)["compound"])
            else:
                sentiments.append(0.0)
        return sentiments
    except ImportError:
        logger.warning("vaderSentiment not available; returning zeros")
        return [0.0] * len(windows)


# ────────────────────────────────────────────────────────────────
# Main analyzer
# ────────────────────────────────────────────────────────────────

class InfoTheoreticCoherenceAnalyzer:
    """
    Information-Theoretic Cross-Modal Coherence Analysis.

    Analyses three channel pairs:
      1. **Energy ↔ Sentiment** — does vocal force track emotional content?
      2. **Pitch variability ↔ Speech rate** — are prosodic dimensions coupled?
      3. **Pause frequency ↔ Complexity** — do pauses align with cognitive demands?

    For each pair, computes:
      - Mutual Information (MI) — total statistical dependency
      - Normalised MI — coupling strength ∈ [0, 1]
      - Transfer Entropy (TE) — directional information flow

    Usage::

        analyzer = InfoTheoreticCoherenceAnalyzer()
        result = analyzer.analyze(window_metrics)
    """

    def __init__(self, k: int = _DEFAULT_K) -> None:
        self.k = k

    def analyze(self, windows: list[WindowMetrics]
                ) -> InfoTheoreticCoherenceResult:
        """
        Compute IT coherence for all channel pairs.

        Parameters
        ----------
        windows : per-window metrics from Phase 1.

        Returns
        -------
        InfoTheoreticCoherenceResult
        """
        if len(windows) < _MIN_SERIES_LEN:
            logger.warning(
                "IT Coherence: only %d windows (need %d) — returning default",
                len(windows), _MIN_SERIES_LEN,
            )
            return InfoTheoreticCoherenceResult(
                interpretation="Insufficient windows for information-theoretic analysis.",
            )

        # Extract per-window series
        energy = np.array([w.rms_mean or 0.0 for w in windows])
        pitch_var = np.array([w.pitch_std or 0.0 for w in windows])
        speech_rate = np.array([w.speech_rate_wpm or 0.0 for w in windows])
        pause_freq = np.array([w.pause_frequency_per_min or 0.0 for w in windows])
        sentiment = np.array(_extract_per_window_sentiment(windows))

        # Estimate per-window sentence complexity from word count as proxy
        # (More words in a fixed window → denser, more complex content)
        word_density = np.array([w.word_count for w in windows], dtype=np.float64)

        # Define channel pairs
        pairs_def = [
            ("energy", "sentiment", energy, sentiment),
            ("pitch_variability", "speech_rate", pitch_var, speech_rate),
            ("pause_frequency", "word_density", pause_freq, word_density),
        ]

        pair_results: list[ChannelPairInfo] = []
        nmi_values: list[float] = []
        te_asymmetries: list[float] = []

        for name_x, name_y, x, y in pairs_def:
            # Z-normalise for numerical stability
            x_z = self._z_normalise(x)
            y_z = self._z_normalise(y)

            # Mutual Information
            mi = knn_mutual_information(x_z, y_z, k=self.k)

            # Normalised MI: MI / min(H(X), H(Y))
            hx = knn_entropy(x_z.reshape(-1, 1), k=self.k)
            hy = knn_entropy(y_z.reshape(-1, 1), k=self.k)
            min_h = min(abs(hx), abs(hy))
            nmi = mi / min_h if min_h > 1e-6 else 0.0
            nmi = float(np.clip(nmi, 0.0, 1.0))

            # Transfer Entropy (both directions)
            te_xy = transfer_entropy(x_z, y_z, k=self.k)
            te_yx = transfer_entropy(y_z, x_z, k=self.k)

            # Determine dominant direction
            te_diff = te_xy - te_yx
            if abs(te_diff) < 0.01:
                direction = "bidirectional"
            elif te_xy > te_yx:
                direction = f"{name_x}→{name_y}"
            else:
                direction = f"{name_y}→{name_x}"

            # Coupling strength from NMI
            if nmi > 0.5:
                coupling = "strong"
            elif nmi > 0.2:
                coupling = "moderate"
            elif nmi > 0.05:
                coupling = "weak"
            else:
                coupling = "none"

            pair_results.append(ChannelPairInfo(
                channel_x=name_x,
                channel_y=name_y,
                mutual_information=round(mi, 4),
                normalised_mi=round(nmi, 4),
                transfer_entropy_x_to_y=round(te_xy, 4),
                transfer_entropy_y_to_x=round(te_yx, 4),
                dominant_direction=direction,
                coupling_strength=coupling,
                series_length=len(x),
            ))
            nmi_values.append(nmi)

            # TE asymmetry as a signal of structured information flow
            max_te = max(te_xy, te_yx)
            te_asymmetries.append(max_te)

        # Composite scores
        # NMI → coherence score: higher NMI = more cross-modal coupling (good)
        # Weighted: energy-sentiment 40%, pitch-rate 35%, pause-complexity 25%
        w = [0.40, 0.35, 0.25]

        nonlinear_coherence = sum(
            wi * self._nmi_to_score(nmi)
            for wi, nmi in zip(w, nmi_values)
        )

        # TE → directional flow score: any TE > 0 is structured flow (good)
        directional_flow = sum(
            wi * self._te_to_score(te)
            for wi, te in zip(w, te_asymmetries)
        )

        composite = 0.60 * nonlinear_coherence + 0.40 * directional_flow

        interpretation = self._interpret(composite, pair_results)

        return InfoTheoreticCoherenceResult(
            channel_pairs=pair_results,
            nonlinear_coherence=round(nonlinear_coherence, 1),
            directional_flow=round(directional_flow, 1),
            composite_it_coherence=round(composite, 1),
            interpretation=interpretation,
            k_neighbours=self.k,
        )

    @staticmethod
    def _z_normalise(x: np.ndarray) -> np.ndarray:
        """Z-normalise for numerical stability."""
        sd = np.std(x)
        if sd < 1e-10:
            return x - np.mean(x)
        return (x - np.mean(x)) / sd

    @staticmethod
    def _nmi_to_score(nmi: float) -> float:
        """
        Map normalised MI to [0, 100].

        NMI ∈ [0, 1]. Higher = more coupled (better for coherence).
        Uses a saturating curve: score = 100 × (1 − exp(−4 × NMI))
        """
        return float(100.0 * (1.0 - np.exp(-4.0 * nmi)))

    @staticmethod
    def _te_to_score(te: float) -> float:
        """
        Map transfer entropy to [0, 100].

        TE ≥ 0. Higher = more directional information flow (structured).
        Uses: score = 100 × (1 − exp(−5 × TE))
        """
        return float(100.0 * (1.0 - np.exp(-5.0 * te)))

    @staticmethod
    def _interpret(composite: float,
                   pairs: list[ChannelPairInfo]) -> str:
        """Generate human-readable interpretation."""
        parts = []

        if composite >= 65:
            parts.append("Cross-modal coherence is strong — speech content and delivery are well-aligned.")
        elif composite >= 40:
            parts.append("Cross-modal coherence is moderate — some alignment between content and delivery.")
        else:
            parts.append("Cross-modal coherence is weak — speech content and delivery show limited coupling.")

        # Highlight strongest coupling
        strongest = max(pairs, key=lambda p: p.normalised_mi)
        if strongest.normalised_mi > 0.1:
            parts.append(
                f"Strongest coupling: {strongest.channel_x} ↔ {strongest.channel_y} "
                f"(NMI={strongest.normalised_mi:.2f}, {strongest.coupling_strength})."
            )

        # Highlight directional flow
        for p in pairs:
            if p.dominant_direction not in ("none", "bidirectional"):
                te_max = max(p.transfer_entropy_x_to_y, p.transfer_entropy_y_to_x)
                if te_max > 0.05:
                    parts.append(
                        f"Information flows {p.dominant_direction} "
                        f"(TE={te_max:.3f})."
                    )

        return " ".join(parts)
