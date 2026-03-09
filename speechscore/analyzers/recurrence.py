"""
SpeechScore 2.0 — Recurrence Quantification Analysis  ⭐ NOVEL  (V2-2)

Applies **Recurrence Quantification Analysis (RQA)** to speech feature
time series to capture self-similarity, predictability, and state
transitions in a speaker's delivery — properties invisible to linear
methods (regression, correlation, t-tests).

Novelty argument
----------------
RQA was introduced by Zbilut & Webber (1992) for physiological signal
analysis.  It has been applied to EEG, EMG, postural sway, heart rate,
and financial time series — but **never to automated speech assessment
scoring**.

Prior speech-related RQA work:
  - Fusaroli & Tylén (2016) used cross-recurrence quantification
    (CRQA) for **conversational alignment** between two speakers.
  - Lancia et al. (2019) applied CRQA to turn-taking dynamics.
  - Leonardi (2012) used RQA on *text* sequences, not acoustic features.

None of these applied RQA to within-speaker prosodic dynamics for
**individual speech quality scoring**.

Our contribution:
  1. First application of RQA to windowed speech features (pitch_std,
     WPM, energy) for communication assessment.
  2. Four RQA measures — Recurrence Rate (RR), Determinism (DET),
     Laminarity (LAM), Trapping Time (TT) — each with a clear
     interpretable mapping to speech quality dimensions.
  3. Phase-space embedding with Takens' theorem applied to
     speech dynamics: delay embedding reconstructs the attractor
     geometry of a speaker's delivery pattern.

Mathematical foundation
-----------------------
Given a time series {x₁, x₂, …, xₙ}:

1. **Phase-space embedding** (Takens' theorem):
   x⃗ᵢ = (xᵢ, xᵢ₊ₜ, xᵢ₊₂ₜ, …, xᵢ₊₍ₘ₋₁₎ₜ)
   where m = embedding dimension, τ = time delay.

2. **Recurrence matrix** R:
   R_{i,j} = Θ(ε − ‖x⃗ᵢ − x⃗ⱼ‖)
   where Θ = Heaviside step function, ε = threshold.

3. **RQA measures:**
   - RR  = Σᵢ,ⱼ R_{i,j} / N²  (recurrence rate)
   - DET = Σ_diag(l≥l_min) l·P(l) / Σᵢ,ⱼ R_{i,j}  (determinism)
   - LAM = Σ_vert(v≥v_min) v·P(v) / Σⱼ Σᵢ R_{i,j}  (laminarity)
   - TT  = Σ_vert(v≥v_min) v·P(v) / Σ_vert(v≥v_min) P(v)  (trapping time)

Interpretation for speech:
  - High RR → speaker returns to similar states often (consistent)
  - High DET → delivery is *predictable* (structured, deterministic)
  - High LAM → speaker "gets stuck" in states (possible hesitation)
  - High TT → how long the speaker remains in a state (monotony risk)

References
----------
  - Zbilut JP, Webber CL Jr (1992). "Embeddings and delays as derived
    from quantification of recurrence plots." Physics Letters A.
  - Marwan N, et al. (2007). "Recurrence plots for the analysis of
    complex systems." Physics Reports, 438(5-6).
  - Webber CL, Zbilut JP (2005). "Recurrence quantification analysis
    of nonlinear dynamical systems." In: Tutorials in Contemporary
    Nonlinear Methods.
  - Takens F (1981). "Detecting strange attractors in turbulence."
    Dynamical Systems and Turbulence, Lecture Notes in Math.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist

from speechscore.models.schemas import WindowMetrics

logger = logging.getLogger(__name__)

# ── Algorithm parameters ────────────────────────────────────────
# Canonical RQA choices (Marwan et al. 2007):
_DEFAULT_EMBED_DIM = 2       # embedding dimension m
_DEFAULT_DELAY = 1           # time delay τ (1 for window-level data)
_DEFAULT_RADIUS_FRAC = 0.25  # recurrence threshold ε as fraction of max distance
_DEFAULT_L_MIN = 2           # minimum diagonal line length for DET
_DEFAULT_V_MIN = 2           # minimum vertical line length for LAM
_MIN_SERIES_LEN = 8          # minimum windows needed


# ── Data classes ────────────────────────────────────────────────

@dataclass
class ChannelRQA:
    """RQA metrics for a single speech channel."""
    channel: str = ""
    recurrence_rate: float = 0.0   # RR ∈ [0, 1]
    determinism: float = 0.0       # DET ∈ [0, 1]
    laminarity: float = 0.0        # LAM ∈ [0, 1]
    trapping_time: float = 0.0     # TT ≥ 0
    max_diagonal: int = 0          # longest diagonal line
    entropy_diagonal: float = 0.0  # Shannon entropy of diagonal distribution
    n_embedded: int = 0            # # of embedded vectors
    radius: float = 0.0           # recurrence threshold used


@dataclass
class RecurrenceResult:
    """Complete RQA analysis across all speech channels."""
    channels: list[ChannelRQA] = field(default_factory=list)

    # Composite scores (mapped to 0–100)
    predictability_score: float = 50.0    # from DET
    consistency_score: float = 50.0       # from RR
    fluidity_score: float = 50.0          # from LAM (inverted — low LAM = fluid)
    composite_rqa: float = 50.0           # weighted combination

    interpretation: str = ""
    embedding_dim: int = _DEFAULT_EMBED_DIM
    delay: int = _DEFAULT_DELAY


# ────────────────────────────────────────────────────────────────
# Core algorithms
# ────────────────────────────────────────────────────────────────

def phase_space_embed(x: np.ndarray, m: int = _DEFAULT_EMBED_DIM,
                      tau: int = _DEFAULT_DELAY) -> np.ndarray:
    """
    Reconstruct phase space via Takens' time-delay embedding.

    Parameters
    ----------
    x   : 1-D time series of length N
    m   : embedding dimension
    tau : time delay

    Returns
    -------
    2-D array of shape (N − (m−1)·τ, m), each row an embedded vector.
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    n_vecs = N - (m - 1) * tau
    if n_vecs < 2:
        return np.array([]).reshape(0, m)

    indices = np.arange(n_vecs)[:, None] + np.arange(m) * tau
    return x[indices]


def recurrence_matrix(embedded: np.ndarray,
                      radius: float | None = None,
                      radius_frac: float = _DEFAULT_RADIUS_FRAC
                      ) -> np.ndarray:
    """
    Compute the binary recurrence matrix.

    R_{i,j} = 1 if ‖x⃗ᵢ − x⃗ⱼ‖ ≤ ε, else 0.

    Uses Euclidean distance (L2 norm).

    Parameters
    ----------
    embedded    : (N, m) array of embedded state vectors.
    radius      : absolute recurrence threshold ε.
                  If None, uses radius_frac × max(pairwise distance).
    radius_frac : fraction of max pairwise distance to use as ε.

    Returns
    -------
    (N, N) binary np.ndarray.
    """
    if len(embedded) < 2:
        return np.zeros((0, 0), dtype=int)

    dist = cdist(embedded, embedded, metric="euclidean")

    if radius is None:
        max_dist = np.max(dist)
        if max_dist < 1e-12:
            # Constant series — everything recurs
            return np.ones(dist.shape, dtype=int)
        radius = radius_frac * max_dist

    R = (dist <= radius).astype(int)
    return R


def _diagonal_lines(R: np.ndarray, l_min: int = _DEFAULT_L_MIN
                    ) -> list[int]:
    """
    Extract lengths of diagonal lines from recurrence matrix.

    Diagonal lines (parallel to main diagonal, excluding the LOI)
    indicate deterministic structure: the system evolves similarly
    from similar states.

    Parameters
    ----------
    R     : (N, N) binary recurrence matrix.
    l_min : minimum line length to count.

    Returns
    -------
    List of diagonal line lengths ≥ l_min.
    """
    N = R.shape[0]
    lines: list[int] = []

    # Scan all diagonals (positive offsets only — R is symmetric)
    for offset in range(1, N):
        diag = np.diag(R, offset)
        run = 0
        for val in diag:
            if val:
                run += 1
            else:
                if run >= l_min:
                    lines.append(run)
                run = 0
        if run >= l_min:
            lines.append(run)

    return lines


def _vertical_lines(R: np.ndarray, v_min: int = _DEFAULT_V_MIN
                    ) -> list[int]:
    """
    Extract lengths of vertical lines from recurrence matrix.

    Vertical lines indicate *laminar* states: the system stays in
    a particular region of phase space (gets "trapped").

    Parameters
    ----------
    R     : (N, N) binary recurrence matrix.
    v_min : minimum line length to count.

    Returns
    -------
    List of vertical line lengths ≥ v_min.
    """
    N = R.shape[0]
    lines: list[int] = []

    for j in range(N):
        col = R[:, j]
        run = 0
        for val in col:
            if val:
                run += 1
            else:
                if run >= v_min:
                    lines.append(run)
                run = 0
        if run >= v_min:
            lines.append(run)

    return lines


def compute_rqa(x: np.ndarray,
                m: int = _DEFAULT_EMBED_DIM,
                tau: int = _DEFAULT_DELAY,
                radius: float | None = None,
                radius_frac: float = _DEFAULT_RADIUS_FRAC,
                l_min: int = _DEFAULT_L_MIN,
                v_min: int = _DEFAULT_V_MIN,
                ) -> dict[str, float]:
    """
    Compute all RQA measures for a 1-D time series.

    Parameters
    ----------
    x : 1-D array (e.g. per-window pitch_std values)
    m : embedding dimension
    tau : time delay
    radius : absolute recurrence threshold (None → auto)
    radius_frac : fraction of max distance for auto ε
    l_min : minimum diagonal line length
    v_min : minimum vertical line length

    Returns
    -------
    Dict with keys: RR, DET, LAM, TT, max_diagonal, entropy_diag,
    n_embedded, radius_used.
    """
    x = np.asarray(x, dtype=np.float64)

    embedded = phase_space_embed(x, m=m, tau=tau)
    if len(embedded) < 2:
        return {
            "RR": 0.0, "DET": 0.0, "LAM": 0.0, "TT": 0.0,
            "max_diagonal": 0, "entropy_diag": 0.0,
            "n_embedded": 0, "radius_used": 0.0,
        }

    R = recurrence_matrix(embedded, radius=radius, radius_frac=radius_frac)
    N = R.shape[0]

    # --- Recurrence Rate ---
    # Exclude main diagonal (line of identity)
    total_off_diag = N * N - N
    recurrence_pts = int(np.sum(R)) - N    # subtract diagonal
    RR = recurrence_pts / total_off_diag if total_off_diag > 0 else 0.0
    RR = max(0.0, RR)

    # --- Determinism ---
    diag_lines = _diagonal_lines(R, l_min=l_min)
    diag_pts = sum(diag_lines)  # points in diagonal structures
    DET = diag_pts / recurrence_pts if recurrence_pts > 0 else 0.0

    max_diag = max(diag_lines) if diag_lines else 0

    # Shannon entropy of diagonal line distribution
    if diag_lines:
        lengths, counts = np.unique(diag_lines, return_counts=True)
        probs = counts / counts.sum()
        entropy_diag = float(-np.sum(probs * np.log2(probs + 1e-15)))
    else:
        entropy_diag = 0.0

    # --- Laminarity & Trapping Time ---
    vert_lines = _vertical_lines(R, v_min=v_min)
    vert_pts = sum(vert_lines)
    col_sums = np.sum(R, axis=0)
    total_vert_pts = int(np.sum(col_sums))

    LAM = vert_pts / total_vert_pts if total_vert_pts > 0 else 0.0
    TT = float(np.mean(vert_lines)) if vert_lines else 0.0

    # Clamp to [0, 1] for safety
    RR = float(np.clip(RR, 0.0, 1.0))
    DET = float(np.clip(DET, 0.0, 1.0))
    LAM = float(np.clip(LAM, 0.0, 1.0))

    used_radius = radius if radius is not None else radius_frac * np.max(
        cdist(embedded, embedded, metric="euclidean"))

    return {
        "RR": round(RR, 4),
        "DET": round(DET, 4),
        "LAM": round(LAM, 4),
        "TT": round(TT, 2),
        "max_diagonal": max_diag,
        "entropy_diag": round(entropy_diag, 4),
        "n_embedded": len(embedded),
        "radius_used": round(float(used_radius), 6),
    }


# ────────────────────────────────────────────────────────────────
# Main analyzer
# ────────────────────────────────────────────────────────────────

class RecurrenceAnalyzer:
    """
    Recurrence Quantification Analysis (RQA) of speech dynamics.

    Analyses three speech channels:
      1. **Pitch variability** (F0 SD per window) — vocal control dynamics
      2. **Speech rate** (WPM per window) — pacing dynamics
      3. **Energy** (RMS per window) — delivery energy dynamics

    Each channel is embedded in phase space (Takens' theorem) and
    analysed via RQA.  The four RQA measures map to interpretable
    speech quality dimensions:

      - **Predictability** (DET) — structured, deterministic delivery
      - **Consistency** (RR) — speaker returns to similar patterns
      - **Fluidity** (1 − LAM) — avoids "getting stuck" in monotony

    Usage::

        analyzer = RecurrenceAnalyzer()
        result = analyzer.analyze(window_metrics)
    """

    def __init__(self, m: int = _DEFAULT_EMBED_DIM,
                 tau: int = _DEFAULT_DELAY,
                 radius_frac: float = _DEFAULT_RADIUS_FRAC) -> None:
        self.m = m
        self.tau = tau
        self.radius_frac = radius_frac

    def analyze(self, windows: list[WindowMetrics]) -> RecurrenceResult:
        """
        Run RQA on all speech channels.

        Parameters
        ----------
        windows : per-window metrics from Phase 1.

        Returns
        -------
        RecurrenceResult
        """
        if len(windows) < _MIN_SERIES_LEN:
            logger.warning(
                "RQA: only %d windows (need %d) — returning default",
                len(windows), _MIN_SERIES_LEN,
            )
            return RecurrenceResult(
                interpretation="Insufficient windows for recurrence analysis.",
            )

        channels_def = [
            ("pitch_variability", [w.pitch_std or 0.0 for w in windows]),
            ("speech_rate", [w.speech_rate_wpm or 0.0 for w in windows]),
            ("energy", [w.rms_mean or 0.0 for w in windows]),
        ]

        channel_results: list[ChannelRQA] = []
        all_det: list[float] = []
        all_rr: list[float] = []
        all_lam: list[float] = []

        for name, series in channels_def:
            arr = np.array(series, dtype=np.float64)
            rqa = compute_rqa(arr, m=self.m, tau=self.tau,
                              radius_frac=self.radius_frac)

            channel_results.append(ChannelRQA(
                channel=name,
                recurrence_rate=rqa["RR"],
                determinism=rqa["DET"],
                laminarity=rqa["LAM"],
                trapping_time=rqa["TT"],
                max_diagonal=rqa["max_diagonal"],
                entropy_diagonal=rqa["entropy_diag"],
                n_embedded=rqa["n_embedded"],
                radius=rqa["radius_used"],
            ))
            all_det.append(rqa["DET"])
            all_rr.append(rqa["RR"])
            all_lam.append(rqa["LAM"])

        # Composite scores (weighted: pitch 40%, rate 35%, energy 25%)
        w = [0.40, 0.35, 0.25]

        # Predictability: DET mapped to 0–100 (higher DET = more predictable)
        # Optimal: moderate-to-high DET (0.5–0.9)
        pred_raw = sum(wi * d for wi, d in zip(w, all_det))
        predictability = self._det_to_score(pred_raw)

        # Consistency: RR mapped to 0–100 via inverted-U
        # Moderate RR is best (returns to patterns without being stuck)
        cons_raw = sum(wi * r for wi, r in zip(w, all_rr))
        consistency = self._rr_to_score(cons_raw)

        # Fluidity: inverse of LAM (low laminarity = fluid transitions)
        lam_raw = sum(wi * la for wi, la in zip(w, all_lam))
        fluidity = self._lam_to_score(lam_raw)

        # Overall: balanced combination
        composite = 0.40 * predictability + 0.30 * consistency + 0.30 * fluidity

        interpretation = self._interpret(predictability, consistency,
                                         fluidity, composite)

        return RecurrenceResult(
            channels=channel_results,
            predictability_score=round(predictability, 1),
            consistency_score=round(consistency, 1),
            fluidity_score=round(fluidity, 1),
            composite_rqa=round(composite, 1),
            interpretation=interpretation,
            embedding_dim=self.m,
            delay=self.tau,
        )

    @staticmethod
    def _det_to_score(det: float) -> float:
        """
        Map determinism to [0, 100].

        DET ∈ [0, 1]. Speech interpretation:
          DET < 0.2 → chaotic, unpredictable (bad)
          DET 0.4–0.8 → structured, deterministic (good)
          DET > 0.95 → overly rigid (slightly worse)
        """
        # Piecewise linear with a plateau + slight decline
        if det <= 0.0:
            return 0.0
        elif det <= 0.3:
            return det / 0.3 * 50.0        # 0–50 linearly
        elif det <= 0.8:
            return 50.0 + (det - 0.3) / 0.5 * 50.0  # 50–100 linearly
        elif det <= 1.0:
            return max(70.0, 100.0 - (det - 0.8) / 0.2 * 30.0)  # 100→70
        return 70.0

    @staticmethod
    def _rr_to_score(rr: float) -> float:
        """
        Map recurrence rate to [0, 100] using inverted-U.

        Optimal RR for speech: 0.15–0.45.
        Too low → no recurring patterns.
        Too high → stuck in repetitive behaviour.
        """
        mu = 0.30   # optimal RR
        sigma = 0.20
        return float(100.0 * np.exp(-((rr - mu) / sigma) ** 2))

    @staticmethod
    def _lam_to_score(lam: float) -> float:
        """
        Map laminarity to fluidity score [0, 100].

        Lower LAM → higher fluidity (less "stuck" in states).
        LAM = 0 → perfect fluidity (100)
        LAM = 0.8+ → very low fluidity (≤20)
        """
        return float(max(0.0, 100.0 * (1.0 - lam * 1.2)))

    @staticmethod
    def _interpret(pred: float, cons: float, fluidity: float,
                   composite: float) -> str:
        """Generate human-readable interpretation."""
        parts = []

        if composite >= 70:
            parts.append("Speech dynamics show well-structured temporal patterns.")
        elif composite >= 45:
            parts.append("Speech dynamics show moderate temporal structure.")
        else:
            parts.append("Speech dynamics show limited temporal structure.")

        if pred >= 60:
            parts.append("Delivery is predictable and deterministic — good for audience comprehension.")
        elif pred < 35:
            parts.append("Delivery is somewhat unpredictable — consider more consistent pacing.")

        if fluidity < 40:
            parts.append("Speaker tends to linger in delivery states — consider more dynamic transitions.")

        return " ".join(parts)
