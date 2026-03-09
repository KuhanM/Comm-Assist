"""
SpeechScore 2.0 — Frame-Level Feature Extraction

Extracts high-resolution (10 ms) feature contours from raw audio
for use by the V2 nonlinear-dynamics analysers (MSE, RQA, MI/TE).

All features are returned at a uniform 10 ms hop.  This gives
N ≈ 100 × duration_seconds data points — enough for reliable
estimation of MSE (requires N > 100), RQA (requires N > 200),
and KNN-based MI/TE (requires N > 200).

Features extracted
------------------
  1. **F0 contour**  — Fundamental frequency via Praat (parselmouth).
     Voiced frames only; unvoiced gaps linearly interpolated.
     Also returned as log₂(F0) (octave scale, more perceptually
     meaningful for entropy/recurrence analysis).
  2. **RMS energy**  — Root-mean-square energy via librosa.
     Converted to dB scale: 20·log₁₀(RMS / ref).
  3. **Spectral centroid** — "brightness" of each frame (Hz).
  4. **Spectral flux** — frame-to-frame spectral change (L2 norm
     of magnitude-spectrum difference), capturing articulatory dynamics.

Down-sampling
-------------
For computational tractability (RQA's O(N²) distance matrix, MSE's
O(N²) template matching), features can be block-averaged to a target
length while preserving temporal structure.  The ``downsample``
function implements this.

References
----------
  - Boersma P, Weenink D (2024). "Praat: doing phonetics by computer."
  - McFee B, et al. (2015). "librosa: Audio and music signal analysis
    in Python."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import librosa
import parselmouth

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────
_HOP_SEC = 0.01          # 10 ms — shared with ProsodyConfig.time_step
_F0_FLOOR = 75.0         # Hz
_F0_CEILING = 500.0      # Hz
_RMS_FRAME = 2048        # ~128 ms at 16 kHz
_RMS_HOP_RAW = 160       # 10 ms at 16 kHz  (overrides the 512 in legacy code)
_CENTROID_HOP = 160      # 10 ms at 16 kHz
_FLUX_HOP = 160          # 10 ms at 16 kHz
_REF_RMS = 1e-6          # dB reference for energy


@dataclass
class FrameFeatures:
    """
    Frame-level feature contours at uniform 10 ms resolution.

    All arrays have the same length ``n_frames``.
    """
    f0: np.ndarray                # (N,) F0 in Hz (interpolated through unvoiced)
    log_f0: np.ndarray            # (N,) log₂(F0) — octave scale
    rms_energy: np.ndarray        # (N,) RMS energy (linear)
    rms_db: np.ndarray            # (N,) 20·log₁₀(RMS / ref)
    spectral_centroid: np.ndarray # (N,) spectral centroid in Hz
    spectral_flux: np.ndarray     # (N,) frame-to-frame spectral change
    voiced_mask: np.ndarray       # (N,) bool — True for originally-voiced frames
    hop_sec: float = _HOP_SEC    # seconds per frame
    sample_rate: int = 16000
    n_frames: int = 0
    duration_sec: float = 0.0


# ────────────────────────────────────────────────────────────────
# Extraction
# ────────────────────────────────────────────────────────────────

def extract_frame_features(audio: np.ndarray, sr: int = 16000) -> FrameFeatures:
    """
    Extract frame-level features from mono audio.

    Parameters
    ----------
    audio : 1-D float32/64 waveform, already at ``sr`` Hz.
    sr    : sample rate (default 16 000).

    Returns
    -------
    FrameFeatures  with all contours at 10 ms resolution.
    """
    duration = len(audio) / sr
    hop_samples = int(_HOP_SEC * sr)  # 160 at 16 kHz

    # ── 1. F0 via Praat ──────────────────────────────────────
    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    pitch_obj = snd.to_pitch(
        time_step=_HOP_SEC,
        pitch_floor=_F0_FLOOR,
        pitch_ceiling=_F0_CEILING,
    )
    f0_raw = pitch_obj.selected_array["frequency"]  # 0 = unvoiced
    voiced_mask = f0_raw > 0

    # Interpolate through unvoiced gaps
    f0_interp = _interpolate_f0(f0_raw)

    # log₂(F0) — octave scale, perceptually linear
    log_f0 = np.log2(np.maximum(f0_interp, 1.0))

    # ── 2. RMS energy ────────────────────────────────────────
    rms_raw = librosa.feature.rms(
        y=audio, frame_length=_RMS_FRAME, hop_length=hop_samples,
    )[0]
    rms_db = 20.0 * np.log10(np.maximum(rms_raw, _REF_RMS) / _REF_RMS)

    # ── 3. Spectral centroid ─────────────────────────────────
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, hop_length=hop_samples,
    )[0]

    # ── 4. Spectral flux (L2 norm of frame-to-frame STFT diff)
    S = np.abs(librosa.stft(audio, hop_length=hop_samples))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    flux = np.concatenate([[0.0], flux])  # first frame has no predecessor

    # ── Align lengths ────────────────────────────────────────
    n = min(len(f0_interp), len(rms_raw), len(centroid), len(flux))
    f0_interp = f0_interp[:n]
    log_f0 = log_f0[:n]
    voiced_mask = voiced_mask[:n]
    rms_raw = rms_raw[:n]
    rms_db = rms_db[:n]
    centroid = centroid[:n]
    flux = flux[:n]

    logger.info(
        "Frame features: %d frames (%.1f s), %.0f%% voiced, F0 %.0f–%.0f Hz",
        n, duration, 100 * voiced_mask.mean(),
        f0_interp[voiced_mask].min() if voiced_mask.any() else 0,
        f0_interp[voiced_mask].max() if voiced_mask.any() else 0,
    )

    return FrameFeatures(
        f0=f0_interp,
        log_f0=log_f0,
        rms_energy=rms_raw,
        rms_db=rms_db,
        spectral_centroid=centroid,
        spectral_flux=flux,
        voiced_mask=voiced_mask,
        hop_sec=_HOP_SEC,
        sample_rate=sr,
        n_frames=n,
        duration_sec=duration,
    )


# ────────────────────────────────────────────────────────────────
# Down-sampling (block averaging)
# ────────────────────────────────────────────────────────────────

def downsample(x: np.ndarray, target_n: int) -> np.ndarray:
    """
    Down-sample a 1-D array to ``target_n`` points by block averaging.

    This is equivalent to coarse-graining at scale τ = ⌈N / target_n⌉
    and preserves the temporal structure while reducing N for O(N²)
    algorithms (RQA distance matrix, SampEn template matching).

    Parameters
    ----------
    x        : 1-D array of length N.
    target_n : desired output length.

    Returns
    -------
    1-D array of length ``target_n`` (or shorter if N < target_n).
    """
    N = len(x)
    if N <= target_n:
        return x.copy()

    block_size = N // target_n
    n_out = N // block_size
    trimmed = x[:n_out * block_size]
    return trimmed.reshape(n_out, block_size).mean(axis=1)


# ────────────────────────────────────────────────────────────────
# Embedding parameter selection (for RQA)
# ────────────────────────────────────────────────────────────────

def optimal_delay_ami(x: np.ndarray, max_lag: int = 50) -> int:
    """
    Select optimal time delay τ via first minimum of Average Mutual
    Information (Fraser & Swinney, 1986).

    Uses histogram-based MI estimation for speed.

    Parameters
    ----------
    x       : 1-D time series (N > max_lag)
    max_lag : search up to this lag

    Returns
    -------
    int : optimal delay τ (≥ 1).

    Reference
    ---------
    Fraser AM, Swinney HL (1986). "Independent coordinates for strange
    attractors from mutual information." Physical Review A, 33(2).
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N < max_lag + 10:
        return 1

    n_bins = max(10, int(np.sqrt(N / 5)))

    ami_values = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        x1 = x[:N - lag]
        x2 = x[lag:]
        # 2D histogram → MI
        hist_2d, _, _ = np.histogram2d(x1, x2, bins=n_bins)
        pxy = hist_2d / hist_2d.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)

        # MI = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
        mask = pxy > 0
        mi = np.sum(pxy[mask] * np.log(pxy[mask] / (px[:, None] * py[None, :])[mask]))
        ami_values[lag - 1] = mi

    # Find first local minimum
    for i in range(1, len(ami_values) - 1):
        if ami_values[i] < ami_values[i - 1] and ami_values[i] <= ami_values[i + 1]:
            return i + 1  # lag is 1-indexed

    # No minimum found — use lag at steepest drop
    return int(np.argmin(ami_values)) + 1


def optimal_dimension_fnn(x: np.ndarray, tau: int,
                          max_dim: int = 10,
                          rtol: float = 15.0,
                          atol: float = 2.0) -> int:
    """
    Select optimal embedding dimension m via False Nearest Neighbours
    (Kennel, Brown & Abarbanel, 1992).

    Parameters
    ----------
    x       : 1-D time series
    tau     : time delay (from optimal_delay_ami)
    max_dim : search up to this dimension
    rtol    : distance ratio threshold for false neighbours
    atol    : absolute tolerance (ratio to attractor size)

    Returns
    -------
    int : optimal embedding dimension m (≥ 2).

    Reference
    ---------
    Kennel MB, Brown R, Abarbanel HDI (1992). "Determining embedding
    dimension for phase-space reconstruction using a geometrical
    construction on the time series." Physical Review A, 45(6).
    """
    from scipy.spatial import KDTree

    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        return 2

    for m in range(1, max_dim + 1):
        n_vecs = N - m * tau
        if n_vecs < 10:
            return max(2, m)

        # Embed at dimension m
        indices = np.arange(n_vecs)[:, None] + np.arange(m) * tau
        emb_m = x[indices]

        # Embed at dimension m+1
        n_vecs_m1 = N - (m + 1) * tau
        if n_vecs_m1 < 10:
            return max(2, m)
        indices_m1 = np.arange(n_vecs_m1)[:, None] + np.arange(m + 1) * tau
        emb_m1 = x[indices_m1]

        # Find nearest neighbour in m-dimensional space
        tree = KDTree(emb_m[:n_vecs_m1])
        dists, idxs = tree.query(emb_m[:n_vecs_m1], k=2)
        nn_dists = dists[:, 1]  # nearest other point
        nn_idxs = idxs[:, 1]

        # Check false neighbours
        n_false = 0
        n_total = 0
        for i in range(n_vecs_m1):
            j = nn_idxs[i]
            d_m = nn_dists[i]
            if d_m < 1e-12:
                continue

            # Distance in the extra (m+1)-th dimension
            d_extra = abs(emb_m1[i, m] - emb_m1[j, m])

            n_total += 1
            # Criterion 1: relative distance increase
            if d_extra / d_m > rtol:
                n_false += 1
            # Criterion 2: absolute distance vs attractor size
            elif d_extra / sigma > atol:
                n_false += 1

        fnn_ratio = n_false / n_total if n_total > 0 else 0.0

        if fnn_ratio < 0.01:  # < 1% false neighbours → good embedding
            return max(2, m)

    return max(2, max_dim)


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _interpolate_f0(f0: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate F0 through unvoiced (zero) regions.

    This is standard practice in prosody research (Xu 2005, de Jong
    & Wempe 2009) to produce a continuous contour for temporal analysis.
    Regions before the first/after the last voiced frame are
    forward/backward filled.
    """
    f0 = f0.copy().astype(np.float64)
    voiced = f0 > 0

    if not voiced.any():
        return np.full_like(f0, _F0_FLOOR)  # no speech — return floor

    indices = np.arange(len(f0))
    f0[~voiced] = np.interp(indices[~voiced], indices[voiced], f0[voiced])
    return f0
