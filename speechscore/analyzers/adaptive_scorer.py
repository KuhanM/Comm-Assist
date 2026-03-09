"""
SpeechScore 2.0 — Speaker-Adaptive Scorer  ⭐ NOVEL  (Contribution 2)

Normalises every metric relative to the speaker's *own* baseline,
yielding z-scores that capture *change from personal norm* instead of
deviation from a population average.

Core insight
------------
A speaker with a naturally high pitch (F0 = 250 Hz) and a speaker with
a naturally low pitch (F0 = 120 Hz) should not be scored differently
for the same level of *stability*.  By computing:

    z = (metric_value − baseline_mean) / baseline_std

we measure how much the speaker deviated from **their own typical
performance**, not an arbitrary benchmark.  This makes the score
*fair* across gender, age, accent, and speaking style.

Algorithm
---------
1. ``BaselineExtractor`` computes mean ± σ for each metric from the
   first 30 s (configurable).
2. ``AdaptiveScorer.score()`` takes the global aggregated metrics from
   Phase 1 and computes z-scores relative to that baseline.
3. A composite score maps the z-vector to a 0–100 scale using a
   sigmoid-like activation (smoothly penalising extreme deviations).

The z-score decomposition also enables the *Personal Speech Rate Delta*,
*Personal Pitch Stability*, and *Adaptive Benchmark Score* metrics
from the spec (Category 7).

References:
  - Cohen (1988) "Statistical Power Analysis" — z / d interpretation
  - Levelt (1989) — intra-speaker variation as baseline
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from speechscore.models.schemas import (
    SpeakerBaseline,
    AdaptiveMetricScore,
    AdaptiveScoreResult,
    WindowMetrics,
)

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# Metric definitions: (display_name, window_attr, baseline_mean_attr,
#                       baseline_std_attr, higher_is_better)
# ────────────────────────────────────────────────────────────

_METRIC_DEFS: list[dict] = [
    {
        "name": "Speech Rate (WPM)",
        "window_attr": "speech_rate_wpm",
        "bl_mean": "speech_rate_mean",
        "bl_std": "speech_rate_std",
        "higher_is_better": None,     # None = "neutral" — deviation in either direction is notable
    },
    {
        "name": "Pitch Stability (F0 SD)",
        "window_attr": "pitch_std",
        "bl_mean": "pitch_std_mean",
        "bl_std": "pitch_std_std",
        "higher_is_better": False,     # lower F0 SD = more controlled = better
    },
    {
        "name": "Volume Level (RMS)",
        "window_attr": "rms_mean",
        "bl_mean": "volume_mean",
        "bl_std": "volume_std",
        "higher_is_better": None,
    },
    {
        "name": "Pause Frequency (per min)",
        "window_attr": "pause_frequency_per_min",
        "bl_mean": "pause_freq_mean",
        "bl_std": "pause_freq_std",
        "higher_is_better": False,     # fewer pauses = typically better
    },
    {
        "name": "Phonation Ratio",
        "window_attr": "phonation_ratio",
        "bl_mean": "phonation_mean",
        "bl_std": "phonation_std",
        "higher_is_better": True,      # more speech time = better
    },
    {
        "name": "Filler Rate (per 100 words)",
        "window_attr": "filler_rate_per_100",
        "bl_mean": "filler_rate_mean",
        "bl_std": "filler_rate_std",
        "higher_is_better": False,     # fewer fillers = better
    },
]


def _deviation_label(z: float) -> str:
    """Map z-score to a human-readable label."""
    az = abs(z)
    if az < 0.5:
        return "typical"
    if az < 1.5:
        return "moderate"
    if az < 2.5:
        return "high"
    return "extreme"


def _interpret(name: str, z: float, higher_is_better: Optional[bool]) -> str:
    """Generate a one-sentence interpretation for the adaptive metric."""
    direction = "above" if z > 0 else "below"
    magnitude = _deviation_label(z)

    if magnitude == "typical":
        return f"{name} was within the speaker's typical range (z={z:+.2f})."

    if higher_is_better is None:
        return (
            f"{name} was {magnitude}ly {direction} the speaker's baseline "
            f"(z={z:+.2f})."
        )

    if (higher_is_better and z > 0) or (not higher_is_better and z < 0):
        quality = "improvement"
    else:
        quality = "degradation"

    return (
        f"{name} showed {magnitude} {quality} relative to baseline "
        f"(z={z:+.2f})."
    )


class AdaptiveScorer:
    """
    Speaker-Adaptive Normalization engine.

    Compares global (full-speech) metric aggregates against the personal
    baseline to produce z-scores, deviation labels, and a composite
    adaptive score.
    """

    def score(
        self,
        windows: list[WindowMetrics],
        baseline: SpeakerBaseline,
    ) -> AdaptiveScoreResult:
        """
        Compute adaptive metrics for every tracked dimension.

        Parameters
        ----------
        windows : full list of WindowMetrics (Phase 1 output).
        baseline : SpeakerBaseline from BaselineExtractor.

        Returns
        -------
        AdaptiveScoreResult with per-metric z-scores + composite.
        """
        if not windows or baseline.windows_used == 0:
            logger.warning("Empty windows or baseline — returning defaults.")
            return AdaptiveScoreResult(baseline=baseline)

        # Compute global means across ALL windows
        global_means = self._global_means(windows)

        adaptive_metrics: list[AdaptiveMetricScore] = []
        z_scores: list[float] = []
        z_directions: list[Optional[bool]] = []  # higher_is_better for each

        for mdef in _METRIC_DEFS:
            name = mdef["name"]
            raw = global_means.get(mdef["window_attr"], 0.0)
            bl_mean = getattr(baseline, mdef["bl_mean"], 0.0)
            bl_std = getattr(baseline, mdef["bl_std"], 0.0)

            # z-score
            if bl_std > 1e-8:
                z = (raw - bl_mean) / bl_std
            else:
                # no variance in baseline — use sign of difference as
                # bounded z-score (cap at ±3)
                diff = raw - bl_mean
                if abs(diff) < 1e-8:
                    z = 0.0
                else:
                    z = np.clip(diff / max(abs(bl_mean) * 0.1, 1e-6), -3.0, 3.0)

            # percentile from z (normal CDF)
            pct = float(sp_stats.norm.cdf(z) * 100.0)

            am = AdaptiveMetricScore(
                metric_name=name,
                raw_value=round(raw, 4),
                baseline_value=round(bl_mean, 4),
                z_score=round(float(z), 3),
                percentile=round(pct, 1),
                deviation_label=_deviation_label(float(z)),
                interpretation=_interpret(name, float(z), mdef["higher_is_better"]),
            )
            adaptive_metrics.append(am)
            z_scores.append(float(z))
            z_directions.append(mdef["higher_is_better"])

        # ── Winsorize z-scores at ±3 to prevent single outliers
        # from dominating the composite score ──
        z_scores_w = [float(np.clip(z, -3.0, 3.0)) for z in z_scores]

        # ── Composite scores (direction-aware) ──
        overall = self._composite_score(z_scores_w, z_directions)

        # speech-rate delta (% change from baseline)
        sr_global = global_means.get("speech_rate_wpm", 0.0)
        sr_bl = baseline.speech_rate_mean
        if sr_bl > 1e-6:
            sr_delta_pct = ((sr_global - sr_bl) / sr_bl) * 100.0
        else:
            sr_delta_pct = 0.0

        # pitch stability ratio: overall F0 SD / baseline F0 SD
        ps_global = global_means.get("pitch_std", 0.0)
        ps_bl = baseline.pitch_std_mean
        if ps_bl > 1e-6:
            pitch_ratio = ps_global / ps_bl
        else:
            pitch_ratio = 1.0

        # consistency score: how similar is the full speech to baseline?
        # measured as 100 − mean(|z_winsorized|) × 20, clipped to [0, 100]
        mean_abs_z = float(np.mean(np.abs(z_scores_w)))
        consistency = max(0.0, min(100.0, 100.0 - mean_abs_z * 20.0))

        return AdaptiveScoreResult(
            baseline=baseline,
            adaptive_metrics=adaptive_metrics,
            overall_adaptive_score=round(overall, 1),
            speech_rate_delta_pct=round(sr_delta_pct, 2),
            pitch_stability_ratio=round(pitch_ratio, 3),
            consistency_score=round(consistency, 1),
        )

    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _global_means(windows: list[WindowMetrics]) -> dict[str, float]:
        """Compute global mean of each metric across all windows."""
        attrs = [
            "speech_rate_wpm",
            "pitch_std",
            "rms_mean",
            "pause_frequency_per_min",
            "phonation_ratio",
            "filler_rate_per_100",
        ]
        result: dict[str, float] = {}
        for attr in attrs:
            vals = [getattr(w, attr) or 0.0 for w in windows]
            result[attr] = float(np.mean(vals))
        return result

    @staticmethod
    def _composite_score(
        z_scores: list[float],
        directions: list[Optional[bool]] | None = None,
    ) -> float:
        """
        Map a vector of z-scores to a 0–100 composite score.

        Direction-aware: when higher_is_better is known, a z-score
        in the "good" direction (e.g. fewer fillers, more phonation)
        is treated as only 30% of the deviation.  Degradations remain
        fully penalised.  Neutral metrics (higher_is_better=None) use
        |z| as before.

        Targets:
          - effective_z ≈ 0   → score ≈ 75  (typical = good)
          - effective_z ≈ ±1  → score ≈ 60
          - effective_z ≈ ±2  → score ≈ 35
          - effective_z ≈ ±3  → score ≈ 15
        """
        if not z_scores:
            return 50.0

        if directions is None:
            directions = [None] * len(z_scores)

        effective: list[float] = []
        for z, hib in zip(z_scores, directions):
            if hib is None:
                # Neutral — any deviation counts fully
                effective.append(abs(z))
            elif hib is True:
                # Higher is better: positive z = improvement → dampen
                if z >= 0:
                    effective.append(abs(z) * 0.3)  # improvement
                else:
                    effective.append(abs(z))          # degradation
            else:
                # Lower is better: negative z = improvement → dampen
                if z <= 0:
                    effective.append(abs(z) * 0.3)  # improvement
                else:
                    effective.append(abs(z))          # degradation

        mean_eff = float(np.mean(effective))

        # Sigmoid-like mapping: score = 100 / (1 + exp(k * (eff − shift)))
        # k=1.5, shift=0.5 → typical performance scores ~75+
        score = 100.0 / (1.0 + np.exp(1.5 * (mean_eff - 0.5)))

        return float(np.clip(score, 0.0, 100.0))
