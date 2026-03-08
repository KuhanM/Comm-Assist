"""
SpeechScore 2.0 — Temporal Dynamics Analyzer  ⭐ NOVEL

Analyses how communication metrics *evolve over time* rather than
computing static aggregates.  This is the primary novel contribution
of SpeechScore 2.0.

Metrics:
  1. Confidence Trajectory   — regression slope of pitch stability
  2. Fluency Warmup Index    — change-point detection on speech rate
  3. Fatigue Detection Score  — first-half vs second-half t-test
  4. Engagement Arc Score    — energy curve vs ideal narrative arc

All methods consume the ``list[WindowMetrics]`` produced by Phase 1
and return structured ``TemporalMetrics``.

References (for paper):
  - Confidence/F0 stability: Mixdorff et al. (2018) INTERSPEECH
  - Narrative arc theory: Vonnegut (1995), Reagan et al. (2016) EPJ
  - Fatigue in speech: Lefter et al. (2021) Speech Communication
  - Change-point detection: Killick et al. (2012) JASA — PELT algorithm
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import stats

from speechscore.models.schemas import (
    WindowMetrics,
    TemporalMetrics,
    ConfidenceTrajectory,
    WarmupIndex,
    FatigueDetection,
    EngagementArc,
)

logger = logging.getLogger(__name__)

# Minimum windows required for meaningful temporal analysis.
# With 10 s window / 5 s hop, this requires ~20 s of audio.
_MIN_WINDOWS = 4

# Significance threshold for statistical tests
_ALPHA = 0.05

# Effect size thresholds (Cohen's d)
_SMALL_EFFECT = 0.2
_MEDIUM_EFFECT = 0.5


class TemporalAnalyzer:
    """
    Temporal dynamics feature extractor.

    Operates on the per-window time series from Phase 1.
    Each method is independent and statistically grounded.
    """

    def analyze(self, windows: list[WindowMetrics]) -> TemporalMetrics:
        """
        Run all four temporal analyses on the window sequence.

        Parameters
        ----------
        windows : list of WindowMetrics from Phase 1 pipeline.

        Returns
        -------
        TemporalMetrics with all four sub-metrics populated.
        """
        if len(windows) < _MIN_WINDOWS:
            logger.warning(
                "Only %d windows — need ≥%d for temporal analysis. "
                "Returning defaults.",
                len(windows),
                _MIN_WINDOWS,
            )
            return TemporalMetrics()

        ct = self._confidence_trajectory(windows)
        wi = self._warmup_index(windows)
        fd = self._fatigue_detection(windows)
        ea = self._engagement_arc(windows)

        return TemporalMetrics(
            confidence_trajectory=ct,
            warmup_index=wi,
            fatigue_detection=fd,
            engagement_arc=ea,
        )

    # ────────────────────────────────────────────────────────────
    # 1. CONFIDENCE TRAJECTORY
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _confidence_trajectory(windows: list[WindowMetrics]) -> ConfidenceTrajectory:
        """
        Compute the linear trend of pitch stability (F0 SD) over time.

        Insight: lower F0 SD → more controlled voice → higher confidence.
        A *negative* slope means pitch is stabilising → gaining confidence.

        Method:
          1. Extract pitch_std per window
          2. Fit OLS regression: pitch_std ~ window_index
          3. Report slope, R², p-value
          4. Classify direction based on slope sign + significance
        """
        pitch_stds = []
        for w in windows:
            val = w.pitch_std if w.pitch_std is not None else 0.0
            pitch_stds.append(val)

        x = np.arange(len(pitch_stds), dtype=np.float64)
        y = np.array(pitch_stds, dtype=np.float64)

        # guard: all-zero or constant pitch
        if np.std(y) < 1e-6:
            return ConfidenceTrajectory(
                slope=0.0,
                r_squared=0.0,
                p_value=1.0,
                direction="stable",
                interpretation="Pitch stability was constant throughout.",
                per_window_pitch_std=pitch_stds,
            )

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2

        # direction classification
        if p_value < _ALPHA:
            if slope < 0:
                direction = "increasing"      # confidence increasing (F0 SD falling)
                interpretation = (
                    f"Speaker's pitch stability improved over time "
                    f"(slope={slope:.2f} Hz/window, p={p_value:.3f}), "
                    f"indicating increasing confidence."
                )
            else:
                direction = "decreasing"      # confidence decreasing
                interpretation = (
                    f"Speaker's pitch became less stable over time "
                    f"(slope=+{slope:.2f} Hz/window, p={p_value:.3f}), "
                    f"indicating decreasing confidence or increasing stress."
                )
        else:
            direction = "stable"
            interpretation = (
                f"No significant trend in pitch stability "
                f"(slope={slope:.2f}, p={p_value:.3f}). "
                f"Confidence level was consistent."
            )

        return ConfidenceTrajectory(
            slope=round(float(slope), 4),
            r_squared=round(float(r_squared), 4),
            p_value=round(float(p_value), 4),
            direction=direction,
            interpretation=interpretation,
            per_window_pitch_std=[round(v, 2) for v in pitch_stds],
        )

    # ────────────────────────────────────────────────────────────
    # 2. FLUENCY WARMUP INDEX
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _warmup_index(windows: list[WindowMetrics]) -> WarmupIndex:
        """
        Detect the point where the speaker settles into a stable
        speech rate, using CUSUM-based change-point detection.

        Method:
          1. Extract speech_rate_wpm per window
          2. Compute cumulative sum of deviations from the overall mean
          3. The peak of the CUSUM indicates the change-point
          4. If the pre-change-point mean is significantly different
             from the post-change-point mean → warmup detected
          5. Warmup index = start_time of the change-point window
        """
        rates = []
        for w in windows:
            val = w.speech_rate_wpm if w.speech_rate_wpm is not None else 0.0
            rates.append(val)

        rates_arr = np.array(rates, dtype=np.float64)

        # CUSUM change-point detection
        # The cumulative sum of deviations from the mean peaks at the
        # point where the distribution shifts.
        overall_mean = np.mean(rates_arr)
        cusum = np.cumsum(rates_arr - overall_mean)

        # change point = index where |CUSUM| is maximised
        cp_idx = int(np.argmax(np.abs(cusum)))

        # guard: change point at the very edge is meaningless
        if cp_idx < 1 or cp_idx >= len(rates) - 1:
            return WarmupIndex(
                warmup_seconds=0.0,
                warmup_window=0,
                pre_warmup_mean_wpm=round(float(overall_mean), 1),
                post_warmup_mean_wpm=round(float(overall_mean), 1),
                change_point_detected=False,
                speech_rate_trajectory=[round(r, 1) for r in rates],
            )

        pre_mean = float(np.mean(rates_arr[:cp_idx]))
        post_mean = float(np.mean(rates_arr[cp_idx:]))

        # statistical test: are pre and post segments different?
        if cp_idx >= 2 and len(rates_arr[cp_idx:]) >= 2:
            _, p_val = stats.ttest_ind(
                rates_arr[:cp_idx],
                rates_arr[cp_idx:],
                equal_var=False,
            )
            detected = p_val < _ALPHA
        else:
            detected = abs(pre_mean - post_mean) > np.std(rates_arr)

        warmup_time = windows[cp_idx].start_time

        return WarmupIndex(
            warmup_seconds=round(warmup_time, 1),
            warmup_window=cp_idx,
            pre_warmup_mean_wpm=round(pre_mean, 1),
            post_warmup_mean_wpm=round(post_mean, 1),
            change_point_detected=detected,
            speech_rate_trajectory=[round(r, 1) for r in rates],
        )

    # ────────────────────────────────────────────────────────────
    # 3. FATIGUE DETECTION
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _fatigue_detection(windows: list[WindowMetrics]) -> FatigueDetection:
        """
        Compare first-half vs second-half metric distributions using
        Welch's t-test and Cohen's d effect size.

        Fatigue indicators (second half vs first half):
          - Pause frequency INCREASES
          - Phonation ratio DECREASES
          - ASR confidence DECREASES (speech becomes less clear)
          - Speech rate CHANGES (either direction = instability)

        The composite fatigue score is the mean normalised effect size
        across degraded metrics, scaled to [0, 100].
        """
        n = len(windows)
        mid = n // 2

        first_half = windows[:mid]
        second_half = windows[mid:]

        # metrics to test, with their expected fatigue direction:
        #   "increase" = higher in 2nd half is bad
        #   "decrease" = lower in 2nd half is bad
        metric_defs = {
            "pause_frequency_per_min": {
                "attr": "pause_frequency_per_min",
                "fatigue_dir": "increase",
            },
            "phonation_ratio": {
                "attr": "phonation_ratio",
                "fatigue_dir": "decrease",
            },
            "asr_confidence": {
                "attr": "asr_confidence",
                "fatigue_dir": "decrease",
            },
            "speech_rate_wpm": {
                "attr": "speech_rate_wpm",
                "fatigue_dir": "change",  # either direction
            },
        }

        first_means: dict = {}
        second_means: dict = {}
        degraded: list[str] = []
        details: dict = {}
        effect_sizes: list[float] = []

        for name, mdef in metric_defs.items():
            attr = mdef["attr"]
            fdir = mdef["fatigue_dir"]

            arr1 = np.array([
                getattr(w, attr) or 0.0 for w in first_half
            ], dtype=np.float64)
            arr2 = np.array([
                getattr(w, attr) or 0.0 for w in second_half
            ], dtype=np.float64)

            m1, m2 = float(np.mean(arr1)), float(np.mean(arr2))
            first_means[name] = round(m1, 3)
            second_means[name] = round(m2, 3)

            # Welch's t-test
            if np.std(arr1) > 1e-8 or np.std(arr2) > 1e-8:
                t_stat, p_val = stats.ttest_ind(arr1, arr2, equal_var=False)
            else:
                # Both halves are constant — compare means directly.
                # If means differ at all the effect is "infinite" (no noise).
                if abs(m1 - m2) > 1e-8:
                    t_stat, p_val = float("inf"), 0.0
                else:
                    t_stat, p_val = 0.0, 1.0

            # Cohen's d effect size
            pooled_std = np.sqrt(
                (np.std(arr1, ddof=1) ** 2 + np.std(arr2, ddof=1) ** 2) / 2
            )
            if pooled_std > 1e-8:
                cohens_d = (m2 - m1) / pooled_std
            else:
                # zero variance in both halves — use the mean difference
                # directly as a large effect indicator
                cohens_d = float(np.sign(m2 - m1)) * 10.0 if abs(m2 - m1) > 1e-8 else 0.0

            # Determine if this metric shows fatigue.
            # Primary: significant p-value + effect size.
            # Fallback: if sample size is small (< 15 per half), accept
            # medium+ effect size alone (d > 0.5) since t-test lacks power.
            is_degraded = False
            small_sample = len(arr1) < 15

            if fdir == "increase":
                if cohens_d > _SMALL_EFFECT and p_val < _ALPHA:
                    is_degraded = True
                elif small_sample and cohens_d > _MEDIUM_EFFECT:
                    is_degraded = True
            elif fdir == "decrease":
                if cohens_d < -_SMALL_EFFECT and p_val < _ALPHA:
                    is_degraded = True
                elif small_sample and cohens_d < -_MEDIUM_EFFECT:
                    is_degraded = True
            elif fdir == "change":
                if abs(cohens_d) > _MEDIUM_EFFECT and p_val < _ALPHA:
                    is_degraded = True
                elif small_sample and abs(cohens_d) > _MEDIUM_EFFECT:
                    is_degraded = True

            if is_degraded:
                degraded.append(name)
                effect_sizes.append(abs(cohens_d))

            details[name] = {
                "t_statistic": round(float(t_stat), 3),
                "p_value": round(float(p_val), 4),
                "cohens_d": round(float(cohens_d), 3),
                "degraded": is_degraded,
            }

        # composite fatigue score: mean |d| of degraded metrics, scaled
        if effect_sizes:
            # scale: d=0.2 → ~20, d=0.8 → ~80, capped at 100
            raw = float(np.mean(effect_sizes))
            fatigue_score = min(100.0, raw * 100.0)
        else:
            fatigue_score = 0.0

        return FatigueDetection(
            fatigue_score=round(fatigue_score, 1),
            significant=len(degraded) > 0,
            first_half_means=first_means,
            second_half_means=second_means,
            degraded_metrics=degraded,
            metric_details=details,
        )

    # ────────────────────────────────────────────────────────────
    # 4. ENGAGEMENT ARC
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _engagement_arc(windows: list[WindowMetrics]) -> EngagementArc:
        """
        Compare the speaker's energy trajectory against an ideal
        narrative arc template.

        Ideal arc (based on presentation science):
          - Strong opening  (energy ~0.85)
          - Slight dip      (content exposition ~0.60)
          - Build-up        (rising towards climax ~0.80)
          - Strong close    (conclusion emphasis ~0.90)

        Method:
          1. Extract RMS energy per window → normalise to [0, 1]
          2. Generate ideal template of same length
          3. Compute Pearson correlation
          4. Classify arc shape from actual trajectory
          5. Score = |r| × 100 if corr is positive, penalise otherwise
        """
        # Filter out unreliable windows (e.g. trailing short segments)
        reliable_w = [w for w in windows if w.reliable]
        if len(reliable_w) >= 4:
            windows = reliable_w

        # extract energy trajectory
        energies = []
        for w in windows:
            val = w.rms_mean if w.rms_mean is not None else 0.0
            energies.append(val)

        energies_arr = np.array(energies, dtype=np.float64)

        # normalise to [0, 1]
        e_min, e_max = np.min(energies_arr), np.max(energies_arr)
        if e_max - e_min > 1e-10:
            norm = (energies_arr - e_min) / (e_max - e_min)
        else:
            norm = np.zeros_like(energies_arr)

        # generate ideal arc template (same length)
        # Piecewise-linear template based on presentation science:
        #   0%–10%  : strong opening  (0.85)
        #   10%–35% : dip to content  (0.85 → 0.55)
        #   35%–70% : build-up        (0.55 → 0.80)
        #   70%–90% : climax plateau  (0.80 → 0.85)
        #   90%–100%: strong close     (0.85 → 0.90)
        n = len(norm)
        ideal = np.zeros(n, dtype=np.float64)
        for i in range(n):
            p = i / max(n - 1, 1)   # position in [0, 1]
            if p <= 0.10:
                ideal[i] = 0.85
            elif p <= 0.35:
                ideal[i] = 0.85 - (p - 0.10) / 0.25 * 0.30   # 0.85 → 0.55
            elif p <= 0.70:
                ideal[i] = 0.55 + (p - 0.35) / 0.35 * 0.25   # 0.55 → 0.80
            elif p <= 0.90:
                ideal[i] = 0.80 + (p - 0.70) / 0.20 * 0.05   # 0.80 → 0.85
            else:
                ideal[i] = 0.85 + (p - 0.90) / 0.10 * 0.05   # 0.85 → 0.90
        # normalise ideal to [0, 1]
        ideal = (ideal - ideal.min()) / (ideal.max() - ideal.min() + 1e-10)

        # Pearson correlation
        if np.std(norm) > 1e-8:
            r_val, p_val = stats.pearsonr(norm, ideal)
        else:
            r_val, p_val = 0.0, 1.0

        # classify actual arc shape from the trajectory
        shape = _classify_arc_shape(norm)

        # ── Multi-factor scoring ──
        # Base score from arc shape + bonus from template correlation.
        # Rewards good delivery dynamics even when exact template
        # doesn't match, while still rewarding strong correlation.
        _shape_base = {
            "ideal": 75, "u-shaped": 55, "rising": 50,
            "variable": 35, "declining": 20, "flat": 25,
        }
        base = _shape_base.get(shape, 35)
        corr_bonus = max(0.0, float(r_val)) * 25.0   # +0 to +25
        score = max(0.0, min(100.0, base + corr_bonus))

        return EngagementArc(
            score=round(score, 1),
            correlation=round(float(r_val), 4),
            p_value=round(float(p_val), 4),
            shape=shape,
            energy_trajectory=[round(float(v), 4) for v in norm],
            ideal_template=[round(float(v), 4) for v in ideal],
        )


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _classify_arc_shape(norm: np.ndarray) -> str:
    """
    Classify the normalised energy trajectory into a named shape.

    Categories:
      "ideal"       — strong open, dip, strong close  (inverted U or wave)
      "u-shaped"    — weak middle, strong edges
      "declining"   — energy decreases over time
      "rising"      — energy increases over time
      "flat"        — minimal variation
    """
    n = len(norm)
    if n < 4:
        return "flat"

    third = n // 3
    start_mean = float(np.mean(norm[:third]))
    mid_mean = float(np.mean(norm[third:2 * third]))
    end_mean = float(np.mean(norm[2 * third:]))

    variation = float(np.std(norm))

    # flat: very low variation
    if variation < 0.10:
        return "flat"

    # linear trend
    x = np.arange(n, dtype=np.float64)
    slope, _, _, _, _ = stats.linregress(x, norm)

    if slope < -0.02 and end_mean < start_mean - 0.15:
        return "declining"

    if slope > 0.02 and end_mean > start_mean + 0.15:
        return "rising"

    # U-shaped: edges higher than middle
    if mid_mean < start_mean - 0.10 and mid_mean < end_mean - 0.10:
        return "u-shaped"

    # ideal: strong open AND close, balanced edges, rich dynamics
    if (start_mean > 0.55 and end_mean > 0.55
            and abs(start_mean - end_mean) < 0.20
            and variation > 0.12):
        return "ideal"

    return "variable"
