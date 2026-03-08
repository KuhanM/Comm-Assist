"""
SpeechScore 2.0 — Cognitive Strain Index (CSI)  ⭐ NOVEL  (Contribution 3)

Estimates per-window cognitive load from five physiological and
linguistic strain indicators, then localises "struggle points" where
the speaker was under high cognitive load.

Novelty argument
----------------
Prior art measures disfluency counts and correlates them with
perceived effort.  Our CSI *decomposes* cognitive strain into five
weighted channels, each computed relative to the speaker's own
baseline (from Day 5), making the metric both speaker-adaptive and
multi-dimensional.  The struggle-point timeline enables *surgical*
feedback ("you struggled at 1:30–1:40 — primarily filler excess,
suggesting word-finding difficulty").

Indicators  (weights from spec §8.1)
-------------------------------------
  1. Pause excess        — 25%  — processing time needed
  2. Filler excess       — 25%  — word-finding difficulty
  3. Speech rate dev.    — 20%  — uncertainty or rushing
  4. Pitch instability   — 15%  — stress / discomfort
  5. Hesitation patterns — 15%  — planning difficulty (long pauses
     + fillers in same window = hesitation)

Each indicator is normalised to [0, 100] relative to the speaker's
baseline: 0 = at or better than baseline, 100 = extreme deviation.
The composite CSI is their weighted sum.

References:
  - Sweller (1988) "Cognitive load during problem solving"
  - Goldman-Eisler (1968) "Psycholinguistics: Pause and speech"
  - Lickley (2015) "Fluency and disfluency" — Handbook of Pragmatics
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from speechscore.models.schemas import (
    WindowMetrics,
    SpeakerBaseline,
    CognitiveStrainResult,
    StrugglePoint,
)

logger = logging.getLogger(__name__)

# ── Indicator weights (revised — added clarity_strain) ──
_WEIGHTS = {
    "pause_excess":        0.20,
    "filler_excess":       0.20,
    "speech_rate_dev":     0.15,
    "pitch_instability":   0.10,
    "hesitation_pattern":  0.15,
    "clarity_strain":      0.20,   # ASR confidence drop → articulatory strain
}

# Default struggle-point threshold (lowered from 60 → 40 for sensitivity)
_DEFAULT_THRESHOLD = 40.0

# Scaling constants — map raw deviations to 0–100.
# These are conservative so that a z=2 deviation ≈ 70 and z=3 ≈ 90.
_SCALE_K = 30.0   # sensitivity (higher = steeper curve)
_SCALE_MID = 1.5  # z-score at which indicator reaches ~50


class CognitiveAnalyzer:
    """
    Per-window Cognitive Strain Index estimator.

    Usage::

        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze(window_metrics, baseline)
    """

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold

    def analyze(
        self,
        windows: list[WindowMetrics],
        baseline: SpeakerBaseline,
    ) -> CognitiveStrainResult:
        """
        Compute CSI for every window and identify struggle points.

        Parameters
        ----------
        windows  : per-window metrics from Phase 1.
        baseline : speaker baseline from Day 5.

        Returns
        -------
        CognitiveStrainResult with per-window scores + struggle points.
        """
        if not windows:
            return CognitiveStrainResult()

        per_window_csi: list[float] = []
        struggle_points: list[StrugglePoint] = []
        indicator_accum: dict[str, list[float]] = {k: [] for k in _WEIGHTS}

        for w in windows:
            indicators = self._compute_indicators(w, baseline)

            # weighted sum → CSI for this window
            csi = sum(
                _WEIGHTS[name] * val for name, val in indicators.items()
            )
            csi = float(np.clip(csi, 0.0, 100.0))
            per_window_csi.append(round(csi, 1))

            # accumulate for global means
            for name, val in indicators.items():
                indicator_accum[name].append(val)

            # struggle-point detection
            if csi > self.threshold:
                primary = max(indicators, key=lambda k: indicators[k])
                struggle_points.append(StrugglePoint(
                    window_id=w.window_id,
                    start_time=w.start_time,
                    end_time=w.end_time,
                    csi_score=round(csi, 1),
                    primary_cause=primary,
                    strain_breakdown={
                        k: round(v, 1) for k, v in indicators.items()
                    },
                    transcript_snippet=w.transcript[:120] if w.transcript else "",
                ))

        # aggregate stats
        csi_arr = np.array(per_window_csi)
        indicator_means = {
            k: round(float(np.mean(v)), 1) for k, v in indicator_accum.items()
        }

        return CognitiveStrainResult(
            per_window_csi=per_window_csi,
            mean_csi=round(float(np.mean(csi_arr)), 1),
            max_csi=round(float(np.max(csi_arr)), 1),
            min_csi=round(float(np.min(csi_arr)), 1),
            std_csi=round(float(np.std(csi_arr)), 1),
            struggle_threshold=self.threshold,
            struggle_points=struggle_points,
            struggle_count=len(struggle_points),
            struggle_pct=round(len(struggle_points) / len(windows) * 100, 1),
            indicator_means=indicator_means,
        )

    # ────────────────────────────────────────────────────────────
    # Per-window indicator computation
    # ────────────────────────────────────────────────────────────

    def _compute_indicators(
        self,
        w: WindowMetrics,
        bl: SpeakerBaseline,
    ) -> dict[str, float]:
        """
        Compute all five strain indicators for a single window.

        Each indicator is a float in [0, 100]:
          0   = performing at or better than baseline
          100 = extreme strain
        """
        return {
            "pause_excess": self._pause_excess(w, bl),
            "filler_excess": self._filler_excess(w, bl),
            "speech_rate_dev": self._speech_rate_deviation(w, bl),
            "pitch_instability": self._pitch_instability(w, bl),
            "hesitation_pattern": self._hesitation_pattern(w, bl),
            "clarity_strain": self._clarity_strain(w, bl),
        }

    # ── 1. Pause Excess (25%) ────────────────────────────────

    @staticmethod
    def _pause_excess(w: WindowMetrics, bl: SpeakerBaseline) -> float:
        """
        How much more the speaker pauses than their baseline.

        Uses pause_frequency_per_min.  If the window's pause frequency
        exceeds the baseline mean, the excess is mapped to [0, 100]
        via a sigmoid-like activation.
        """
        val = w.pause_frequency_per_min or 0.0
        bl_mean = bl.pause_freq_mean
        bl_std = bl.pause_freq_std

        return _deviation_score(val, bl_mean, bl_std, direction="higher_is_worse")

    # ── 2. Filler Excess (25%) ───────────────────────────────

    @staticmethod
    def _filler_excess(w: WindowMetrics, bl: SpeakerBaseline) -> float:
        """
        How many more fillers per 100 words compared to baseline.
        """
        val = w.filler_rate_per_100 or 0.0
        bl_mean = bl.filler_rate_mean
        bl_std = bl.filler_rate_std

        return _deviation_score(val, bl_mean, bl_std, direction="higher_is_worse")

    # ── 3. Speech Rate Deviation (20%) ───────────────────────

    @staticmethod
    def _speech_rate_deviation(w: WindowMetrics, bl: SpeakerBaseline) -> float:
        """
        Absolute deviation of speech rate from baseline — both speeding
        up (rushing) and slowing down (struggling) contribute to strain.
        """
        val = w.speech_rate_wpm or 0.0
        bl_mean = bl.speech_rate_mean
        bl_std = bl.speech_rate_std

        return _deviation_score(val, bl_mean, bl_std, direction="any")

    # ── 4. Pitch Instability (15%) ───────────────────────────

    @staticmethod
    def _pitch_instability(w: WindowMetrics, bl: SpeakerBaseline) -> float:
        """
        Higher-than-baseline F0 SD indicates vocal stress.
        """
        val = w.pitch_std or 0.0
        bl_mean = bl.pitch_std_mean
        bl_std = bl.pitch_std_std

        return _deviation_score(val, bl_mean, bl_std, direction="higher_is_worse")

    # ── 5. Hesitation Pattern (15%) ──────────────────────────

    @staticmethod
    def _hesitation_pattern(w: WindowMetrics, bl: SpeakerBaseline) -> float:
        """
        Co-occurrence of elevated pauses AND fillers in the same window
        indicates planning difficulty (hesitation).

        Score = geometric mean of pause_excess and filler_excess,
        boosted by long max pauses.
        """
        pause_val = w.pause_frequency_per_min or 0.0
        filler_val = w.filler_rate_per_100 or 0.0

        pause_dev = _deviation_score(
            pause_val, bl.pause_freq_mean, bl.pause_freq_std,
            direction="higher_is_worse",
        )
        filler_dev = _deviation_score(
            filler_val, bl.filler_rate_mean, bl.filler_rate_std,
            direction="higher_is_worse",
        )

        # geometric mean rewards co-occurrence:
        # both elevated → high; only one elevated → moderate
        base = float(np.sqrt(max(0.0, pause_dev) * max(0.0, filler_dev)))

        # boost if there's a very long pause (> 1.5 s)
        max_pause = w.max_pause_duration or 0.0
        if max_pause > 1.5:
            boost = min(20.0, (max_pause - 1.5) * 15.0)
            base = min(100.0, base + boost)

        return round(base, 1)

    # ── 6. Clarity Strain (20%) — NEW ────────────────────────

    @staticmethod
    def _clarity_strain(w: WindowMetrics, bl: SpeakerBaseline) -> float:
        """
        ASR confidence drop relative to baseline indicates
        articulatory strain — the speaker is becoming less clear.

        Lower ASR confidence = worse, so direction is 'lower_is_worse'.
        Also adds a hard penalty when ASR drops below 0.60 (likely
        Whisper hallucination or very unclear speech).
        """
        val = w.asr_confidence or 0.0
        # ASR confidence baseline: use global mean from early windows
        # We approximate from phonation baseline (typically ~0.85-0.95)
        bl_mean = 0.90  # conservative ASR baseline
        bl_std = 0.05   # typical ASR confidence SD

        base = _deviation_score(val, bl_mean, bl_std, direction="lower_is_worse")

        # hard penalty for very low ASR confidence (< 0.60)
        if val < 0.60:
            penalty = (0.60 - val) * 200.0  # 0.49 → +22, 0.30 → +60
            base = min(100.0, base + penalty)

        return round(base, 1)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _deviation_score(
    value: float,
    bl_mean: float,
    bl_std: float,
    direction: str = "higher_is_worse",
) -> float:
    """
    Map a raw metric value to [0, 100] relative to the speaker's
    baseline using a sigmoid activation.

    Parameters
    ----------
    value     : current window metric value
    bl_mean   : baseline mean for this metric
    bl_std    : baseline SD for this metric
    direction : "higher_is_worse" — only positive deviations count
                "lower_is_worse"  — only negative deviations count
                "any"             — absolute deviation

    Returns
    -------
    float in [0, 100].
    """
    if bl_std > 1e-8:
        z = (value - bl_mean) / bl_std
    else:
        # no baseline variance — compare to mean directly
        diff = value - bl_mean
        if abs(bl_mean) > 1e-8:
            z = diff / (abs(bl_mean) * 0.2)   # treat 20% of mean as 1σ
        else:
            z = diff * 5.0 if abs(diff) > 1e-8 else 0.0

    # apply direction filter
    if direction == "higher_is_worse":
        z = max(0.0, z)        # only penalise above-baseline
    elif direction == "lower_is_worse":
        z = max(0.0, -z)       # only penalise below-baseline
    else:  # "any"
        z = abs(z)

    # sigmoid mapping: score = 100 / (1 + exp(−k(z − mid)))
    score = 100.0 / (1.0 + np.exp(-_SCALE_K / 10.0 * (z - _SCALE_MID)))

    # ensure 0 when z ≈ 0 (subtract the baseline sigmoid value)
    floor = 100.0 / (1.0 + np.exp(-_SCALE_K / 10.0 * (0.0 - _SCALE_MID)))
    score = max(0.0, (score - floor) / (100.0 - floor) * 100.0)

    return round(float(score), 1)
