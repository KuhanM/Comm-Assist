"""
SpeechScore 2.0 — Speaker Baseline Extractor  ⭐ NOVEL

Extracts a personalised speaker baseline from the first N seconds of
audio.  This baseline captures the speaker's *natural* communication
profile (speech rate, pitch patterns, fluency habits) before fatigue,
nervousness, or topic difficulty has an effect.

The baseline is used by ``AdaptiveScorer`` to normalise all metrics
relative to the speaker's own norms — the key insight behind
Speaker-Adaptive Normalization (Contribution 2 in the paper).

Novelty argument
----------------
Prior art (Praat, MyVoice, IELTS raters, etc.) compares speakers
against *population* norms.  This penalises speakers with naturally
fast/slow rates, high/low pitch, etc.  By establishing a *personal*
baseline in the first 30 s, we can separate communicative *change*
(nervousness, fatigue, struggle) from *style*.

References:
  - Levelt (1989) "Speaking: From Intention to Articulation" — individual
    variation as the norm rather than the exception.
  - Nolan (2009) "The Phonetic Bases of Speaker Recognition" — speakers
    have stable long-term averages.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from speechscore.config.settings import SpeechScoreConfig
from speechscore.models.schemas import SpeakerBaseline, WindowMetrics

logger = logging.getLogger(__name__)

# Minimum windows for a useful baseline
_MIN_BASELINE_WINDOWS = 2


class BaselineExtractor:
    """
    Extract a personalised speaker baseline from early windows.

    Usage::

        extractor = BaselineExtractor(config)
        baseline = extractor.extract(window_metrics)
    """

    def __init__(self, config: SpeechScoreConfig | None = None) -> None:
        self.config = config or SpeechScoreConfig()
        self._baseline_duration = self.config.baseline.baseline_duration

    def extract(self, windows: list[WindowMetrics]) -> SpeakerBaseline:
        """
        Extract baseline statistics from the first ``baseline_duration``
        seconds of window metrics.

        Parameters
        ----------
        windows : full list of WindowMetrics from Phase 1.

        Returns
        -------
        SpeakerBaseline with mean + std for each tracked metric.
        """
        if not windows:
            logger.warning("No windows provided — returning empty baseline.")
            return SpeakerBaseline()

        # select windows whose *start* falls within baseline duration
        baseline_wins = [
            w for w in windows
            if w.start_time < self._baseline_duration
        ]

        if len(baseline_wins) < _MIN_BASELINE_WINDOWS:
            # audio too short — use all windows as baseline
            logger.info(
                "Only %d windows within first %.0f s; "
                "using all %d windows as baseline.",
                len(baseline_wins),
                self._baseline_duration,
                len(windows),
            )
            baseline_wins = windows

        return self._compute_stats(baseline_wins)

    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_stats(wins: list[WindowMetrics]) -> SpeakerBaseline:
        """
        Compute mean ± std for each baseline metric across the given
        windows.  Uses ddof=1 for sample standard deviation.
        """

        def _extract(attr: str) -> np.ndarray:
            return np.array(
                [getattr(w, attr) or 0.0 for w in wins],
                dtype=np.float64,
            )

        def _mean_std(arr: np.ndarray) -> tuple[float, float]:
            m = float(np.mean(arr))
            s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            return round(m, 4), round(s, 4)

        sr_m, sr_s = _mean_std(_extract("speech_rate_wpm"))
        pm_m, pm_s = _mean_std(_extract("pitch_mean"))
        ps_m, ps_s = _mean_std(_extract("pitch_std"))
        vm_m, vm_s = _mean_std(_extract("rms_mean"))
        pf_m, pf_s = _mean_std(_extract("pause_frequency_per_min"))
        ph_m, ph_s = _mean_std(_extract("phonation_ratio"))
        fr_m, fr_s = _mean_std(_extract("filler_rate_per_100"))

        actual_dur = wins[-1].end_time - wins[0].start_time

        return SpeakerBaseline(
            baseline_duration=round(actual_dur, 1),
            windows_used=len(wins),
            speech_rate_mean=sr_m,
            speech_rate_std=sr_s,
            pitch_mean=pm_m,
            pitch_std_mean=ps_m,
            pitch_std_std=ps_s,
            volume_mean=vm_m,
            volume_std=vm_s,
            pause_freq_mean=pf_m,
            pause_freq_std=pf_s,
            phonation_mean=ph_m,
            phonation_std=ph_s,
            filler_rate_mean=fr_m,
            filler_rate_std=fr_s,
        )
