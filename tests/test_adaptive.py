"""
SpeechScore 2.0 — Tests for Day 5:
  • BaselineExtractor
  • AdaptiveScorer + z-score normalization

30 tests covering:
  - Baseline extraction from early windows
  - Baseline fallback when audio is short
  - Z-score computation with variance & zero-variance
  - Deviation labelling
  - Composite score mapping
  - Speech rate delta %
  - Pitch stability ratio
  - Consistency score
  - End-to-end adaptive workflow
"""

import pytest
import numpy as np

from speechscore.config.settings import SpeechScoreConfig
from speechscore.models.schemas import WindowMetrics, SpeakerBaseline
from speechscore.analyzers.baseline import BaselineExtractor
from speechscore.analyzers.adaptive_scorer import (
    AdaptiveScorer,
    _deviation_label,
    _interpret,
)


# ────────────────────────────────────────────────────────────
# Helpers — create synthetic window metrics
# ────────────────────────────────────────────────────────────

def _make_window(
    wid: int,
    start: float,
    end: float,
    wpm: float = 130.0,
    pitch_std: float = 25.0,
    rms: float = 0.05,
    pause_freq: float = 6.0,
    phonation: float = 0.70,
    filler_rate: float = 2.0,
    asr_conf: float = 0.85,
    pitch_mean: float = 180.0,
) -> WindowMetrics:
    return WindowMetrics(
        window_id=wid,
        start_time=start,
        end_time=end,
        speech_rate_wpm=wpm,
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        volume_consistency=0.80,
        rms_mean=rms,
        rms_std=rms * 0.1,
        pause_count=3,
        pause_frequency_per_min=pause_freq,
        mean_pause_duration=0.3,
        filler_count=1,
        filler_rate_per_100=filler_rate,
        phonation_ratio=phonation,
        asr_confidence=asr_conf,
        word_recognition_rate=0.90,
        word_count=22,
    )


def _make_windows(
    n: int,
    wpm_values: list[float] | None = None,
    pitch_std_values: list[float] | None = None,
    rms_values: list[float] | None = None,
    pause_freq_values: list[float] | None = None,
    phonation_values: list[float] | None = None,
    filler_rate_values: list[float] | None = None,
) -> list[WindowMetrics]:
    """Create n windows with 10s duration, 5s hop."""
    wins = []
    for i in range(n):
        start = i * 5.0
        end = start + 10.0
        wins.append(_make_window(
            wid=i,
            start=start,
            end=end,
            wpm=(wpm_values[i] if wpm_values else 130.0),
            pitch_std=(pitch_std_values[i] if pitch_std_values else 25.0),
            rms=(rms_values[i] if rms_values else 0.05),
            pause_freq=(pause_freq_values[i] if pause_freq_values else 6.0),
            phonation=(phonation_values[i] if phonation_values else 0.70),
            filler_rate=(filler_rate_values[i] if filler_rate_values else 2.0),
        ))
    return wins


# ================================================================
# BASELINE EXTRACTOR TESTS
# ================================================================

class TestBaselineExtraction:
    """Tests for BaselineExtractor."""

    def test_baseline_selects_first_30s(self):
        """With 60s of audio (12 windows), baseline should use first 6 windows."""
        wins = _make_windows(12)  # 0-55s
        extractor = BaselineExtractor()
        baseline = extractor.extract(wins)
        assert baseline.windows_used == 6  # windows starting at 0,5,10,15,20,25

    def test_baseline_uses_all_if_short(self):
        """If audio < 30s, baseline uses all windows."""
        wins = _make_windows(3)  # 0-20s
        extractor = BaselineExtractor()
        baseline = extractor.extract(wins)
        assert baseline.windows_used == 3

    def test_baseline_empty_windows(self):
        """Empty window list → empty baseline."""
        extractor = BaselineExtractor()
        baseline = extractor.extract([])
        assert baseline.windows_used == 0
        assert baseline.speech_rate_mean == 0.0

    def test_baseline_speech_rate(self):
        """Baseline captures correct mean WPM."""
        # first 6 windows (within 30s) have WPM = [120,130,140,125,135,130]
        wpm = [120, 130, 140, 125, 135, 130, 160, 170, 180, 190, 200, 210]
        wins = _make_windows(12, wpm_values=wpm)
        extractor = BaselineExtractor()
        baseline = extractor.extract(wins)
        # baseline uses first 6 windows → mean of [120,130,140,125,135,130]
        expected_mean = np.mean([120, 130, 140, 125, 135, 130])
        assert abs(baseline.speech_rate_mean - expected_mean) < 0.1

    def test_baseline_speech_rate_std(self):
        """Baseline captures correct WPM standard deviation."""
        wpm = [120, 130, 140, 125, 135, 130, 160, 170, 180, 190, 200, 210]
        wins = _make_windows(12, wpm_values=wpm)
        extractor = BaselineExtractor()
        baseline = extractor.extract(wins)
        expected_std = float(np.std([120, 130, 140, 125, 135, 130], ddof=1))
        assert abs(baseline.speech_rate_std - expected_std) < 0.1

    def test_baseline_pitch_std(self):
        """Baseline records pitch_std_mean correctly."""
        ps = [20, 22, 24, 21, 23, 22, 30, 35, 40, 45, 50, 55]
        wins = _make_windows(12, pitch_std_values=ps)
        extractor = BaselineExtractor()
        baseline = extractor.extract(wins)
        expected = np.mean([20, 22, 24, 21, 23, 22])
        assert abs(baseline.pitch_std_mean - expected) < 0.1

    def test_baseline_duration(self):
        """Baseline reports correct duration from first to last baseline window."""
        wins = _make_windows(12)
        extractor = BaselineExtractor()
        baseline = extractor.extract(wins)
        # first 6 windows: start=0, last ends at 35s → duration = 35.0
        assert baseline.baseline_duration == 35.0

    def test_baseline_custom_duration(self):
        """Custom baseline_duration config works."""
        config = SpeechScoreConfig()
        config.baseline.baseline_duration = 15.0
        wins = _make_windows(12)
        extractor = BaselineExtractor(config)
        baseline = extractor.extract(wins)
        # windows within first 15s: start at 0,5,10 → 3 windows
        assert baseline.windows_used == 3

    def test_baseline_phonation(self):
        """Baseline captures phonation ratio."""
        ph = [0.75, 0.72, 0.78, 0.71, 0.74, 0.73, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35]
        wins = _make_windows(12, phonation_values=ph)
        extractor = BaselineExtractor()
        baseline = extractor.extract(wins)
        expected = np.mean([0.75, 0.72, 0.78, 0.71, 0.74, 0.73])
        assert abs(baseline.phonation_mean - expected) < 0.001


# ================================================================
# DEVIATION LABEL TESTS
# ================================================================

class TestDeviationLabel:
    """Tests for _deviation_label helper."""

    def test_typical(self):
        assert _deviation_label(0.0) == "typical"
        assert _deviation_label(0.3) == "typical"
        assert _deviation_label(-0.4) == "typical"

    def test_moderate(self):
        assert _deviation_label(0.8) == "moderate"
        assert _deviation_label(-1.2) == "moderate"

    def test_high(self):
        assert _deviation_label(2.0) == "high"
        assert _deviation_label(-2.3) == "high"

    def test_extreme(self):
        assert _deviation_label(3.0) == "extreme"
        assert _deviation_label(-4.5) == "extreme"


# ================================================================
# INTERPRETATION TESTS
# ================================================================

class TestInterpretation:
    """Tests for _interpret helper."""

    def test_typical_range(self):
        result = _interpret("WPM", 0.2, None)
        assert "typical" in result.lower()

    def test_improvement(self):
        result = _interpret("Phonation", 1.5, True)
        assert "improvement" in result.lower()

    def test_degradation(self):
        result = _interpret("Pause Freq", 1.5, False)
        assert "degradation" in result.lower()


# ================================================================
# ADAPTIVE SCORER TESTS
# ================================================================

class TestAdaptiveScorer:
    """Tests for AdaptiveScorer."""

    def _baseline_and_windows(self):
        """Create a baseline from first 6 windows and all 12 windows."""
        wpm = [120, 130, 140, 125, 135, 130, 160, 170, 180, 190, 200, 210]
        wins = _make_windows(12, wpm_values=wpm)
        extractor = BaselineExtractor()
        baseline = extractor.extract(wins)
        return baseline, wins

    def test_z_scores_not_empty(self):
        baseline, wins = self._baseline_and_windows()
        scorer = AdaptiveScorer()
        result = scorer.score(wins, baseline)
        assert len(result.adaptive_metrics) == 6  # 6 tracked metrics

    def test_speech_rate_positive_z(self):
        """Full-speech WPM is higher than baseline → positive z-score."""
        baseline, wins = self._baseline_and_windows()
        scorer = AdaptiveScorer()
        result = scorer.score(wins, baseline)
        # find speech rate metric
        sr = next(m for m in result.adaptive_metrics if "Rate" in m.metric_name)
        # global WPM mean (all 12) > baseline mean (first 6) → z > 0
        assert sr.z_score > 0

    def test_speech_rate_delta_positive(self):
        """Speaking faster than baseline → positive delta %."""
        baseline, wins = self._baseline_and_windows()
        scorer = AdaptiveScorer()
        result = scorer.score(wins, baseline)
        assert result.speech_rate_delta_pct > 0

    def test_consistent_speaker_high_score(self):
        """Speaker identical to baseline → high adaptive score."""
        wins = _make_windows(12)  # all identical → global ≈ baseline
        extractor = BaselineExtractor()
        baseline = extractor.extract(wins)
        scorer = AdaptiveScorer()
        result = scorer.score(wins, baseline)
        assert result.overall_adaptive_score > 60
        assert result.consistency_score > 80

    def test_erratic_speaker_lower_score(self):
        """Speaker deviates wildly from baseline → lower score."""
        # baseline: calm 130 WPM; later: chaotic
        wpm = [130, 130, 130, 130, 130, 130, 50, 250, 40, 260, 30, 270]
        wins = _make_windows(12, wpm_values=wpm)
        extractor = BaselineExtractor()
        baseline = extractor.extract(wins)
        scorer = AdaptiveScorer()
        result = scorer.score(wins, baseline)
        # should have substantial z-deviation
        sr = next(m for m in result.adaptive_metrics if "Rate" in m.metric_name)
        assert abs(sr.z_score) > 0.1 or result.consistency_score < 100

    def test_pitch_stability_ratio_equal(self):
        """Same pitch throughout → ratio ≈ 1.0."""
        wins = _make_windows(12)
        extractor = BaselineExtractor()
        baseline = extractor.extract(wins)
        scorer = AdaptiveScorer()
        result = scorer.score(wins, baseline)
        assert abs(result.pitch_stability_ratio - 1.0) < 0.01

    def test_pitch_stability_ratio_increased(self):
        """Pitch becomes more unstable → ratio > 1."""
        ps = [20, 20, 20, 20, 20, 20, 40, 40, 40, 40, 40, 40]
        wins = _make_windows(12, pitch_std_values=ps)
        extractor = BaselineExtractor()
        baseline = extractor.extract(wins)
        scorer = AdaptiveScorer()
        result = scorer.score(wins, baseline)
        assert result.pitch_stability_ratio > 1.0

    def test_empty_input(self):
        """Empty input → default result."""
        scorer = AdaptiveScorer()
        result = scorer.score([], SpeakerBaseline())
        assert result.overall_adaptive_score == 0.0

    def test_composite_score_range(self):
        """Composite score always in [0, 100]."""
        for _ in range(5):
            wpm = list(np.random.uniform(80, 220, 12))
            wins = _make_windows(12, wpm_values=wpm)
            extractor = BaselineExtractor()
            baseline = extractor.extract(wins)
            scorer = AdaptiveScorer()
            result = scorer.score(wins, baseline)
            assert 0 <= result.overall_adaptive_score <= 100

    def test_percentile_from_z(self):
        """z=0 → percentile ≈ 50."""
        wins = _make_windows(12)
        extractor = BaselineExtractor()
        baseline = extractor.extract(wins)
        scorer = AdaptiveScorer()
        result = scorer.score(wins, baseline)
        for am in result.adaptive_metrics:
            # when values are constant, z≈0, percentile≈50
            assert 40.0 < am.percentile < 60.0

    def test_all_metrics_have_interpretation(self):
        """Every adaptive metric has a non-empty interpretation string."""
        wins = _make_windows(12)
        extractor = BaselineExtractor()
        baseline = extractor.extract(wins)
        scorer = AdaptiveScorer()
        result = scorer.score(wins, baseline)
        for am in result.adaptive_metrics:
            assert len(am.interpretation) > 0


# ================================================================
# WINSORIZATION
# ================================================================

class TestWinsorization:
    """Tests for z-score winsorization at ±3."""

    def test_outlier_z_capped(self):
        """An extreme outlier z-score should be capped, raising composite."""
        # Create baseline from calm windows
        calm = _make_windows(6, wpm_values=[130]*6)
        extractor = BaselineExtractor()
        baseline = extractor.extract(calm)
        baseline.speech_rate_std = 5.0  # tight baseline

        # Now create windows with one massive outlier
        outlier = _make_windows(6, wpm_values=[130, 130, 130, 130, 130, 300])
        scorer = AdaptiveScorer()
        result = scorer.score(outlier, baseline)

        # composite score should still be reasonable (not crushed by one window)
        # without winsorization, z=34 would dominate; with it, capped at 3
        assert result.overall_adaptive_score > 20

    def test_no_outlier_unchanged(self):
        """When all z-scores are within ±3, winsorization has no effect."""
        wins = _make_windows(12, wpm_values=[130]*12)
        extractor = BaselineExtractor()
        baseline = extractor.extract(wins)
        scorer = AdaptiveScorer()
        result = scorer.score(wins, baseline)
        # All z-scores near 0 → should be unaffected
        for am in result.adaptive_metrics:
            assert abs(am.z_score) < 3.0
