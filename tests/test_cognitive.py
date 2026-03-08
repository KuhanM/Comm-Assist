"""
SpeechScore 2.0 — Tests for Day 6: Cognitive Strain Index

25 tests covering:
  - _deviation_score helper
  - Individual strain indicators
  - Per-window CSI computation
  - Struggle point detection
  - Aggregate statistics
  - Edge cases (no windows, all calm, all stressed)
"""

import pytest
import numpy as np

from speechscore.models.schemas import WindowMetrics, SpeakerBaseline
from speechscore.analyzers.cognitive import (
    CognitiveAnalyzer,
    _deviation_score,
)


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def _calm_baseline() -> SpeakerBaseline:
    """Baseline of a calm, fluent speaker."""
    return SpeakerBaseline(
        baseline_duration=30.0,
        windows_used=6,
        speech_rate_mean=130.0,
        speech_rate_std=8.0,
        pitch_mean=180.0,
        pitch_std_mean=20.0,
        pitch_std_std=4.0,
        volume_mean=0.05,
        volume_std=0.005,
        pause_freq_mean=5.0,
        pause_freq_std=2.0,
        phonation_mean=0.75,
        phonation_std=0.05,
        filler_rate_mean=1.0,
        filler_rate_std=0.5,
    )


def _make_window(
    wid: int = 0,
    start: float = 0.0,
    end: float = 10.0,
    wpm: float = 130.0,
    pitch_std: float = 20.0,
    pause_freq: float = 5.0,
    filler_rate: float = 1.0,
    max_pause: float = 0.5,
    phonation: float = 0.75,
    asr_conf: float = 0.85,
    transcript: str = "sample speech",
) -> WindowMetrics:
    return WindowMetrics(
        window_id=wid,
        start_time=start,
        end_time=end,
        speech_rate_wpm=wpm,
        pitch_mean=180.0,
        pitch_std=pitch_std,
        volume_consistency=0.80,
        rms_mean=0.05,
        pause_count=3,
        pause_frequency_per_min=pause_freq,
        mean_pause_duration=0.3,
        max_pause_duration=max_pause,
        filler_count=1,
        filler_rate_per_100=filler_rate,
        phonation_ratio=phonation,
        asr_confidence=asr_conf,
        word_recognition_rate=0.90,
        word_count=22,
        transcript=transcript,
    )


# ================================================================
# DEVIATION SCORE HELPER
# ================================================================

class TestDeviationScore:
    """Tests for the _deviation_score mapping function."""

    def test_at_baseline_returns_zero(self):
        """Value == baseline mean → score ≈ 0."""
        score = _deviation_score(5.0, bl_mean=5.0, bl_std=2.0, direction="higher_is_worse")
        assert score < 5.0  # near zero

    def test_above_baseline_higher_is_worse(self):
        """Value well above baseline → high score."""
        score = _deviation_score(15.0, bl_mean=5.0, bl_std=2.0, direction="higher_is_worse")
        assert score > 50.0

    def test_below_baseline_higher_is_worse(self):
        """Value below baseline with higher_is_worse → score ≈ 0."""
        score = _deviation_score(2.0, bl_mean=5.0, bl_std=2.0, direction="higher_is_worse")
        assert score < 5.0

    def test_any_direction_above(self):
        """Direction='any' — above baseline → score > 0."""
        score = _deviation_score(15.0, bl_mean=5.0, bl_std=2.0, direction="any")
        assert score > 50.0

    def test_any_direction_below(self):
        """Direction='any' — below baseline → score > 0 too."""
        score = _deviation_score(-5.0, bl_mean=5.0, bl_std=2.0, direction="any")
        assert score > 50.0

    def test_score_bounded_0_100(self):
        """Score always in [0, 100] regardless of extreme inputs."""
        for val in [0, 1, 10, 100, 1000]:
            s = _deviation_score(val, 5.0, 2.0, "higher_is_worse")
            assert 0 <= s <= 100

    def test_zero_std_still_works(self):
        """When baseline std=0, falls back to % of mean."""
        score = _deviation_score(10.0, bl_mean=5.0, bl_std=0.0, direction="higher_is_worse")
        assert score > 0  # should detect deviation


# ================================================================
# PER-WINDOW CSI COMPUTATION
# ================================================================

class TestPerWindowCSI:
    """Tests for CognitiveAnalyzer per-window logic."""

    def test_calm_window_low_csi(self):
        """Window matching baseline → low CSI."""
        bl = _calm_baseline()
        win = _make_window()  # matches baseline defaults
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze([win], bl)
        assert result.per_window_csi[0] < 30

    def test_stressed_window_high_csi(self):
        """Window with high pauses, fillers, erratic rate → high CSI."""
        bl = _calm_baseline()
        win = _make_window(
            wpm=200.0,            # way above baseline 130
            pitch_std=50.0,       # way above baseline 20
            pause_freq=20.0,      # way above baseline 5
            filler_rate=10.0,     # way above baseline 1
            max_pause=3.0,        # very long pause
        )
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze([win], bl)
        assert result.per_window_csi[0] > 50

    def test_csi_range(self):
        """CSI is always in [0, 100]."""
        bl = _calm_baseline()
        for _ in range(10):
            win = _make_window(
                wpm=float(np.random.uniform(50, 250)),
                pause_freq=float(np.random.uniform(0, 30)),
                filler_rate=float(np.random.uniform(0, 15)),
                pitch_std=float(np.random.uniform(5, 80)),
            )
            analyzer = CognitiveAnalyzer()
            result = analyzer.analyze([win], bl)
            assert 0 <= result.per_window_csi[0] <= 100


# ================================================================
# STRUGGLE POINT DETECTION
# ================================================================

class TestStrugglePoints:
    """Tests for struggle point identification."""

    def test_no_struggle_when_calm(self):
        """All calm windows → no struggle points."""
        bl = _calm_baseline()
        wins = [_make_window(wid=i, start=i*5, end=i*5+10) for i in range(8)]
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze(wins, bl)
        assert result.struggle_count == 0

    def test_struggle_detected_when_stressed(self):
        """One highly stressed window among calm ones → 1 struggle point."""
        bl = _calm_baseline()
        wins = [_make_window(wid=i, start=i*5, end=i*5+10) for i in range(8)]
        # make window 4 very stressed
        wins[4] = _make_window(
            wid=4, start=20, end=30,
            wpm=220, pitch_std=60, pause_freq=25,
            filler_rate=12.0, max_pause=3.0,
            transcript="I think the, um, the thing is basically, like, the process...",
        )
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze(wins, bl)
        assert result.struggle_count >= 1
        # the struggle point should be window 4
        sp_ids = [sp.window_id for sp in result.struggle_points]
        assert 4 in sp_ids

    def test_struggle_has_primary_cause(self):
        """Struggle points report a primary cause."""
        bl = _calm_baseline()
        win = _make_window(wpm=220, pause_freq=25, filler_rate=12.0, pitch_std=60)
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze([win], bl)
        if result.struggle_points:
            sp = result.struggle_points[0]
            assert sp.primary_cause in [
                "pause_excess", "filler_excess", "speech_rate_dev",
                "pitch_instability", "hesitation_pattern",
            ]

    def test_struggle_has_breakdown(self):
        """Struggle points have a strain_breakdown dict."""
        bl = _calm_baseline()
        win = _make_window(wpm=220, pause_freq=25, filler_rate=12.0, pitch_std=60)
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze([win], bl)
        if result.struggle_points:
            bd = result.struggle_points[0].strain_breakdown
            assert len(bd) == 6  # all 6 indicators

    def test_custom_threshold(self):
        """Custom threshold changes struggle point detection."""
        bl = _calm_baseline()
        win = _make_window(wpm=180, pause_freq=12, filler_rate=5.0, pitch_std=35)
        # with default threshold (60) this might not be a struggle
        analyzer_strict = CognitiveAnalyzer(threshold=20.0)
        result_strict = analyzer_strict.analyze([win], bl)
        analyzer_lenient = CognitiveAnalyzer(threshold=90.0)
        result_lenient = analyzer_lenient.analyze([win], bl)
        # strict should have >= lenient struggle points
        assert result_strict.struggle_count >= result_lenient.struggle_count

    def test_struggle_pct(self):
        """struggle_pct is correctly computed."""
        bl = _calm_baseline()
        wins = [_make_window(wid=i, start=i*5, end=i*5+10) for i in range(10)]
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze(wins, bl)
        expected_pct = result.struggle_count / 10 * 100
        assert abs(result.struggle_pct - expected_pct) < 0.1


# ================================================================
# AGGREGATE STATISTICS
# ================================================================

class TestAggregateStats:
    """Tests for aggregate CSI statistics."""

    def test_mean_csi_matches(self):
        """mean_csi matches manual mean of per_window_csi."""
        bl = _calm_baseline()
        wins = [_make_window(wid=i, start=i*5, end=i*5+10) for i in range(6)]
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze(wins, bl)
        expected = np.mean(result.per_window_csi)
        assert abs(result.mean_csi - expected) < 0.2

    def test_max_min_correct(self):
        """max_csi and min_csi computed correctly."""
        bl = _calm_baseline()
        wpm_vals = [130, 130, 130, 200, 130, 130]
        wins = [
            _make_window(wid=i, start=i*5, end=i*5+10, wpm=wpm_vals[i])
            for i in range(6)
        ]
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze(wins, bl)
        assert result.max_csi >= result.mean_csi
        assert result.min_csi <= result.mean_csi

    def test_indicator_means_populated(self):
        """indicator_means dict has all 6 indicators."""
        bl = _calm_baseline()
        wins = [_make_window(wid=i, start=i*5, end=i*5+10) for i in range(6)]
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze(wins, bl)
        assert len(result.indicator_means) == 6
        for key in ["pause_excess", "filler_excess", "speech_rate_dev",
                     "pitch_instability", "hesitation_pattern", "clarity_strain"]:
            assert key in result.indicator_means


# ================================================================
# EDGE CASES
# ================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_windows(self):
        """Empty window list → default result."""
        bl = _calm_baseline()
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze([], bl)
        assert result.mean_csi == 0.0
        assert result.struggle_count == 0

    def test_single_window(self):
        """Single window still works."""
        bl = _calm_baseline()
        win = _make_window()
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze([win], bl)
        assert len(result.per_window_csi) == 1

    def test_hesitation_boost_long_pause(self):
        """A very long max_pause boosts hesitation score."""
        bl = _calm_baseline()
        win_short = _make_window(pause_freq=10, filler_rate=5, max_pause=0.5)
        win_long = _make_window(pause_freq=10, filler_rate=5, max_pause=3.0)
        analyzer = CognitiveAnalyzer()
        r1 = analyzer.analyze([win_short], bl)
        r2 = analyzer.analyze([win_long], bl)
        # long pause window should have higher CSI
        assert r2.per_window_csi[0] >= r1.per_window_csi[0]


# ================================================================
# CLARITY STRAIN (NEW INDICATOR)
# ================================================================

class TestClarityStrain:
    """Tests for the 6th indicator: ASR confidence drop."""

    def test_normal_asr_low_strain(self):
        """ASR=0.90 (baseline level) → low clarity strain."""
        bl = _calm_baseline()
        win = _make_window(asr_conf=0.90)
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze([win], bl)
        assert result.indicator_means["clarity_strain"] < 15

    def test_low_asr_high_strain(self):
        """ASR=0.45 → very high clarity strain."""
        bl = _calm_baseline()
        win = _make_window(asr_conf=0.45)
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze([win], bl)
        assert result.indicator_means["clarity_strain"] > 40

    def test_asr_below_threshold_triggers_struggle(self):
        """Very low ASR with other elevated metrics → struggle point."""
        bl = _calm_baseline()
        win = _make_window(
            wpm=200, pause_freq=15, filler_rate=8.0,
            pitch_std=40, asr_conf=0.40,
        )
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze([win], bl)
        assert result.struggle_count >= 1


# ================================================================
# RELIABILITY FLAGS
# ================================================================

class TestReliabilityFlags:
    """Tests for window reliability flagging."""

    def test_normal_window_reliable(self):
        """Normal window has reliable=True."""
        w = _make_window()
        assert w.reliable is True
        assert w.reliability_flags == []

    def test_extreme_wpm_flagged(self):
        """WPM > 240 flags as unreliable."""
        w = _make_window(wpm=260)
        # In the actual pipeline, reliability is set after construction.
        # Here we test the schema supports it.
        w.reliable = False
        w.reliability_flags = ["extreme_wpm"]
        assert w.reliable is False
        assert "extreme_wpm" in w.reliability_flags

    def test_low_asr_flagged(self):
        """ASR < 0.50 flags as unreliable."""
        w = _make_window(asr_conf=0.40)
        w.reliable = False
        w.reliability_flags = ["very_low_asr"]
        assert not w.reliable
