"""
SpeechScore 2.0 — Temporal Analyzer Unit Tests

Tests all four temporal metrics using synthetic WindowMetrics sequences.
No audio or Whisper required.

Run:
    cd Comm-Assist && python -m pytest tests/test_temporal.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from speechscore.models.schemas import WindowMetrics
from speechscore.analyzers.temporal import TemporalAnalyzer


def _make_windows(
    n: int = 15,
    wpm_list: list | None = None,
    pitch_std_list: list | None = None,
    rms_list: list | None = None,
    pause_freq_list: list | None = None,
    phonation_list: list | None = None,
    asr_list: list | None = None,
) -> list[WindowMetrics]:
    """Generate a sequence of WindowMetrics for testing."""
    windows = []
    for i in range(n):
        windows.append(WindowMetrics(
            window_id=i,
            start_time=i * 5.0,
            end_time=i * 5.0 + 10.0,
            speech_rate_wpm=(wpm_list[i] if wpm_list else 150.0),
            pitch_std=(pitch_std_list[i] if pitch_std_list else 25.0),
            pitch_mean=120.0,
            volume_consistency=0.6,
            rms_mean=(rms_list[i] if rms_list else 0.05),
            rms_std=0.01,
            pause_count=2,
            pause_frequency_per_min=(pause_freq_list[i] if pause_freq_list else 6.0),
            mean_pause_duration=0.3,
            filler_count=0,
            filler_rate_per_100=0.0,
            phonation_ratio=(phonation_list[i] if phonation_list else 0.85),
            asr_confidence=(asr_list[i] if asr_list else 0.92),
            word_recognition_rate=0.95,
            word_count=25,
        ))
    return windows


# ────────────────────────────────────────────────────────────────────
# Confidence Trajectory
# ────────────────────────────────────────────────────────────────────

class TestConfidenceTrajectory:

    def setup_method(self):
        self.analyzer = TemporalAnalyzer()

    def test_gaining_confidence(self):
        """Decreasing F0 SD → negative slope → gaining confidence."""
        pitch_stds = [40.0, 37.0, 34.0, 31.0, 28.0, 25.0, 22.0, 19.0,
                      16.0, 13.0, 10.0, 8.0, 7.0, 6.0, 5.0]
        windows = _make_windows(pitch_std_list=pitch_stds)
        result = self.analyzer.analyze(windows)
        ct = result.confidence_trajectory
        assert ct.slope < 0
        assert ct.direction == "increasing"
        assert ct.p_value < 0.05

    def test_losing_confidence(self):
        """Increasing F0 SD → positive slope → losing confidence."""
        pitch_stds = [10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0, 31.0,
                      34.0, 37.0, 40.0, 43.0, 46.0, 49.0, 52.0]
        windows = _make_windows(pitch_std_list=pitch_stds)
        result = self.analyzer.analyze(windows)
        ct = result.confidence_trajectory
        assert ct.slope > 0
        assert ct.direction == "decreasing"
        assert ct.p_value < 0.05

    def test_stable_confidence(self):
        """Flat F0 SD → near-zero slope → stable."""
        pitch_stds = [25.0] * 15
        windows = _make_windows(pitch_std_list=pitch_stds)
        result = self.analyzer.analyze(windows)
        ct = result.confidence_trajectory
        assert ct.direction == "stable"

    def test_has_r_squared(self):
        pitch_stds = list(np.linspace(40, 10, 15))
        windows = _make_windows(pitch_std_list=pitch_stds)
        result = self.analyzer.analyze(windows)
        assert result.confidence_trajectory.r_squared > 0.5

    def test_returns_trajectory_array(self):
        windows = _make_windows()
        result = self.analyzer.analyze(windows)
        assert len(result.confidence_trajectory.per_window_pitch_std) == 15


# ────────────────────────────────────────────────────────────────────
# Warmup Index
# ────────────────────────────────────────────────────────────────────

class TestWarmupIndex:

    def setup_method(self):
        self.analyzer = TemporalAnalyzer()

    def test_clear_warmup(self):
        """Speaker starts fast then settles down."""
        wpm = [220, 210, 200, 155, 150, 150, 148, 152, 150, 149,
               151, 150, 148, 152, 150]
        windows = _make_windows(wpm_list=wpm)
        result = self.analyzer.analyze(windows)
        wi = result.warmup_index
        # change point should be early (windows 2-4)
        assert wi.warmup_window <= 5
        assert wi.pre_warmup_mean_wpm > wi.post_warmup_mean_wpm

    def test_no_warmup(self):
        """Constant speech rate → no warmup needed."""
        wpm = [150] * 15
        windows = _make_windows(wpm_list=wpm)
        result = self.analyzer.analyze(windows)
        wi = result.warmup_index
        assert wi.change_point_detected is False

    def test_returns_trajectory(self):
        windows = _make_windows()
        result = self.analyzer.analyze(windows)
        assert len(result.warmup_index.speech_rate_trajectory) == 15


# ────────────────────────────────────────────────────────────────────
# Fatigue Detection
# ────────────────────────────────────────────────────────────────────

class TestFatigueDetection:

    def setup_method(self):
        self.analyzer = TemporalAnalyzer()

    def test_clear_fatigue(self):
        """Phonation drops, pauses increase in second half."""
        n = 15
        mid = n // 2
        phonation = [0.90] * mid + [0.70] * (n - mid)
        pause_freq = [4.0] * mid + [12.0] * (n - mid)
        windows = _make_windows(
            phonation_list=phonation,
            pause_freq_list=pause_freq,
        )
        result = self.analyzer.analyze(windows)
        fd = result.fatigue_detection
        assert fd.fatigue_score > 20
        assert fd.significant is True
        assert len(fd.degraded_metrics) >= 1

    def test_no_fatigue(self):
        """Consistent metrics throughout."""
        windows = _make_windows()
        result = self.analyzer.analyze(windows)
        fd = result.fatigue_detection
        assert fd.fatigue_score == 0.0
        assert fd.significant is False

    def test_has_metric_details(self):
        windows = _make_windows()
        result = self.analyzer.analyze(windows)
        fd = result.fatigue_detection
        assert "pause_frequency_per_min" in fd.metric_details
        assert "phonation_ratio" in fd.metric_details
        assert "p_value" in fd.metric_details["pause_frequency_per_min"]

    def test_half_means_populated(self):
        windows = _make_windows()
        result = self.analyzer.analyze(windows)
        fd = result.fatigue_detection
        assert "phonation_ratio" in fd.first_half_means
        assert "phonation_ratio" in fd.second_half_means

    def test_small_sample_effect_size_fallback(self):
        """With N<15 per half, medium effect size alone triggers fatigue."""
        # Use only 8 windows (4 per half), so t-test lacks power
        # but effect size is clear
        n = 8
        mid = n // 2
        phonation = [0.90] * mid + [0.60] * (n - mid)
        pause_freq = [4.0] * mid + [14.0] * (n - mid)
        windows = _make_windows(
            n=n,
            phonation_list=phonation,
            pause_freq_list=pause_freq,
        )
        analyzer = TemporalAnalyzer()
        result = analyzer.analyze(windows)
        fd = result.fatigue_detection
        # Fatigue should be detected via effect-size fallback
        assert len(fd.degraded_metrics) >= 1
        assert fd.fatigue_score > 0

    def test_small_sample_no_degradation(self):
        """With N<15 but constant metrics, no fatigue via fallback either."""
        windows = _make_windows(n=8)
        analyzer = TemporalAnalyzer()
        result = analyzer.analyze(windows)
        fd = result.fatigue_detection
        assert fd.fatigue_score == 0.0


# ────────────────────────────────────────────────────────────────────
# Engagement Arc
# ────────────────────────────────────────────────────────────────────

class TestEngagementArc:

    def setup_method(self):
        self.analyzer = TemporalAnalyzer()

    def test_declining_energy(self):
        """Energy drops over time → declining shape."""
        rms = list(np.linspace(0.08, 0.02, 15))
        windows = _make_windows(rms_list=rms)
        result = self.analyzer.analyze(windows)
        ea = result.engagement_arc
        assert ea.shape == "declining"

    def test_rising_energy(self):
        """Energy rises over time → rising shape."""
        rms = list(np.linspace(0.02, 0.08, 15))
        windows = _make_windows(rms_list=rms)
        result = self.analyzer.analyze(windows)
        ea = result.engagement_arc
        assert ea.shape == "rising"

    def test_flat_energy(self):
        """Constant energy → flat shape."""
        rms = [0.05] * 15
        windows = _make_windows(rms_list=rms)
        result = self.analyzer.analyze(windows)
        ea = result.engagement_arc
        assert ea.shape == "flat"

    def test_returns_trajectories(self):
        windows = _make_windows()
        result = self.analyzer.analyze(windows)
        ea = result.engagement_arc
        assert len(ea.energy_trajectory) == 15
        assert len(ea.ideal_template) == 15

    def test_score_in_range(self):
        windows = _make_windows()
        result = self.analyzer.analyze(windows)
        ea = result.engagement_arc
        assert 0 <= ea.score <= 100

    def test_ideal_arc_pattern_high_score(self):
        """Energy following a strong-open-dip-build-close pattern → high score."""
        # Simulate: strong open, dip mid, build toward end
        rms = [0.08, 0.07, 0.06, 0.05, 0.045, 0.05, 0.06, 0.065,
               0.07, 0.075, 0.08, 0.08, 0.082, 0.085, 0.08]
        windows = _make_windows(rms_list=rms)
        result = self.analyzer.analyze(windows)
        ea = result.engagement_arc
        # With piecewise template, this roughly matching pattern → decent score
        assert ea.score > 30


# ────────────────────────────────────────────────────────────────────
# Edge Cases
# ────────────────────────────────────────────────────────────────────

class TestTemporalEdgeCases:

    def setup_method(self):
        self.analyzer = TemporalAnalyzer()

    def test_too_few_windows(self):
        """< 4 windows → returns defaults."""
        windows = _make_windows(n=3)
        result = self.analyzer.analyze(windows)
        assert result.confidence_trajectory.direction == "stable"
        assert result.fatigue_detection.fatigue_score == 0.0

    def test_minimum_windows(self):
        """Exactly 4 windows → should work."""
        windows = _make_windows(n=4)
        result = self.analyzer.analyze(windows)
        assert result.confidence_trajectory.per_window_pitch_std is not None

    def test_many_windows(self):
        """50 windows → should work without error."""
        windows = _make_windows(n=50)
        result = self.analyzer.analyze(windows)
        assert len(result.confidence_trajectory.per_window_pitch_std) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
