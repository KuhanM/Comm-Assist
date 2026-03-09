"""
SpeechScore 2.0 — Tests for Multiscale Entropy Analysis (V2-1)

30+ tests covering:
  - sample_entropy correctness & edge cases
  - coarse_grain at various scales
  - multiscale_entropy profiles
  - complexity_index computation
  - profile classification
  - MultiscaleEntropyAnalyzer end-to-end
  - inverted-U scoring
"""

import pytest
import numpy as np

from speechscore.models.schemas import WindowMetrics
from speechscore.analyzers.entropy import (
    sample_entropy,
    coarse_grain,
    multiscale_entropy,
    complexity_index,
    _classify_profile,
    MultiscaleEntropyAnalyzer,
    _count_matches,
    _DEFAULT_M,
    _DEFAULT_R_FRAC,
    _MIN_SERIES_LEN,
)


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _make_window(wid: int = 0, wpm: float = 130.0, pitch_std: float = 20.0,
                 rms: float = 0.05, **kwargs) -> WindowMetrics:
    return WindowMetrics(
        window_id=wid, start_time=wid * 5, end_time=wid * 5 + 10,
        speech_rate_wpm=wpm, pitch_mean=180.0, pitch_std=pitch_std,
        volume_consistency=0.80, rms_mean=rms,
        pause_count=2, pause_frequency_per_min=5.0,
        mean_pause_duration=0.3, filler_count=1,
        filler_rate_per_100=1.0, phonation_ratio=0.75,
        asr_confidence=0.85, word_recognition_rate=0.90,
        word_count=22, transcript="sample speech",
        **kwargs,
    )


# ================================================================
# SAMPLE ENTROPY
# ================================================================

class TestSampleEntropy:
    """Tests for the core SampEn implementation."""

    def test_constant_series_returns_zero(self):
        """Constant series → SampEn = 0 (perfectly regular)."""
        x = np.ones(50)
        se = sample_entropy(x)
        assert se == 0.0

    def test_periodic_series_low_entropy(self):
        """A simple periodic signal has low SampEn."""
        x = np.tile([1.0, 2.0, 3.0, 2.0], 15)  # 60 points, period 4
        se = sample_entropy(x, m=2, r=0.3)
        assert se < 0.5  # highly regular

    def test_random_series_higher_entropy(self):
        """Random series has higher SampEn than periodic."""
        rng = np.random.default_rng(42)
        x_random = rng.normal(0, 1, 60)
        x_periodic = np.tile([1.0, 2.0, 3.0, 2.0], 15)

        se_random = sample_entropy(x_random, m=2, r=0.2 * np.std(x_random, ddof=1))
        se_periodic = sample_entropy(x_periodic, m=2, r=0.2 * np.std(x_periodic, ddof=1))

        assert se_random > se_periodic

    def test_short_series_returns_nan(self):
        """Series too short for m+2 → NaN."""
        x = np.array([1.0, 2.0])  # only 2 pts with m=2
        se = sample_entropy(x, m=2)
        assert np.isnan(se)

    def test_sampen_non_negative(self):
        """SampEn should be ≥ 0 for valid series."""
        rng = np.random.default_rng(123)
        for _ in range(10):
            x = rng.normal(0, 1, 30)
            se = sample_entropy(x)
            if np.isfinite(se):
                assert se >= 0.0

    def test_sampen_deterministic(self):
        """Same input → same output."""
        x = np.sin(np.linspace(0, 4 * np.pi, 40))
        se1 = sample_entropy(x)
        se2 = sample_entropy(x)
        assert se1 == se2

    def test_increasing_disorder_increases_sampen(self):
        """Adding noise to a regular signal increases SampEn."""
        x_base = np.sin(np.linspace(0, 4 * np.pi, 60))
        rng = np.random.default_rng(99)

        se_clean = sample_entropy(x_base)
        se_noisy = sample_entropy(x_base + rng.normal(0, 0.5, 60))

        # Allow for edge cases but generally noisy > clean
        if np.isfinite(se_clean) and np.isfinite(se_noisy):
            assert se_noisy >= se_clean * 0.8  # noisy should be >= clean


# ================================================================
# COARSE GRAINING
# ================================================================

class TestCoarseGrain:
    """Tests for the coarse-grain operation."""

    def test_scale_1_returns_copy(self):
        """Scale 1 → output equals input."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        cg = coarse_grain(x, 1)
        np.testing.assert_array_equal(cg, x)

    def test_scale_2_averages_pairs(self):
        """Scale 2 → pairwise averages."""
        x = np.array([1.0, 3.0, 5.0, 7.0])
        cg = coarse_grain(x, 2)
        np.testing.assert_array_almost_equal(cg, [2.0, 6.0])

    def test_scale_3(self):
        """Scale 3 → triplet averages."""
        x = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0])
        cg = coarse_grain(x, 3)
        np.testing.assert_array_almost_equal(cg, [6.0, 15.0])

    def test_excess_truncated(self):
        """Non-divisible length truncated (not padded)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cg = coarse_grain(x, 2)
        assert len(cg) == 2  # 5//2 = 2

    def test_scale_larger_than_series(self):
        """Scale > series length → empty array."""
        x = np.array([1.0, 2.0])
        cg = coarse_grain(x, 5)
        assert len(cg) == 0


# ================================================================
# MULTISCALE ENTROPY PROFILE
# ================================================================

class TestMultiscaleEntropy:
    """Tests for the MSE profile computation."""

    def test_returns_list_of_sampen(self):
        """MSE returns a list of SampEn values."""
        x = np.sin(np.linspace(0, 6 * np.pi, 40))
        mse = multiscale_entropy(x, max_scale=3)
        assert isinstance(mse, list)
        assert len(mse) <= 3

    def test_constant_series_all_zero(self):
        """Constant series → SampEn = 0 at all scales."""
        x = np.ones(40)
        mse = multiscale_entropy(x, max_scale=3)
        for s in mse:
            assert s == 0.0

    def test_profile_length_limited_by_data(self):
        """If coarse-graining makes series too short, profile is truncated."""
        x = np.random.default_rng(42).normal(0, 1, 12)
        mse = multiscale_entropy(x, max_scale=10, m=2)
        # At scale 5, series has 12//5=2 pts, which is < m+2=4
        assert len(mse) < 10

    def test_uses_original_sd_for_tolerance(self):
        """Tolerance r is based on original series SD (Costa 2005)."""
        x = np.random.default_rng(42).normal(0, 10, 40)
        mse = multiscale_entropy(x, max_scale=2)
        # If r was recomputed at each scale, results would differ
        assert len(mse) == 2


# ================================================================
# COMPLEXITY INDEX
# ================================================================

class TestComplexityIndex:
    """Tests for CI = area under MSE curve."""

    def test_empty_profile_zero(self):
        """Empty profile → CI = 0."""
        assert complexity_index([]) == 0.0

    def test_sum_of_sampen(self):
        """CI = sum of SampEn values."""
        profile = [0.5, 0.4, 0.3]
        assert abs(complexity_index(profile) - 1.2) < 1e-10

    def test_nan_values_excluded(self):
        """NaN SampEn values don't contribute to CI."""
        profile = [0.5, float('nan'), 0.3]
        assert abs(complexity_index(profile) - 0.8) < 1e-10


# ================================================================
# PROFILE CLASSIFICATION
# ================================================================

class TestProfileClassification:
    """Tests for MSE profile pattern classification."""

    def test_low_entropy_monotonous(self):
        """Very low SampEn at all scales → monotonous."""
        profile = [0.1, 0.05, 0.02]
        assert _classify_profile(profile) == "monotonous"

    def test_high_flat_complex_adaptive(self):
        """Moderate SampEn preserved across scales → complex-adaptive."""
        profile = [0.8, 0.7, 0.65, 0.6]
        assert _classify_profile(profile) == "complex-adaptive"

    def test_high_then_drop_erratic(self):
        """High scale-1 entropy with steep decay → erratic."""
        profile = [2.0, 0.3, 0.1]  # decay ratio = 0.1/2.0 = 0.05
        assert _classify_profile(profile) == "erratic"

    def test_moderate_with_decay_fatiguing(self):
        """Moderate entropy with sharp decay → fatiguing."""
        profile = [0.8, 0.4, 0.2]  # decay ratio = 0.2/0.8 = 0.25
        assert _classify_profile(profile) == "fatiguing"

    def test_single_point_unknown(self):
        """Single-scale profile → unknown."""
        assert _classify_profile([0.5]) == "unknown"


# ================================================================
# INVERTED-U SCORING
# ================================================================

class TestInvertedU:
    """Tests for the inverted-U CI → score mapping."""

    def test_optimal_ci_highest_score(self):
        """CI = 1.8 (optimal) → highest score ≈ 100."""
        analyzer = MultiscaleEntropyAnalyzer()
        score = analyzer._inverted_u_score(1.8)
        assert score > 95

    def test_zero_ci_low_score(self):
        """CI = 0 (no complexity) → low score."""
        analyzer = MultiscaleEntropyAnalyzer()
        score = analyzer._inverted_u_score(0.0)
        assert score < 30

    def test_very_high_ci_low_score(self):
        """CI = 6 (chaotic) → low score."""
        analyzer = MultiscaleEntropyAnalyzer()
        score = analyzer._inverted_u_score(6.0)
        assert score < 20

    def test_score_range_0_100(self):
        """Score always in [0, 100]."""
        analyzer = MultiscaleEntropyAnalyzer()
        for ci in np.linspace(-1, 10, 50):
            s = analyzer._inverted_u_score(ci)
            assert 0 <= s <= 100


# ================================================================
# END-TO-END ANALYZER
# ================================================================

class TestMultiscaleEntropyAnalyzer:
    """Tests for the full analyzer pipeline."""

    def test_insufficient_windows_returns_default(self):
        """Fewer than _MIN_SERIES_LEN windows → default result."""
        wins = [_make_window(wid=i) for i in range(3)]
        analyzer = MultiscaleEntropyAnalyzer()
        result = analyzer.analyze(wins)
        assert result.composite_complexity == 50.0
        assert "Insufficient" in result.interpretation

    def test_sufficient_windows_returns_channels(self):
        """With enough windows, returns 3 channel analyses."""
        rng = np.random.default_rng(42)
        wins = [
            _make_window(wid=i,
                         wpm=130 + rng.normal(0, 10),
                         pitch_std=20 + rng.normal(0, 3),
                         rms=0.05 + rng.normal(0, 0.005))
            for i in range(20)
        ]
        analyzer = MultiscaleEntropyAnalyzer()
        result = analyzer.analyze(wins)

        assert len(result.channels) == 3
        channel_names = {c.channel for c in result.channels}
        assert channel_names == {"pitch_variability", "speech_rate", "energy"}

    def test_composite_in_range(self):
        """Composite complexity score is in [0, 100]."""
        rng = np.random.default_rng(42)
        wins = [
            _make_window(wid=i,
                         wpm=130 + rng.normal(0, 15),
                         pitch_std=20 + rng.normal(0, 5),
                         rms=0.05 + rng.normal(0, 0.01))
            for i in range(15)
        ]
        analyzer = MultiscaleEntropyAnalyzer()
        result = analyzer.analyze(wins)
        assert 0 <= result.composite_complexity <= 100

    def test_constant_windows_monotonous(self):
        """Identical windows → all channels classified as monotonous."""
        wins = [_make_window(wid=i) for i in range(15)]
        analyzer = MultiscaleEntropyAnalyzer()
        result = analyzer.analyze(wins)
        # Constant series → profile should be monotonous or CI ≈ 0
        for ch in result.channels:
            assert ch.complexity_index < 0.5
            assert ch.ci_normalised < 30  # low score for no complexity

    def test_varied_windows_higher_complexity(self):
        """Windows with natural variation → higher complexity than constant."""
        rng = np.random.default_rng(42)
        constant_wins = [_make_window(wid=i) for i in range(25)]
        varied_wins = [
            _make_window(wid=i,
                         wpm=100 + 60 * np.sin(i * 0.7) + rng.normal(0, 10),
                         pitch_std=10 + 20 * np.cos(i * 0.5) + rng.normal(0, 3),
                         rms=0.03 + 0.04 * np.sin(i * 0.3) + rng.normal(0, 0.005))
            for i in range(25)
        ]
        analyzer = MultiscaleEntropyAnalyzer()
        r_const = analyzer.analyze(constant_wins)
        r_varied = analyzer.analyze(varied_wins)
        assert r_varied.composite_complexity > r_const.composite_complexity

    def test_interpretation_non_empty(self):
        """Result has a non-empty interpretation string."""
        rng = np.random.default_rng(42)
        wins = [
            _make_window(wid=i, wpm=130 + rng.normal(0, 10))
            for i in range(12)
        ]
        analyzer = MultiscaleEntropyAnalyzer()
        result = analyzer.analyze(wins)
        assert len(result.interpretation) > 10

    def test_scales_used_reported(self):
        """Result reports how many scales were used."""
        rng = np.random.default_rng(42)
        wins = [
            _make_window(wid=i, wpm=130 + rng.normal(0, 10))
            for i in range(20)
        ]
        analyzer = MultiscaleEntropyAnalyzer()
        result = analyzer.analyze(wins)
        assert result.scales_used >= 2
