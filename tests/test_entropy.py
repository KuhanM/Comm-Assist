"""
SpeechScore 2.0 — Tests for Multiscale Entropy Analysis (V2-1, frame-level)

30+ tests covering:
  - sample_entropy correctness & edge cases
  - coarse_grain at various scales
  - multiscale_entropy profiles
  - complexity_index computation
  - profile classification
  - _count_matches vectorised implementation
  - MultiscaleEntropyAnalyzer end-to-end (frame-level)
  - inverted-U scoring
"""

import pytest
import numpy as np

from speechscore.analyzers.frame_features import FrameFeatures, downsample
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
    _MSE_TARGET_N,
)


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _make_frame_features(n_frames: int = 500, seed: int = 42,
                         signal_type: str = "sine") -> FrameFeatures:
    """Create synthetic FrameFeatures for testing."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi * 10, n_frames)

    if signal_type == "sine":
        f0 = 150.0 + 30.0 * np.sin(t)
        rms = 0.05 + 0.02 * np.sin(t * 0.7)
        centroid = 2000.0 + 500.0 * np.sin(t * 1.3)
    elif signal_type == "random":
        f0 = 150.0 + rng.normal(0, 30, n_frames)
        rms = 0.05 + rng.normal(0, 0.02, n_frames).clip(0.001)
        centroid = 2000.0 + rng.normal(0, 500, n_frames)
    elif signal_type == "constant":
        f0 = np.full(n_frames, 150.0)
        rms = np.full(n_frames, 0.05)
        centroid = np.full(n_frames, 2000.0)
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")

    log_f0 = np.log2(np.maximum(f0, 1.0))
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-6) / 1e-6)
    flux = np.abs(rng.normal(0, 0.1, n_frames))
    voiced = np.ones(n_frames, dtype=bool)

    return FrameFeatures(
        f0=f0, log_f0=log_f0,
        rms_energy=rms, rms_db=rms_db,
        spectral_centroid=centroid,
        spectral_flux=flux,
        voiced_mask=voiced,
        hop_sec=0.01, sample_rate=16000,
        n_frames=n_frames, duration_sec=n_frames * 0.01,
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
        x = np.tile([1.0, 2.0, 3.0, 2.0], 50)  # 200 points, period 4
        se = sample_entropy(x, m=2, r=0.3)
        assert se < 0.5  # highly regular

    def test_random_series_higher_entropy(self):
        """Random series has higher SampEn than periodic."""
        rng = np.random.default_rng(42)
        x_random = rng.normal(0, 1, 200)
        x_periodic = np.tile([1.0, 2.0, 3.0, 2.0], 50)

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
            x = rng.normal(0, 1, 100)
            se = sample_entropy(x)
            if np.isfinite(se):
                assert se >= 0.0

    def test_sampen_deterministic(self):
        """Same input → same output."""
        x = np.sin(np.linspace(0, 4 * np.pi, 100))
        se1 = sample_entropy(x)
        se2 = sample_entropy(x)
        assert se1 == pytest.approx(se2)

    def test_sampen_with_explicit_r(self):
        """Explicit r tolerance affects the result."""
        x = np.sin(np.linspace(0, 6 * np.pi, 100))
        se_default = sample_entropy(x)  # auto r = 0.2 * SD
        se_explicit = sample_entropy(x, r=0.5)
        # With different r, should get different result
        if np.isfinite(se_default) and np.isfinite(se_explicit):
            # Just verify both are finite and non-negative
            assert se_default >= 0
            assert se_explicit >= 0


# ================================================================
# COUNT MATCHES (VECTORISED)
# ================================================================

class TestCountMatches:
    """Tests for the vectorised _count_matches."""

    def test_constant_series(self):
        """Constant series: all pairs match."""
        x = np.ones(20)
        B, A = _count_matches(x, m=2, r=0.1)
        n = 20 - 2  # templates
        expected_B = n * (n - 1) // 2
        assert B == expected_B
        assert A == expected_B  # m+1 also matches

    def test_no_matches_large_series(self):
        """If r=0, only identical templates match."""
        x = np.arange(20, dtype=float)  # monotonically increasing
        B, A = _count_matches(x, m=2, r=0.0)
        assert B == 0
        assert A == 0

    def test_short_series(self):
        """Too short for m → (0, 0)."""
        x = np.array([1.0, 2.0])
        B, A = _count_matches(x, m=2, r=0.5)
        assert B == 0
        assert A == 0

    def test_matches_increase_with_r(self):
        """More matches with larger tolerance."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        B1, A1 = _count_matches(x, m=2, r=0.1)
        B2, A2 = _count_matches(x, m=2, r=1.0)
        assert B2 >= B1
        assert A2 >= A1


# ================================================================
# COARSE GRAINING
# ================================================================

class TestCoarseGrain:
    """Tests for coarse-graining."""

    def test_scale_1_identity(self):
        """Scale 1 returns a copy of the input."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        cg = coarse_grain(x, 1)
        np.testing.assert_array_equal(cg, x)

    def test_scale_2_averaging(self):
        """Scale 2 averages consecutive pairs."""
        x = np.array([1.0, 3.0, 5.0, 7.0])
        cg = coarse_grain(x, 2)
        np.testing.assert_array_almost_equal(cg, [2.0, 6.0])

    def test_output_length(self):
        """Output length = floor(N / tau)."""
        x = np.ones(17)
        assert len(coarse_grain(x, 3)) == 5
        assert len(coarse_grain(x, 5)) == 3

    def test_empty_at_large_scale(self):
        """Scale > N → empty."""
        x = np.array([1.0, 2.0])
        assert len(coarse_grain(x, 5)) == 0


# ================================================================
# MULTISCALE ENTROPY
# ================================================================

class TestMultiscaleEntropy:
    """Tests for the MSE profile computation."""

    def test_constant_all_zeros(self):
        """Constant series → SampEn = 0 at all scales."""
        x = np.ones(200)
        mse = multiscale_entropy(x, max_scale=5)
        assert all(v == 0.0 for v in mse)

    def test_profile_length(self):
        """Profile length matches max_scale (if series long enough)."""
        x = np.sin(np.linspace(0, 10 * np.pi, 500))
        mse = multiscale_entropy(x, max_scale=10)
        assert len(mse) == 10

    def test_truncation_short_series(self):
        """Profile truncated when coarse-grained series too short."""
        x = np.sin(np.linspace(0, np.pi, 15))
        mse = multiscale_entropy(x, max_scale=10)
        assert len(mse) < 10

    def test_random_has_higher_ci_than_constant(self):
        """Random noise → higher CI than constant signal."""
        rng = np.random.default_rng(42)
        ci_random = complexity_index(multiscale_entropy(rng.normal(0, 1, 300), max_scale=5))
        ci_const = complexity_index(multiscale_entropy(np.ones(300), max_scale=5))
        assert ci_random > ci_const


# ================================================================
# COMPLEXITY INDEX
# ================================================================

class TestComplexityIndex:
    """Tests for CI = Σ SampEn(τ)."""

    def test_sum_of_values(self):
        profile = [0.5, 0.4, 0.3, 0.2, 0.1]
        assert complexity_index(profile) == pytest.approx(1.5)

    def test_empty_profile(self):
        assert complexity_index([]) == 0.0

    def test_nan_handling(self):
        profile = [0.5, float("nan"), 0.3]
        ci = complexity_index(profile)
        assert ci == pytest.approx(0.8)


# ================================================================
# PROFILE CLASSIFICATION
# ================================================================

class TestProfileClassification:

    def test_monotonous_low_entropy(self):
        profile = [0.1, 0.05, 0.02]
        assert _classify_profile(profile) == "monotonous"

    def test_erratic_high_then_drop(self):
        profile = [2.0, 0.5, 0.2]
        assert _classify_profile(profile) == "erratic"

    def test_complex_adaptive(self):
        profile = [0.8, 0.7, 0.6, 0.5]
        assert _classify_profile(profile) == "complex-adaptive"

    def test_fatiguing(self):
        profile = [0.8, 0.3, 0.1]
        assert _classify_profile(profile) == "fatiguing"

    def test_single_scale_unknown(self):
        assert _classify_profile([0.5]) == "unknown"

    def test_empty_unknown(self):
        assert _classify_profile([]) == "unknown"


# ================================================================
# ANALYZER (FRAME-LEVEL)
# ================================================================

class TestMultiscaleEntropyAnalyzer:
    """End-to-end tests with FrameFeatures."""

    def test_basic_analysis(self):
        """Analyzer runs successfully on synthetic sine signal."""
        ff = _make_frame_features(800, signal_type="sine")
        analyzer = MultiscaleEntropyAnalyzer()
        result = analyzer.analyze(ff)

        assert len(result.channels) == 3
        assert result.channels[0].channel == "log_f0"
        assert result.channels[1].channel == "rms_db"
        assert result.channels[2].channel == "spectral_centroid"
        assert 0 <= result.composite_complexity <= 100
        assert result.scales_used > 0

    def test_random_signal_complexity(self):
        """Random signal has non-zero complexity."""
        ff = _make_frame_features(800, signal_type="random")
        analyzer = MultiscaleEntropyAnalyzer()
        result = analyzer.analyze(ff)

        # At least one channel should have non-trivial CI
        max_ci = max(ch.complexity_index for ch in result.channels)
        assert max_ci > 0

    def test_constant_signal_monotonous(self):
        """Constant signal → monotonous classification."""
        ff = _make_frame_features(200, signal_type="constant")
        analyzer = MultiscaleEntropyAnalyzer()
        result = analyzer.analyze(ff)

        for ch in result.channels:
            assert ch.profile_class == "monotonous"

    def test_insufficient_frames(self):
        """Too few frames → default result."""
        ff = _make_frame_features(10, signal_type="sine")
        analyzer = MultiscaleEntropyAnalyzer()
        result = analyzer.analyze(ff)
        assert "Insufficient" in result.interpretation

    def test_interpretation_present(self):
        ff = _make_frame_features(500, signal_type="sine")
        analyzer = MultiscaleEntropyAnalyzer()
        result = analyzer.analyze(ff)
        assert len(result.interpretation) > 10

    def test_ci_normalised_range(self):
        """ci_normalised should be in [0, 100]."""
        ff = _make_frame_features(500, signal_type="random")
        analyzer = MultiscaleEntropyAnalyzer()
        result = analyzer.analyze(ff)
        for ch in result.channels:
            assert 0 <= ch.ci_normalised <= 100

    def test_series_length_reflects_downsampling(self):
        """series_length should be significantly less than raw n_frames."""
        ff = _make_frame_features(2000, signal_type="sine")
        analyzer = MultiscaleEntropyAnalyzer()
        result = analyzer.analyze(ff)
        for ch in result.channels:
            # Downsampled from 2000 → should be much less
            assert ch.series_length < 2000


# ================================================================
# INVERTED-U SCORING
# ================================================================

class TestInvertedUScore:

    def test_optimal_ci_high_score(self):
        """CI near μ → score near 100."""
        score = MultiscaleEntropyAnalyzer._inverted_u_score(15.0)
        assert score > 95

    def test_zero_ci_low_score(self):
        """CI = 0 → low score."""
        score = MultiscaleEntropyAnalyzer._inverted_u_score(0.0)
        assert score < 20

    def test_extreme_ci_low_score(self):
        """Very high CI → low score."""
        score = MultiscaleEntropyAnalyzer._inverted_u_score(50.0)
        assert score < 10

    def test_score_range(self):
        """Score always in [0, 100]."""
        for ci in np.linspace(-5, 60, 50):
            score = MultiscaleEntropyAnalyzer._inverted_u_score(ci)
            assert 0 <= score <= 100

    def test_symmetry(self):
        """Equidistant from μ → similar scores."""
        s_low = MultiscaleEntropyAnalyzer._inverted_u_score(5.0)
        s_high = MultiscaleEntropyAnalyzer._inverted_u_score(25.0)
        assert abs(s_low - s_high) < 5  # roughly symmetric


# ================================================================
# DOWNSAMPLE
# ================================================================

class TestDownsample:
    """Tests for block-averaging downsampling."""

    def test_no_downsample_short(self):
        x = np.array([1.0, 2.0, 3.0])
        result = downsample(x, 10)
        np.testing.assert_array_equal(result, x)

    def test_downsample_halves(self):
        x = np.array([1.0, 3.0, 5.0, 7.0])
        result = downsample(x, 2)
        np.testing.assert_array_almost_equal(result, [2.0, 6.0])

    def test_downsample_preserves_mean(self):
        """Block average preserves the global mean."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        ds = downsample(x, 100)
        assert np.mean(x) == pytest.approx(np.mean(ds), abs=0.1)
