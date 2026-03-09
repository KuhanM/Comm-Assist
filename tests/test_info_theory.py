"""
SpeechScore 2.0 — Tests for Information-Theoretic Coherence (V2-3, frame-level)

25+ tests covering:
  - knn_entropy (Kozachenko-Leonenko)
  - knn_mutual_information (KSG Algorithm 1)
  - transfer_entropy (Schreiber 2000)
  - InfoTheoreticCoherenceAnalyzer end-to-end (frame-level)
  - normalised MI & coupling classification
"""

import pytest
import numpy as np

from speechscore.analyzers.frame_features import FrameFeatures, downsample
from speechscore.analyzers.info_theory import (
    knn_entropy,
    knn_mutual_information,
    transfer_entropy,
    InfoTheoreticCoherenceAnalyzer,
    _MIN_SERIES_LEN,
    _IT_TARGET_N,
)


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _make_frame_features(n_frames: int = 500, seed: int = 42,
                         signal_type: str = "sine",
                         coupling: float = 0.0) -> FrameFeatures:
    """
    Create synthetic FrameFeatures for testing.

    coupling > 0: inject cross-channel dependency (F0 drives RMS).
    """
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
    elif signal_type == "coupled":
        f0 = 150.0 + 30.0 * np.sin(t)
        # RMS is a lagged, noisy copy of F0
        rms = 0.05 + coupling * 0.001 * np.roll(f0 - 150.0, 5) + rng.normal(0, 0.005, n_frames)
        rms = rms.clip(0.001)
        centroid = 2000.0 + coupling * 10 * (f0 - 150.0) + rng.normal(0, 100, n_frames)
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
# KNN-ENTROPY (Kozachenko-Leonenko)
# ================================================================

class TestKNNEntropy:
    """Tests for the k-NN entropy estimator."""

    def test_low_entropy_constant(self):
        """Constant + small noise → low entropy."""
        rng = np.random.default_rng(42)
        x = 5.0 + rng.normal(0, 0.001, 100)
        h = knn_entropy(x, k=3)
        assert h < 0  # very low entropy (log of tiny distances)

    def test_wider_spread_higher_entropy(self):
        """Wider spread → higher entropy."""
        rng = np.random.default_rng(42)
        x_narrow = rng.normal(0, 0.1, 100)
        x_wide = rng.normal(0, 10.0, 100)
        h_narrow = knn_entropy(x_narrow, k=3)
        h_wide = knn_entropy(x_wide, k=3)
        assert h_wide > h_narrow

    def test_deterministic_same_seed(self):
        """Same input → same output."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        h1 = knn_entropy(x, k=3)
        h2 = knn_entropy(x, k=3)
        assert h1 == pytest.approx(h2)

    def test_returns_float(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 30)
        h = knn_entropy(x, k=3)
        assert isinstance(h, float)

    def test_empty_input(self):
        """Empty array → 0."""
        h = knn_entropy(np.array([]))
        assert h == 0.0


# ================================================================
# KNN MUTUAL INFORMATION (KSG Algorithm 1)
# ================================================================

class TestKNNMutualInformation:
    """Tests for the k-NN mutual information estimator."""

    def test_independent_signals_low_mi(self):
        """Independent random signals → MI ≈ 0."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        mi = knn_mutual_information(x, y, k=3)
        assert mi < 0.3  # near zero

    def test_identical_signals_high_mi(self):
        """Identical signals → high MI."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        mi = knn_mutual_information(x, x, k=3)
        assert mi > 1.0

    def test_correlated_higher_than_independent(self):
        """Correlated pair → higher MI than independent pair."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y_corr = x + rng.normal(0, 0.1, 200)
        y_ind = rng.normal(0, 1, 200)

        mi_corr = knn_mutual_information(x, y_corr, k=3)
        mi_ind = knn_mutual_information(x, y_ind, k=3)
        assert mi_corr > mi_ind

    def test_mi_non_negative(self):
        """MI should be ≥ 0."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            x = rng.normal(0, 1, 100)
            y = rng.normal(0, 1, 100)
            mi = knn_mutual_information(x, y, k=3)
            assert mi >= 0

    def test_short_series_returns_zero(self):
        """Too short → 0."""
        x = np.array([1.0])
        y = np.array([2.0])
        mi = knn_mutual_information(x, y, k=3)
        assert mi == 0.0


# ================================================================
# TRANSFER ENTROPY
# ================================================================

class TestTransferEntropy:
    """Tests for the transfer entropy estimator."""

    def test_independent_low_te(self):
        """Independent signals → TE ≈ 0."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        te = transfer_entropy(x, y, k=3)
        assert te < 0.2

    def test_causal_signal_positive_te(self):
        """X causes Y → TE(X→Y) should be positive."""
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.normal(0, 1, 300))
        # y follows x with a lag
        y = np.zeros(300)
        for i in range(1, 300):
            y[i] = 0.8 * x[i - 1] + 0.2 * rng.normal()
        te = transfer_entropy(x, y, lag=1, k=3)
        assert te > 0

    def test_te_non_negative(self):
        """TE should be ≥ 0."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        te = transfer_entropy(x, y, k=3)
        assert te >= 0

    def test_short_series_returns_zero(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        assert transfer_entropy(x, y) == 0.0


# ================================================================
# ANALYZER (FRAME-LEVEL)
# ================================================================

class TestInfoTheoreticCoherenceAnalyzer:
    """End-to-end tests with FrameFeatures."""

    def test_basic_analysis(self):
        """Analyzer runs on synthetic sine signal."""
        ff = _make_frame_features(600, signal_type="sine")
        analyzer = InfoTheoreticCoherenceAnalyzer()
        result = analyzer.analyze(ff)

        assert len(result.channel_pairs) == 3
        assert result.channel_pairs[0].channel_x == "log_f0"
        assert result.channel_pairs[0].channel_y == "rms_db"
        assert result.channel_pairs[1].channel_x == "log_f0"
        assert result.channel_pairs[1].channel_y == "spectral_centroid"
        assert result.channel_pairs[2].channel_x == "rms_db"
        assert result.channel_pairs[2].channel_y == "spectral_flux"
        assert 0 <= result.composite_it_coherence <= 100

    def test_coupled_higher_coherence(self):
        """Strongly coupled channels → higher coherence than independent."""
        ff_coupled = _make_frame_features(600, signal_type="coupled", coupling=5.0)
        ff_random = _make_frame_features(600, signal_type="random")

        analyzer = InfoTheoreticCoherenceAnalyzer()
        result_coupled = analyzer.analyze(ff_coupled)
        result_random = analyzer.analyze(ff_random)

        # Coupled should have higher NMI on at least the F0-centroid pair
        coupled_nmi = result_coupled.channel_pairs[1].normalised_mi
        random_nmi = result_random.channel_pairs[1].normalised_mi
        assert coupled_nmi > random_nmi

    def test_insufficient_frames(self):
        """Too few frames → default result."""
        ff = _make_frame_features(10, signal_type="sine")
        analyzer = InfoTheoreticCoherenceAnalyzer()
        result = analyzer.analyze(ff)
        assert "Insufficient" in result.interpretation

    def test_score_ranges(self):
        """All scores in [0, 100]."""
        ff = _make_frame_features(600, signal_type="random")
        analyzer = InfoTheoreticCoherenceAnalyzer()
        result = analyzer.analyze(ff)

        assert 0 <= result.nonlinear_coherence <= 100
        assert 0 <= result.directional_flow <= 100
        assert 0 <= result.composite_it_coherence <= 100

    def test_coupling_strength_labels(self):
        """Coupling strength labels make sense."""
        ff = _make_frame_features(600, signal_type="sine")
        analyzer = InfoTheoreticCoherenceAnalyzer()
        result = analyzer.analyze(ff)

        valid_labels = {"strong", "moderate", "weak", "none"}
        for pair in result.channel_pairs:
            assert pair.coupling_strength in valid_labels

    def test_direction_labels(self):
        """Direction labels are valid strings."""
        ff = _make_frame_features(600, signal_type="sine")
        analyzer = InfoTheoreticCoherenceAnalyzer()
        result = analyzer.analyze(ff)

        for pair in result.channel_pairs:
            assert pair.dominant_direction  # non-empty string

    def test_interpretation_present(self):
        ff = _make_frame_features(600, signal_type="sine")
        analyzer = InfoTheoreticCoherenceAnalyzer()
        result = analyzer.analyze(ff)
        assert len(result.interpretation) > 10


# ================================================================
# SCORE MAPPINGS
# ================================================================

class TestScoreMappings:

    def test_nmi_score_range(self):
        """NMI score always in [0, 100]."""
        for nmi in np.linspace(0, 1, 20):
            score = InfoTheoreticCoherenceAnalyzer._nmi_to_score(nmi)
            assert 0 <= score <= 100

    def test_nmi_score_monotonic(self):
        """NMI score increases with NMI."""
        scores = [InfoTheoreticCoherenceAnalyzer._nmi_to_score(n) for n in np.linspace(0, 1, 20)]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1]

    def test_te_score_range(self):
        """TE score always in [0, 100]."""
        for te in np.linspace(0, 2, 20):
            score = InfoTheoreticCoherenceAnalyzer._te_to_score(te)
            assert 0 <= score <= 100
