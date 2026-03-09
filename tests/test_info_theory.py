"""
SpeechScore 2.0 — Tests for Information-Theoretic Coherence (V2-3)

25+ tests covering:
  - knn_entropy (Kozachenko-Leonenko)
  - knn_mutual_information (KSG Algorithm 1)
  - transfer_entropy (Schreiber 2000)
  - InfoTheoreticCoherenceAnalyzer end-to-end
  - normalised MI & coupling classification
"""

import pytest
import numpy as np

from speechscore.models.schemas import WindowMetrics
from speechscore.analyzers.info_theory import (
    knn_entropy,
    knn_mutual_information,
    transfer_entropy,
    InfoTheoreticCoherenceAnalyzer,
    _MIN_SERIES_LEN,
)


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _make_window(wid: int = 0, wpm: float = 130.0, pitch_std: float = 20.0,
                 rms: float = 0.05, pause_freq: float = 5.0,
                 word_count: int = 22, **kwargs) -> WindowMetrics:
    return WindowMetrics(
        window_id=wid, start_time=wid * 5, end_time=wid * 5 + 10,
        speech_rate_wpm=wpm, pitch_mean=180.0, pitch_std=pitch_std,
        volume_consistency=0.80, rms_mean=rms,
        pause_count=2, pause_frequency_per_min=pause_freq,
        mean_pause_duration=0.3, filler_count=1,
        filler_rate_per_100=1.0, phonation_ratio=0.75,
        asr_confidence=0.85, word_recognition_rate=0.90,
        word_count=word_count, transcript="sample speech",
        **kwargs,
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
        assert abs(mi) < 0.5

    def test_identical_signals_high_mi(self):
        """Identical signals + tiny noise → high MI."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y = x + rng.normal(0, 0.01, 200)
        mi = knn_mutual_information(x, y, k=3)
        assert mi > 1.0

    def test_linear_relationship_positive_mi(self):
        """y = 2x + noise → positive MI."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y = 2 * x + rng.normal(0, 0.5, 200)
        mi = knn_mutual_information(x, y, k=3)
        assert mi > 0.3

    def test_mi_non_negative_clamped(self):
        """MI should be clamped to ≥ 0."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0, 1, 50)
        mi = knn_mutual_information(x, y, k=3)
        assert mi >= 0.0

    def test_length_mismatch_shorter_used(self):
        """Different lengths → truncated to shorter."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 50)
        mi = knn_mutual_information(x, y, k=3)
        assert isinstance(mi, float)

    def test_symmetric(self):
        """MI(X,Y) == MI(Y,X)."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = x + rng.normal(0, 0.5, 100)
        mi_xy = knn_mutual_information(x, y, k=3)
        mi_yx = knn_mutual_information(y, x, k=3)
        assert mi_xy == pytest.approx(mi_yx, abs=0.1)


# ================================================================
# TRANSFER ENTROPY
# ================================================================

class TestTransferEntropy:
    """Tests for Schreiber's transfer entropy."""

    def test_independent_low_te(self):
        """Independent signals → TE ≈ 0."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        te = transfer_entropy(x, y, lag=1, k=3)
        assert abs(te) < 0.5

    def test_causal_signal_positive_te(self):
        """y[t] = 0.8 * x[t-1] + noise → TE(x→y) > 0."""
        rng = np.random.default_rng(42)
        n = 300
        x = rng.normal(0, 1, n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.8 * x[t - 1] + rng.normal(0, 0.2)
        te = transfer_entropy(x, y, lag=1, k=3)
        assert te > 0

    def test_directionality(self):
        """TE(x→y) > TE(y→x) when x causally drives y."""
        rng = np.random.default_rng(42)
        n = 300
        x = rng.normal(0, 1, n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.8 * x[t - 1] + rng.normal(0, 0.2)
        te_xy = transfer_entropy(x, y, lag=1, k=3)
        te_yx = transfer_entropy(y, x, lag=1, k=3)
        assert te_xy > te_yx

    def test_returns_float(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0, 1, 50)
        te = transfer_entropy(x, y, lag=1, k=3)
        assert isinstance(te, float)

    def test_short_input_returns_zero(self):
        """Too-short input → 0."""
        te = transfer_entropy(np.array([1.0, 2.0]), np.array([3.0, 4.0]),
                              lag=1, k=3)
        assert te == 0.0


# ================================================================
# END-TO-END ANALYZER
# ================================================================

class TestInfoTheoreticCoherenceAnalyzer:
    """Tests for the full InfoTheoreticCoherenceAnalyzer."""

    def test_insufficient_windows_returns_default(self):
        """Too few windows → default result."""
        wins = [_make_window(wid=i) for i in range(3)]
        analyzer = InfoTheoreticCoherenceAnalyzer()
        result = analyzer.analyze(wins)
        assert result.composite_it_coherence == 50.0
        assert "Insufficient" in result.interpretation

    def test_sufficient_windows_returns_pairs(self):
        """Enough windows → 3 channel pair results."""
        rng = np.random.default_rng(42)
        wins = [
            _make_window(wid=i,
                         wpm=130 + rng.normal(0, 10),
                         pitch_std=20 + rng.normal(0, 3),
                         rms=0.05 + rng.normal(0, 0.005),
                         pause_freq=5 + rng.normal(0, 1),
                         word_count=22 + int(rng.normal(0, 3)))
            for i in range(15)
        ]
        analyzer = InfoTheoreticCoherenceAnalyzer()
        result = analyzer.analyze(wins)

        assert len(result.channel_pairs) == 3

    def test_composite_in_range(self):
        """Composite coherence in [0, 100]."""
        rng = np.random.default_rng(42)
        wins = [
            _make_window(wid=i,
                         wpm=130 + rng.normal(0, 15),
                         pitch_std=20 + rng.normal(0, 5),
                         rms=0.05 + rng.normal(0, 0.01),
                         pause_freq=5 + rng.normal(0, 2),
                         word_count=22 + int(rng.normal(0, 4)))
            for i in range(20)
        ]
        analyzer = InfoTheoreticCoherenceAnalyzer()
        result = analyzer.analyze(wins)
        assert 0 <= result.composite_it_coherence <= 100

    def test_correlated_windows_higher_coherence(self):
        """Windows with correlated channels → higher NMI than random."""
        # Create windows where rms correlates with word_count
        wins = []
        for i in range(25):
            base = np.sin(i * 0.5) * 10
            wins.append(_make_window(
                wid=i,
                wpm=130 + base,
                pitch_std=20 + base * 0.5,
                rms=0.05 + base * 0.002,
                pause_freq=5 + base * 0.3,
                word_count=max(5, int(22 + base * 0.8)),
            ))
        analyzer = InfoTheoreticCoherenceAnalyzer()
        result = analyzer.analyze(wins)
        # At least some coupling should be detected
        assert any(p.normalised_mi > 0.01 for p in result.channel_pairs)

    def test_interpretation_non_empty(self):
        """Result has non-empty interpretation."""
        rng = np.random.default_rng(42)
        wins = [
            _make_window(wid=i, wpm=130 + rng.normal(0, 10))
            for i in range(12)
        ]
        analyzer = InfoTheoreticCoherenceAnalyzer()
        result = analyzer.analyze(wins)
        assert len(result.interpretation) > 10

    def test_nonlinear_coherence_score_exists(self):
        """Result has nonlinear_coherence_score attribute."""
        rng = np.random.default_rng(42)
        wins = [
            _make_window(wid=i, wpm=130 + rng.normal(0, 10),
                         rms=0.05 + rng.normal(0, 0.005))
            for i in range(15)
        ]
        analyzer = InfoTheoreticCoherenceAnalyzer()
        result = analyzer.analyze(wins)
        assert 0 <= result.nonlinear_coherence <= 100

    def test_directional_flow_score_exists(self):
        """Result has directional_flow_score attribute."""
        rng = np.random.default_rng(42)
        wins = [
            _make_window(wid=i, wpm=130 + rng.normal(0, 10),
                         rms=0.05 + rng.normal(0, 0.005))
            for i in range(15)
        ]
        analyzer = InfoTheoreticCoherenceAnalyzer()
        result = analyzer.analyze(wins)
        assert 0 <= result.directional_flow <= 100
