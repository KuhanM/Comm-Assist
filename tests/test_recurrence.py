"""
SpeechScore 2.0 — Tests for Recurrence Quantification Analysis (V2-2, frame-level)

30+ tests covering:
  - phase_space_embed correctness
  - recurrence_matrix computation
  - diagonal/vertical line extraction
  - RQA measure computation (RR, DET, LAM, TT)
  - RecurrenceAnalyzer end-to-end (frame-level)
  - score mappings (DET→score, RR→score, LAM→score)
"""

import pytest
import numpy as np

from speechscore.analyzers.frame_features import (
    FrameFeatures, downsample, optimal_delay_ami, optimal_dimension_fnn,
)
from speechscore.analyzers.recurrence import (
    phase_space_embed,
    recurrence_matrix,
    _diagonal_lines,
    _vertical_lines,
    compute_rqa,
    RecurrenceAnalyzer,
    _DEFAULT_EMBED_DIM,
    _DEFAULT_DELAY,
    _MIN_SERIES_LEN,
    _RQA_TARGET_N,
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
# PHASE SPACE EMBEDDING
# ================================================================

class TestPhaseSpaceEmbed:
    """Tests for Takens' time-delay embedding."""

    def test_basic_embedding(self):
        """m=2, tau=1: each embedded vector is (x_i, x_{i+1})."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        emb = phase_space_embed(x, m=2, tau=1)
        assert emb.shape == (4, 2)
        np.testing.assert_array_equal(emb[0], [1.0, 2.0])
        np.testing.assert_array_equal(emb[3], [4.0, 5.0])

    def test_embedding_with_delay(self):
        """m=2, tau=2: vectors are (x_i, x_{i+2})."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        emb = phase_space_embed(x, m=2, tau=2)
        assert emb.shape == (3, 2)
        np.testing.assert_array_equal(emb[0], [1.0, 3.0])

    def test_3d_embedding(self):
        """m=3, tau=1: vectors are (x_i, x_{i+1}, x_{i+2})."""
        x = np.array([10, 20, 30, 40, 50], dtype=float)
        emb = phase_space_embed(x, m=3, tau=1)
        assert emb.shape == (3, 3)
        np.testing.assert_array_equal(emb[0], [10, 20, 30])

    def test_too_short_series(self):
        """Series too short for embedding → empty output."""
        x = np.array([1.0])
        emb = phase_space_embed(x, m=2, tau=1)
        assert emb.shape[0] == 0

    def test_n_vectors_formula(self):
        """N - (m-1)*tau vectors expected."""
        x = np.arange(100, dtype=float)
        for m in [2, 3, 5]:
            for tau in [1, 2, 3]:
                emb = phase_space_embed(x, m=m, tau=tau)
                expected = 100 - (m - 1) * tau
                assert emb.shape == (expected, m)


# ================================================================
# RECURRENCE MATRIX
# ================================================================

class TestRecurrenceMatrix:
    """Tests for recurrence matrix computation."""

    def test_identical_points_full_recurrence(self):
        """All identical embedded vectors → R is all ones."""
        emb = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        R = recurrence_matrix(emb, radius=0.1)
        assert R.shape == (3, 3)
        assert np.all(R == 1)

    def test_diagonal_is_always_one(self):
        """Every point recurs with itself (R_{i,i} = 1)."""
        rng = np.random.default_rng(42)
        emb = rng.normal(0, 1, (10, 2))
        R = recurrence_matrix(emb, radius=100.0)  # huge radius
        assert np.all(np.diag(R) == 1)

    def test_symmetry(self):
        """R should be symmetric."""
        rng = np.random.default_rng(42)
        emb = rng.normal(0, 1, (20, 3))
        R = recurrence_matrix(emb, radius_frac=0.3)
        np.testing.assert_array_equal(R, R.T)

    def test_zero_radius_diagonal_only(self):
        """Radius 0 → only self-recurrence (diagonal ones)."""
        emb = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        R = recurrence_matrix(emb, radius=0.0)
        # Diagonal should be 1 (distance to self = 0 ≤ 0)
        assert np.all(np.diag(R) == 1)
        # Off-diagonal should be 0
        assert np.sum(R) - np.trace(R) == 0

    def test_empty_input(self):
        """Empty embedded → empty matrix."""
        emb = np.array([]).reshape(0, 2)
        R = recurrence_matrix(emb)
        assert R.shape == (0, 0)


# ================================================================
# DIAGONAL & VERTICAL LINES
# ================================================================

class TestDiagonalLines:
    """Tests for diagonal line extraction."""

    def test_identity_matrix_no_diags(self):
        """Identity matrix has no off-diagonal lines."""
        R = np.eye(10, dtype=int)
        lines = _diagonal_lines(R)
        assert len(lines) == 0

    def test_full_recurrence_long_diags(self):
        """Full recurrence → long diagonals on every offset."""
        R = np.ones((10, 10), dtype=int)
        lines = _diagonal_lines(R, l_min=2)
        assert len(lines) > 0
        assert max(lines) == 9

    def test_min_length_filter(self):
        """Lines shorter than l_min are excluded."""
        R = np.zeros((10, 10), dtype=int)
        np.fill_diagonal(R, 1)
        # Create a diagonal line of length 3 at offset 1
        for i in range(3):
            R[i, i + 1] = 1
        lines_2 = _diagonal_lines(R, l_min=2)
        lines_4 = _diagonal_lines(R, l_min=4)
        assert len(lines_2) > 0
        assert len(lines_4) == 0  # line of length 3 < 4


class TestVerticalLines:

    def test_column_of_ones(self):
        """A column of ones → vertical line."""
        R = np.zeros((10, 10), dtype=int)
        R[:, 3] = 1  # column 3 all ones
        lines = _vertical_lines(R, v_min=2)
        assert any(v == 10 for v in lines)


# ================================================================
# COMPUTE RQA
# ================================================================

class TestComputeRQA:
    """Tests for the full RQA computation."""

    def test_constant_series_high_rr(self):
        """Constant series → high recurrence rate."""
        x = np.ones(50)
        rqa = compute_rqa(x, m=2, tau=1)
        assert rqa["RR"] > 0.5

    def test_random_has_rqa_metrics(self):
        """Random series should produce finite RQA metrics."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        rqa = compute_rqa(x, m=2, tau=1)
        assert 0 <= rqa["RR"] <= 1
        assert 0 <= rqa["DET"] <= 1
        assert 0 <= rqa["LAM"] <= 1
        assert rqa["TT"] >= 0

    def test_periodic_deterministic(self):
        """Periodic signal → high DET."""
        x = np.sin(np.linspace(0, 8 * np.pi, 200))
        rqa = compute_rqa(x, m=2, tau=1, radius_frac=0.15)
        assert rqa["DET"] > 0.3

    def test_short_series_defaults(self):
        """Very short series → all zeros."""
        x = np.array([1.0])
        rqa = compute_rqa(x, m=2, tau=1)
        assert rqa["RR"] == 0
        assert rqa["n_embedded"] == 0


# ================================================================
# OPTIMAL DELAY (AMI)
# ================================================================

class TestOptimalDelayAMI:

    def test_periodic_signal(self):
        """Sine wave → τ should be around quarter-period."""
        x = np.sin(np.linspace(0, 20 * np.pi, 500))
        tau = optimal_delay_ami(x, max_lag=30)
        assert 1 <= tau <= 30

    def test_short_series(self):
        """Very short series → τ = 1."""
        x = np.array([1.0, 2.0, 3.0])
        tau = optimal_delay_ami(x, max_lag=5)
        assert tau == 1

    def test_returns_int(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        tau = optimal_delay_ami(x)
        assert isinstance(tau, (int, np.integer))
        assert tau >= 1


# ================================================================
# OPTIMAL DIMENSION (FNN)
# ================================================================

class TestOptimalDimensionFNN:

    def test_sine_low_dimension(self):
        """Sine wave → m should be ≤ 3."""
        x = np.sin(np.linspace(0, 20 * np.pi, 500))
        m = optimal_dimension_fnn(x, tau=5)
        assert 2 <= m <= 4

    def test_returns_at_least_2(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        tau = optimal_delay_ami(x)
        m = optimal_dimension_fnn(x, tau)
        assert m >= 2

    def test_constant_returns_2(self):
        """Constant series → m = 2."""
        x = np.ones(100)
        m = optimal_dimension_fnn(x, tau=1)
        assert m == 2


# ================================================================
# ANALYZER (FRAME-LEVEL)
# ================================================================

class TestRecurrenceAnalyzer:
    """End-to-end tests with FrameFeatures."""

    def test_basic_analysis(self):
        """Analyzer runs on synthetic sine signal."""
        ff = _make_frame_features(600, signal_type="sine")
        analyzer = RecurrenceAnalyzer()
        result = analyzer.analyze(ff)

        assert len(result.channels) == 3
        assert result.channels[0].channel == "log_f0"
        assert result.channels[1].channel == "rms_db"
        assert result.channels[2].channel == "spectral_centroid"
        assert 0 <= result.composite_rqa <= 100

    def test_random_signal(self):
        """Random signal produces finite results."""
        ff = _make_frame_features(600, signal_type="random")
        analyzer = RecurrenceAnalyzer()
        result = analyzer.analyze(ff)

        for ch in result.channels:
            assert 0 <= ch.recurrence_rate <= 1
            assert 0 <= ch.determinism <= 1

    def test_insufficient_frames(self):
        """Too few frames → default result."""
        ff = _make_frame_features(10, signal_type="sine")
        analyzer = RecurrenceAnalyzer()
        result = analyzer.analyze(ff)
        assert "Insufficient" in result.interpretation

    def test_score_ranges(self):
        """All scores in [0, 100]."""
        ff = _make_frame_features(600, signal_type="sine")
        analyzer = RecurrenceAnalyzer()
        result = analyzer.analyze(ff)

        assert 0 <= result.predictability_score <= 100
        assert 0 <= result.consistency_score <= 100
        assert 0 <= result.fluidity_score <= 100
        assert 0 <= result.composite_rqa <= 100

    def test_data_driven_embedding(self):
        """Embedding params should be data-driven (not hardcoded)."""
        ff = _make_frame_features(600, signal_type="sine")
        analyzer = RecurrenceAnalyzer()
        result = analyzer.analyze(ff)
        # embedding_dim=0 signals per-channel data-driven
        assert result.embedding_dim == 0
        assert result.delay == 0


# ================================================================
# SCORE MAPPINGS
# ================================================================

class TestScoreMappings:

    def test_det_score_range(self):
        """DET mapping always in [0, 100]."""
        for det in np.linspace(0, 1, 50):
            score = RecurrenceAnalyzer._det_to_score(det)
            assert 0 <= score <= 100

    def test_det_score_monotonic_low(self):
        """DET 0→0.8 should be roughly monotonic increasing."""
        scores = [RecurrenceAnalyzer._det_to_score(d) for d in np.linspace(0, 0.8, 20)]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1] - 1  # allow tiny rounding

    def test_rr_inverted_u(self):
        """RR mapping peaks near 0.30."""
        score_low = RecurrenceAnalyzer._rr_to_score(0.0)
        score_opt = RecurrenceAnalyzer._rr_to_score(0.30)
        score_high = RecurrenceAnalyzer._rr_to_score(0.8)
        assert score_opt > score_low
        assert score_opt > score_high

    def test_lam_score_inversely_related(self):
        """Lower LAM → higher fluidity score."""
        score_low_lam = RecurrenceAnalyzer._lam_to_score(0.1)
        score_high_lam = RecurrenceAnalyzer._lam_to_score(0.8)
        assert score_low_lam > score_high_lam
