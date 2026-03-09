"""
SpeechScore 2.0 — Tests for Recurrence Quantification Analysis (V2-2)

30+ tests covering:
  - phase_space_embed correctness
  - recurrence_matrix computation
  - diagonal/vertical line extraction
  - RQA measure computation (RR, DET, LAM, TT)
  - RecurrenceAnalyzer end-to-end
  - score mappings (DET→score, RR→score, LAM→score)
"""

import pytest
import numpy as np

from speechscore.models.schemas import WindowMetrics
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
        R = recurrence_matrix(emb, radius_frac=0.5)
        np.testing.assert_array_equal(np.diag(R), np.ones(10))

    def test_symmetry(self):
        """R is symmetric: R_{i,j} = R_{j,i}."""
        rng = np.random.default_rng(42)
        emb = rng.normal(0, 1, (10, 2))
        R = recurrence_matrix(emb, radius_frac=0.3)
        np.testing.assert_array_equal(R, R.T)

    def test_larger_radius_more_recurrence(self):
        """Larger radius → more recurrence points."""
        rng = np.random.default_rng(42)
        emb = rng.normal(0, 1, (10, 2))
        R_small = recurrence_matrix(emb, radius_frac=0.1)
        R_large = recurrence_matrix(emb, radius_frac=0.5)
        assert np.sum(R_large) >= np.sum(R_small)

    def test_empty_input(self):
        """Empty embedded array → empty matrix."""
        emb = np.array([]).reshape(0, 2)
        R = recurrence_matrix(emb)
        assert R.shape == (0, 0)


# ================================================================
# DIAGONAL & VERTICAL LINES
# ================================================================

class TestLineExtraction:
    """Tests for diagonal and vertical line extraction."""

    def test_no_diagonals_in_identity(self):
        """Identity matrix has no off-diagonal lines."""
        R = np.eye(5, dtype=int)
        lines = _diagonal_lines(R, l_min=2)
        assert len(lines) == 0

    def test_full_matrix_has_long_diagonals(self):
        """All-ones matrix has maximum-length diagonals."""
        R = np.ones((5, 5), dtype=int)
        diag_lines = _diagonal_lines(R, l_min=2)
        assert len(diag_lines) > 0
        assert max(diag_lines) == 4  # longest off-diagonal

    def test_vertical_lines_in_column_of_ones(self):
        """A column of consecutive 1s → vertical line detected."""
        R = np.zeros((6, 6), dtype=int)
        R[1:5, 2] = 1  # vertical line of length 4 in column 2
        vert_lines = _vertical_lines(R, v_min=2)
        assert 4 in vert_lines

    def test_l_min_filters_short_lines(self):
        """Lines shorter than l_min are excluded."""
        R = np.eye(5, dtype=int)
        # Add a diagonal of length 1
        R[0, 1] = 1
        lines = _diagonal_lines(R, l_min=2)
        assert all(l >= 2 for l in lines)


# ================================================================
# RQA MEASURES
# ================================================================

class TestRQAMeasures:
    """Tests for the full compute_rqa function."""

    def test_constant_series_high_rr(self):
        """Constant series → high recurrence rate (everything recurs)."""
        x = np.ones(20)
        rqa = compute_rqa(x)
        assert rqa["RR"] > 0.8

    def test_rr_range(self):
        """RR ∈ [0, 1]."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 25)
        rqa = compute_rqa(x)
        assert 0 <= rqa["RR"] <= 1

    def test_det_range(self):
        """DET ∈ [0, 1]."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 25)
        rqa = compute_rqa(x)
        assert 0 <= rqa["DET"] <= 1

    def test_lam_range(self):
        """LAM ∈ [0, 1]."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 25)
        rqa = compute_rqa(x)
        assert 0 <= rqa["LAM"] <= 1

    def test_periodic_series_high_det(self):
        """Highly periodic series → high determinism."""
        x = np.tile([1.0, 5.0, 1.0, 5.0], 8)  # 32 pts, period 4
        rqa = compute_rqa(x, radius_frac=0.15)
        assert rqa["DET"] > 0.3

    def test_random_series_lower_det(self):
        """Random series → lower DET than periodic."""
        rng = np.random.default_rng(42)
        x_rand = rng.normal(0, 1, 30)
        x_per = np.tile([1.0, 5.0, 1.0, 5.0, 1.0], 6)  # 30 pts

        rqa_rand = compute_rqa(x_rand, radius_frac=0.25)
        rqa_per = compute_rqa(x_per, radius_frac=0.25)

        assert rqa_per["DET"] >= rqa_rand["DET"]

    def test_short_series_returns_zeros(self):
        """Too-short series → all measures zero."""
        x = np.array([1.0])
        rqa = compute_rqa(x)
        assert rqa["RR"] == 0.0
        assert rqa["DET"] == 0.0

    def test_returns_correct_keys(self):
        """compute_rqa returns all expected keys."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 20)
        rqa = compute_rqa(x)
        expected_keys = {"RR", "DET", "LAM", "TT", "max_diagonal",
                         "entropy_diag", "n_embedded", "radius_used"}
        assert set(rqa.keys()) == expected_keys


# ================================================================
# SCORE MAPPINGS
# ================================================================

class TestScoreMappings:
    """Tests for DET/RR/LAM → score conversions."""

    def test_det_zero_gives_zero(self):
        assert RecurrenceAnalyzer._det_to_score(0.0) == 0.0

    def test_det_mid_gives_high(self):
        """DET = 0.7 → score > 70."""
        s = RecurrenceAnalyzer._det_to_score(0.7)
        assert s > 70

    def test_det_very_high_slightly_penalised(self):
        """DET = 0.99 → score < DET = 0.7."""
        s_99 = RecurrenceAnalyzer._det_to_score(0.99)
        s_70 = RecurrenceAnalyzer._det_to_score(0.7)
        assert s_99 < s_70

    def test_rr_optimal_score_high(self):
        """RR = 0.30 (optimal) → highest score ≈ 100."""
        s = RecurrenceAnalyzer._rr_to_score(0.30)
        assert s > 95

    def test_rr_extreme_low_score(self):
        """RR = 0 or 1 → low score."""
        s0 = RecurrenceAnalyzer._rr_to_score(0.0)
        s1 = RecurrenceAnalyzer._rr_to_score(1.0)
        assert s0 < 30
        assert s1 < 10

    def test_lam_zero_gives_full_fluidity(self):
        """LAM = 0 → fluidity 100."""
        s = RecurrenceAnalyzer._lam_to_score(0.0)
        assert s == 100.0

    def test_lam_high_gives_low_fluidity(self):
        """LAM = 0.9 → very low fluidity."""
        s = RecurrenceAnalyzer._lam_to_score(0.9)
        assert s < 20


# ================================================================
# END-TO-END ANALYZER
# ================================================================

class TestRecurrenceAnalyzer:
    """Tests for the full RecurrenceAnalyzer pipeline."""

    def test_insufficient_windows_returns_default(self):
        """Too few windows → default result."""
        wins = [_make_window(wid=i) for i in range(3)]
        analyzer = RecurrenceAnalyzer()
        result = analyzer.analyze(wins)
        assert result.composite_rqa == 50.0
        assert "Insufficient" in result.interpretation

    def test_sufficient_windows_returns_channels(self):
        """Enough windows → 3 channel results."""
        rng = np.random.default_rng(42)
        wins = [
            _make_window(wid=i,
                         wpm=130 + rng.normal(0, 10),
                         pitch_std=20 + rng.normal(0, 3),
                         rms=0.05 + rng.normal(0, 0.005))
            for i in range(15)
        ]
        analyzer = RecurrenceAnalyzer()
        result = analyzer.analyze(wins)

        assert len(result.channels) == 3
        names = {c.channel for c in result.channels}
        assert names == {"pitch_variability", "speech_rate", "energy"}

    def test_composite_in_range(self):
        """Composite RQA score in [0, 100]."""
        rng = np.random.default_rng(42)
        wins = [
            _make_window(wid=i,
                         wpm=130 + rng.normal(0, 15),
                         pitch_std=20 + rng.normal(0, 5),
                         rms=0.05 + rng.normal(0, 0.01))
            for i in range(20)
        ]
        analyzer = RecurrenceAnalyzer()
        result = analyzer.analyze(wins)
        assert 0 <= result.composite_rqa <= 100

    def test_periodic_windows_high_predictability(self):
        """Windows with periodic values → higher predictability."""
        # Alternate between two states
        wins = []
        for i in range(20):
            wpm = 120.0 if i % 2 == 0 else 140.0
            pitch = 18.0 if i % 2 == 0 else 22.0
            rms = 0.04 if i % 2 == 0 else 0.06
            wins.append(_make_window(wid=i, wpm=wpm, pitch_std=pitch, rms=rms))

        analyzer = RecurrenceAnalyzer()
        result = analyzer.analyze(wins)
        assert result.predictability_score > 30

    def test_interpretation_non_empty(self):
        """Result has non-empty interpretation."""
        rng = np.random.default_rng(42)
        wins = [
            _make_window(wid=i, wpm=130 + rng.normal(0, 10))
            for i in range(12)
        ]
        analyzer = RecurrenceAnalyzer()
        result = analyzer.analyze(wins)
        assert len(result.interpretation) > 10
