"""
SpeechScore 2.0 — Tests for Day 7: Composite Scoring Engine

Tests:
  - Piecewise-linear interpolation
  - Grade mapping
  - Per-category scorers (good vs poor input)
  - Composite score properties (range, weights, categories)
  - Edge cases (empty result)
"""

import pytest

from speechscore.models.schemas import (
    SpeechAnalysisResult,
    LanguageMetrics,
    TemporalMetrics,
    ConfidenceTrajectory,
    WarmupIndex,
    FatigueDetection,
    EngagementArc,
    AdaptiveScoreResult,
    SpeakerBaseline,
    CognitiveStrainResult,
    WindowMetrics,
)

from speechscore.analyzers.scoring import (
    _piecewise,
    _grade,
    compute_composite,
    CATEGORY_WEIGHTS,
)


# ────────────────────────────────────────────────────────────
# Helpers — synthetic results
# ────────────────────────────────────────────────────────────

def _good_result() -> SpeechAnalysisResult:
    """A fluent, confident speaker with good metrics."""
    windows = []
    for i in range(10):
        windows.append(WindowMetrics(
            window_id=i,
            start_time=i * 5.0,
            end_time=i * 5.0 + 10.0,
            speech_rate_wpm=135.0,
            pitch_mean=180.0,
            pitch_std=30.0,
            volume_consistency=0.85,
            rms_mean=0.05,
            pause_count=3,
            pause_frequency_per_min=18.0,
            mean_pause_duration=0.35,
            filler_count=0,
            filler_rate_per_100=1.0,
            phonation_ratio=0.72,
            asr_confidence=0.92,
            word_recognition_rate=0.96,
            word_count=22,
        ))
    return SpeechAnalysisResult(
        duration=55.0,
        total_windows=10,
        window_metrics=windows,
        global_acoustic={
            "avg_speech_rate_wpm": 135.0,
            "speech_rate_std": 10.0,
            "global_pitch_mean": 180.0,
            "global_pitch_std": 30.0,
            "global_volume_consistency": 0.85,
        },
        global_fluency={
            "total_pause_count": 30,
            "global_mean_pause_duration": 0.35,
            "total_filler_count": 3,
            "global_filler_rate_per_100": 1.0,
            "avg_phonation_ratio": 0.72,
        },
        global_clarity={
            "global_asr_confidence": 0.92,
            "global_word_recognition_rate": 0.96,
        },
        language_metrics=LanguageMetrics(
            grammar_score=0.92,
            vocabulary_richness=0.55,
            sentence_complexity=2.0,
        ),
        temporal_metrics=TemporalMetrics(
            confidence_trajectory=ConfidenceTrajectory(
                direction="increasing",
                slope=-0.3,
                p_value=0.03,
            ),
            warmup_index=WarmupIndex(
                warmup_seconds=10,
                change_point_detected=False,
            ),
            fatigue_detection=FatigueDetection(
                fatigue_score=15.0,
                significant=False,
            ),
            engagement_arc=EngagementArc(
                score=60.0,
                shape="rising",
                correlation=0.5,
            ),
        ),
        adaptive_score=AdaptiveScoreResult(
            overall_adaptive_score=72.0,
            consistency_score=85.0,
        ),
        cognitive_strain=CognitiveStrainResult(
            mean_csi=15.0,
            max_csi=35.0,
            struggle_pct=5.0,
            struggle_count=1,
        ),
    )


def _poor_result() -> SpeechAnalysisResult:
    """A speaker with poor metrics across the board."""
    windows = []
    for i in range(10):
        windows.append(WindowMetrics(
            window_id=i,
            start_time=i * 5.0,
            end_time=i * 5.0 + 10.0,
            speech_rate_wpm=200.0,
            pitch_mean=180.0,
            pitch_std=80.0,
            volume_consistency=0.30,
            rms_mean=0.02,
            pause_count=8,
            pause_frequency_per_min=48.0,
            mean_pause_duration=1.2,
            filler_count=3,
            filler_rate_per_100=8.0,
            phonation_ratio=0.40,
            asr_confidence=0.65,
            word_recognition_rate=0.75,
            word_count=33,
        ))
    return SpeechAnalysisResult(
        duration=55.0,
        total_windows=10,
        window_metrics=windows,
        global_acoustic={
            "avg_speech_rate_wpm": 200.0,
            "global_pitch_std": 80.0,
            "global_volume_consistency": 0.30,
        },
        global_fluency={
            "global_mean_pause_duration": 1.2,
            "global_filler_rate_per_100": 8.0,
            "avg_phonation_ratio": 0.40,
        },
        global_clarity={
            "global_asr_confidence": 0.65,
            "global_word_recognition_rate": 0.75,
        },
        language_metrics=LanguageMetrics(
            grammar_score=0.65,
            vocabulary_richness=0.25,
            sentence_complexity=3.5,
        ),
        temporal_metrics=TemporalMetrics(
            confidence_trajectory=ConfidenceTrajectory(
                direction="decreasing",
                p_value=0.02,
            ),
            warmup_index=WarmupIndex(
                warmup_seconds=30,
                change_point_detected=True,
            ),
            fatigue_detection=FatigueDetection(
                fatigue_score=70.0,
                significant=True,
            ),
            engagement_arc=EngagementArc(
                score=15.0,
                shape="declining",
            ),
        ),
        adaptive_score=AdaptiveScoreResult(
            overall_adaptive_score=30.0,
            consistency_score=40.0,
        ),
        cognitive_strain=CognitiveStrainResult(
            mean_csi=55.0,
            max_csi=85.0,
            struggle_pct=30.0,
        ),
    )


# ================================================================
# PIECEWISE INTERPOLATION
# ================================================================

class TestPiecewise:

    def test_below_range(self):
        bp = [(10, 50), (20, 100)]
        assert _piecewise(5, bp) == 50.0

    def test_above_range(self):
        bp = [(10, 50), (20, 100)]
        assert _piecewise(25, bp) == 100.0

    def test_at_breakpoint(self):
        bp = [(10, 50), (20, 100)]
        assert _piecewise(10, bp) == 50.0
        assert _piecewise(20, bp) == 100.0

    def test_interpolation(self):
        bp = [(10, 50), (20, 100)]
        assert abs(_piecewise(15, bp) - 75.0) < 0.01

    def test_multi_segment(self):
        bp = [(0, 0), (50, 50), (100, 100)]
        assert abs(_piecewise(25, bp) - 25.0) < 0.01

    def test_descending_scores(self):
        """Breakpoints where score decreases."""
        bp = [(0, 100), (10, 100), (20, 50), (30, 0)]
        assert _piecewise(0, bp) == 100.0
        assert _piecewise(15, bp) == 75.0
        assert _piecewise(30, bp) == 0.0


# ================================================================
# GRADE
# ================================================================

class TestGrade:

    def test_a_plus(self):
        assert _grade(95) == "A+"

    def test_a(self):
        assert _grade(87) == "A"

    def test_b_plus(self):
        assert _grade(82) == "B+"

    def test_b(self):
        assert _grade(76) == "B"

    def test_c_plus(self):
        assert _grade(71) == "C+"

    def test_c(self):
        assert _grade(66) == "C"

    def test_d(self):
        assert _grade(58) == "D"

    def test_f(self):
        assert _grade(40) == "F"


# ================================================================
# COMPOSITE SCORE
# ================================================================

class TestCompositeScore:

    def test_good_speaker_high_score(self):
        result = _good_result()
        composite = compute_composite(result)
        assert composite.composite_score >= 65
        assert composite.grade in ("A+", "A", "B+", "B", "C+")

    def test_poor_speaker_low_score(self):
        result = _poor_result()
        composite = compute_composite(result)
        assert composite.composite_score < 55
        assert composite.grade in ("D", "F")

    def test_good_beats_poor(self):
        good = compute_composite(_good_result())
        poor = compute_composite(_poor_result())
        assert good.composite_score > poor.composite_score

    def test_nine_categories(self):
        result = compute_composite(_good_result())
        assert len(result.category_scores) == 9

    def test_weights_sum_to_one(self):
        total = sum(CATEGORY_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_category_names(self):
        result = compute_composite(_good_result())
        names = {cs.category for cs in result.category_scores}
        expected = {
            "vocal_delivery", "fluency", "clarity", "language",
            "temporal_dynamics", "cognitive_load", "speaker_adaptive",
            "coherence", "listener_score",
        }
        assert names == expected

    def test_score_range(self):
        for r in [_good_result(), _poor_result()]:
            composite = compute_composite(r)
            assert 0 <= composite.composite_score <= 100
            for cs in composite.category_scores:
                assert 0 <= cs.score <= 100

    def test_has_grade(self):
        result = compute_composite(_good_result())
        assert result.grade in ("A+", "A", "B+", "B", "C+", "C", "D", "F")

    def test_has_summary(self):
        result = compute_composite(_good_result())
        assert len(result.summary) > 0

    def test_empty_result(self):
        result = compute_composite(SpeechAnalysisResult())
        assert 0 <= result.composite_score <= 100

    def test_weighted_equals_score_times_weight(self):
        result = compute_composite(_good_result())
        for cs in result.category_scores:
            expected = round(cs.score * cs.weight, 2)
            assert abs(cs.weighted - expected) < 0.1


# ================================================================
# CATEGORY SCORES
# ================================================================

class TestCategoryScores:

    def test_vocal_delivery_optimal_wpm(self):
        """WPM=135 (optimal) → high vocal delivery score."""
        result = _good_result()
        composite = compute_composite(result)
        vd = next(c for c in composite.category_scores
                  if c.category == "vocal_delivery")
        assert vd.score > 75

    def test_clarity_high_asr(self):
        """ASR=0.92 → high clarity score."""
        result = _good_result()
        composite = compute_composite(result)
        cl = next(c for c in composite.category_scores
                  if c.category == "clarity")
        assert cl.score > 85

    def test_cognitive_low_strain(self):
        """Low CSI → high cognitive load score."""
        result = _good_result()
        composite = compute_composite(result)
        cg = next(c for c in composite.category_scores
                  if c.category == "cognitive_load")
        assert cg.score > 70

    def test_poor_vocal_delivery(self):
        """WPM=200 + pitch_std=80 → low vocal delivery."""
        result = _poor_result()
        composite = compute_composite(result)
        vd = next(c for c in composite.category_scores
                  if c.category == "vocal_delivery")
        assert vd.score < 50

    def test_poor_temporal(self):
        """Decreasing confidence + high fatigue → low temporal."""
        result = _poor_result()
        composite = compute_composite(result)
        td = next(c for c in composite.category_scores
                  if c.category == "temporal_dynamics")
        assert td.score < 50

    def test_details_populated(self):
        """Category scores include sub-metric detail breakdown."""
        result = compute_composite(_good_result())
        # coherence/listener return empty details when not populated on result
        skip = {"language", "coherence", "listener_score"}
        for cs in result.category_scores:
            if cs.category not in skip:
                assert len(cs.details) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
