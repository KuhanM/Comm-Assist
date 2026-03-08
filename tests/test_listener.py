"""
Tests for the Listener Prediction module.

Tests cover:
  - Comprehension prediction
  - Engagement prediction
  - Trust prediction
  - Retention (multiplicative)
  - Attention sustainability
  - Overall score calculation
  - Edge cases (missing data, no temporal/adaptive)
"""

import pytest
import numpy as np

from speechscore.models.schemas import (
    SpeechAnalysisResult,
    WindowMetrics,
    TranscriptionResult,
    LanguageMetrics,
    TemporalMetrics,
    ConfidenceTrajectory,
    WarmupIndex,
    FatigueDetection,
    EngagementArc,
    AdaptiveScoreResult,
    SpeakerBaseline,
    CognitiveStrainResult,
    CoherenceResult,
    ListenerPrediction,
)
from speechscore.analyzers.listener import ListenerPredictor


# ── Helpers ─────────────────────────────────────────────────────

def _make_window(wid, start, end, **kwargs):
    defaults = dict(
        window_id=wid,
        start_time=start,
        end_time=end,
        speech_rate_wpm=135,
        pitch_mean=180,
        pitch_std=25,
        volume_consistency=0.75,
        rms_mean=0.05,
        rms_std=0.01,
        pause_count=2,
        pause_frequency_per_min=12,
        mean_pause_duration=0.35,
        filler_count=0,
        filler_rate_per_100=0,
        phonation_ratio=0.72,
        asr_confidence=0.88,
        word_recognition_rate=0.95,
        word_count=22,
        transcript="sample text",
        reliable=True,
    )
    defaults.update(kwargs)
    return WindowMetrics(**defaults)


def _minimal_result(**overrides) -> SpeechAnalysisResult:
    """Create a minimal SpeechAnalysisResult with good defaults."""
    wins = [_make_window(i, i * 10, (i + 1) * 10) for i in range(6)]
    r = SpeechAnalysisResult(
        audio_file="test.wav",
        duration=60.0,
        sample_rate=16000,
        transcription=TranscriptionResult(full_text="hello world", duration=60.0),
        window_metrics=wins,
        global_acoustic={
            "avg_speech_rate_wpm": 135.0,
            "speech_rate_std": 10.0,
            "global_pitch_mean": 180.0,
            "global_pitch_std": 30.0,
            "global_volume_consistency": 0.75,
        },
        global_fluency={
            "total_pause_count": 12,
            "global_mean_pause_duration": 0.35,
            "total_filler_count": 3,
            "global_filler_rate_per_100": 2.5,
            "avg_phonation_ratio": 0.72,
        },
        global_clarity={
            "global_asr_confidence": 0.88,
            "global_word_recognition_rate": 0.95,
        },
        language_metrics=LanguageMetrics(
            grammar_score=0.90,
            vocabulary_richness=0.52,
            sentence_complexity=1.8,
        ),
        temporal_metrics=TemporalMetrics(
            confidence_trajectory=ConfidenceTrajectory(
                slope=-0.5, direction="increasing", p_value=0.04,
                per_window_pitch_std=[28, 26, 25, 24, 23, 22],
            ),
            warmup_index=WarmupIndex(warmup_seconds=15, warmup_window=2),
            fatigue_detection=FatigueDetection(fatigue_score=15.0, significant=False),
            engagement_arc=EngagementArc(score=72.0, shape="ideal"),
        ),
        adaptive_score=AdaptiveScoreResult(
            baseline=SpeakerBaseline(),
            overall_adaptive_score=75.0,
            consistency_score=80.0,
        ),
        cognitive_strain=CognitiveStrainResult(
            mean_csi=22.0, max_csi=45.0, struggle_count=0,
        ),
        coherence=CoherenceResult(
            composite_coherence=68.0,
        ),
        total_windows=6,
    )
    for k, v in overrides.items():
        setattr(r, k, v)
    return r


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def predictor():
    return ListenerPredictor()


@pytest.fixture
def good_result():
    return _minimal_result()


# ════════════════════════════════════════════════════════════════
# TEST SUITE
# ════════════════════════════════════════════════════════════════


class TestListenerPredictionSchema:
    """Schema defaults and custom values."""

    def test_defaults(self):
        lp = ListenerPrediction()
        assert lp.comprehension == 50.0
        assert lp.overall_listener_score == 50.0

    def test_custom(self):
        lp = ListenerPrediction(comprehension=85.0, engagement=90.0)
        assert lp.comprehension == 85.0


class TestComprehension:
    """Predicted comprehension from speech/language metrics."""

    def test_good_speaker_high_comprehension(self, predictor, good_result):
        lp = predictor.predict(good_result)
        assert lp.comprehension >= 70

    def test_fast_speaker_lower_comprehension(self, predictor):
        r = _minimal_result()
        r.global_acoustic["avg_speech_rate_wpm"] = 210.0
        lp = predictor.predict(r)
        comp_fast = lp.comprehension

        r2 = _minimal_result()
        r2.global_acoustic["avg_speech_rate_wpm"] = 135.0
        lp2 = predictor.predict(r2)
        comp_normal = lp2.comprehension

        assert comp_fast < comp_normal

    def test_low_asr_lowers_comprehension(self, predictor):
        r = _minimal_result()
        r.global_clarity["global_asr_confidence"] = 0.50
        lp = predictor.predict(r)
        assert lp.comprehension < 80

    def test_no_language_metrics(self, predictor):
        r = _minimal_result(language_metrics=None)
        lp = predictor.predict(r)
        assert 0 <= lp.comprehension <= 100


class TestEngagement:
    """Predicted listener engagement."""

    def test_good_engagement(self, predictor, good_result):
        lp = predictor.predict(good_result)
        assert lp.engagement >= 50

    def test_monotone_lowers_engagement(self, predictor):
        r = _minimal_result()
        r.global_acoustic["global_pitch_std"] = 5.0
        lp = predictor.predict(r)
        eng_mono = lp.engagement

        r2 = _minimal_result()
        r2.global_acoustic["global_pitch_std"] = 40.0
        lp2 = predictor.predict(r2)
        eng_varied = lp2.engagement

        assert eng_mono < eng_varied

    def test_high_fillers_lower_engagement(self, predictor):
        r = _minimal_result()
        r.global_fluency["global_filler_rate_per_100"] = 15.0
        lp = predictor.predict(r)
        assert lp.engagement < 80


class TestTrust:
    """Predicted trust/credibility."""

    def test_good_trust(self, predictor, good_result):
        lp = predictor.predict(good_result)
        assert lp.trust >= 50

    def test_high_strain_lowers_trust(self, predictor):
        r = _minimal_result(
            cognitive_strain=CognitiveStrainResult(mean_csi=70.0, max_csi=90.0)
        )
        lp = predictor.predict(r)
        trust_strained = lp.trust

        r2 = _minimal_result(
            cognitive_strain=CognitiveStrainResult(mean_csi=10.0, max_csi=20.0)
        )
        lp2 = predictor.predict(r2)
        trust_relaxed = lp2.trust

        assert trust_strained < trust_relaxed

    def test_no_coherence_uses_default(self, predictor):
        r = _minimal_result(coherence=None)
        lp = predictor.predict(r)
        assert 0 <= lp.trust <= 100


class TestRetention:
    """Retention = f(comprehension, engagement), multiplicative."""

    def test_retention_is_geometric_mean(self, predictor):
        # Directly test the static method
        ret = ListenerPredictor._retention(80.0, 80.0)
        expected = (80.0 * 80.0) ** 0.5  # = 80.0
        assert abs(ret - expected) < 0.1

    def test_low_comprehension_tanks_retention(self, predictor):
        ret_low = ListenerPredictor._retention(30.0, 90.0)
        ret_high = ListenerPredictor._retention(90.0, 90.0)
        assert ret_low < ret_high

    def test_zero_engagement_zero_retention(self, predictor):
        ret = ListenerPredictor._retention(90.0, 0.0)
        assert ret == 0.0

    def test_capped_at_100(self, predictor):
        ret = ListenerPredictor._retention(100.0, 100.0)
        assert ret <= 100.0


class TestAttentionSustainability:
    """How long listener stays engaged."""

    def test_good_attention(self, predictor, good_result):
        lp = predictor.predict(good_result)
        assert lp.attention_sustainability >= 50

    def test_fatigue_lowers_attention(self, predictor):
        r = _minimal_result()
        r.temporal_metrics.fatigue_detection = FatigueDetection(
            fatigue_score=80.0, significant=True,
        )
        lp = predictor.predict(r)
        attn_fatigued = lp.attention_sustainability

        r2 = _minimal_result()
        r2.temporal_metrics.fatigue_detection = FatigueDetection(
            fatigue_score=5.0, significant=False,
        )
        lp2 = predictor.predict(r2)
        attn_fresh = lp2.attention_sustainability

        assert attn_fatigued < attn_fresh

    def test_declining_arc_lowers_attention(self, predictor):
        r = _minimal_result()
        r.temporal_metrics.engagement_arc = EngagementArc(shape="declining", score=30)
        lp = predictor.predict(r)
        attn_declining = lp.attention_sustainability

        r2 = _minimal_result()
        r2.temporal_metrics.engagement_arc = EngagementArc(shape="ideal", score=90)
        lp2 = predictor.predict(r2)
        attn_ideal = lp2.attention_sustainability

        assert attn_declining < attn_ideal

    def test_no_temporal_metrics(self, predictor):
        r = _minimal_result(temporal_metrics=None)
        lp = predictor.predict(r)
        assert lp.attention_sustainability == 60.0


class TestOverallScore:
    """Overall listener score — weighted combination."""

    def test_overall_in_range(self, predictor, good_result):
        lp = predictor.predict(good_result)
        assert 0 <= lp.overall_listener_score <= 100

    def test_overall_is_weighted_average(self, predictor, good_result):
        lp = predictor.predict(good_result)
        expected = (
            lp.comprehension * 0.25
            + lp.engagement * 0.25
            + lp.trust * 0.20
            + lp.retention * 0.15
            + lp.attention_sustainability * 0.15
        )
        assert abs(lp.overall_listener_score - round(expected, 1)) <= 0.2

    def test_details_populated(self, predictor, good_result):
        lp = predictor.predict(good_result)
        assert "comprehension_inputs" in lp.details
        assert "engagement_inputs" in lp.details
        assert "trust_inputs" in lp.details

    def test_all_scores_positive(self, predictor, good_result):
        lp = predictor.predict(good_result)
        assert lp.comprehension > 0
        assert lp.engagement > 0
        assert lp.trust > 0
        assert lp.retention > 0
        assert lp.attention_sustainability > 0


class TestEdgeCases:
    """Missing modules, empty data."""

    def test_minimal_result_no_optional_modules(self, predictor):
        """No temporal, adaptive, cognitive, coherence → still works."""
        r = SpeechAnalysisResult(
            global_acoustic={"avg_speech_rate_wpm": 130, "global_pitch_std": 25},
            global_fluency={"global_filler_rate_per_100": 2, "avg_phonation_ratio": 0.7},
            global_clarity={"global_asr_confidence": 0.85},
        )
        lp = predictor.predict(r)
        assert 0 <= lp.overall_listener_score <= 100

    def test_empty_result(self, predictor):
        r = SpeechAnalysisResult()
        lp = predictor.predict(r)
        assert isinstance(lp, ListenerPrediction)
        assert 0 <= lp.overall_listener_score <= 100
