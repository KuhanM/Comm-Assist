"""
Tests for the Multi-Modal Coherence Analyzer.

Tests cover:
  - Sentiment-Prosody alignment scoring
  - Emphasis-Importance alignment
  - Pause-Semantic synchronization
  - Composite coherence aggregation
  - Edge cases (empty data, few windows)
"""

import pytest
import numpy as np

from speechscore.models.schemas import (
    WindowMetrics,
    TranscriptionResult,
    WordInfo,
    CoherenceResult,
)
from speechscore.analyzers.coherence import CoherenceAnalyzer


# ── Helpers ─────────────────────────────────────────────────────

def _make_window(
    wid: int,
    start: float,
    end: float,
    rms_mean: float = 0.05,
    transcript: str = "hello world",
    reliable: bool = True,
    pitch_std: float = 25.0,
) -> WindowMetrics:
    return WindowMetrics(
        window_id=wid,
        start_time=start,
        end_time=end,
        speech_rate_wpm=130.0,
        pitch_mean=180.0,
        pitch_std=pitch_std,
        volume_consistency=0.7,
        rms_mean=rms_mean,
        rms_std=0.01,
        pause_count=2,
        pause_frequency_per_min=12.0,
        mean_pause_duration=0.3,
        filler_count=0,
        filler_rate_per_100=0.0,
        phonation_ratio=0.7,
        asr_confidence=0.9,
        word_recognition_rate=0.95,
        word_count=20,
        transcript=transcript,
        reliable=reliable,
    )


def _make_transcription(
    words: list[tuple[str, float, float, float]] | None = None,
    full_text: str = "",
) -> TranscriptionResult:
    """Create a TranscriptionResult from (word, start, end, prob) tuples."""
    word_infos = []
    if words:
        for w, s, e, p in words:
            word_infos.append(WordInfo(word=w, start=s, end=e, probability=p))
        if not full_text:
            full_text = " ".join(w for w, _, _, _ in words)
    return TranscriptionResult(
        full_text=full_text,
        words=word_infos,
        segments=[],
        duration=max(e for _, _, e, _ in words) if words else 0.0,
    )


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def analyzer():
    return CoherenceAnalyzer()


@pytest.fixture
def basic_windows():
    """6 windows with varying energy and sentiment."""
    transcripts = [
        "I am really excited about this amazing opportunity",
        "This is wonderful and I feel great about it",
        "The results were disappointing and frustrating",
        "We need to address several concerns here",
        "I am thrilled with the outcome of the project",
        "Overall the experience was very positive",
    ]
    # Higher energy for positive, lower for negative
    energies = [0.08, 0.09, 0.03, 0.04, 0.085, 0.07]
    return [
        _make_window(i, i * 10.0, (i + 1) * 10.0, rms_mean=e, transcript=t)
        for i, (t, e) in enumerate(zip(transcripts, energies))
    ]


@pytest.fixture
def basic_transcription():
    """Transcription with enough words for emphasis/pause analysis."""
    words = [
        # Sentence 1: "The university announced new research grants today."
        ("The", 0.0, 0.2, 0.9),
        ("university", 0.3, 0.8, 0.9),   # important: NER/noun_chunk, gap=0.1
        ("announced", 0.9, 1.3, 0.9),     # verb, gap=0.1
        ("new", 1.4, 1.6, 0.9),
        ("research", 1.7, 2.1, 0.9),      # important: noun_chunk
        ("grants", 2.2, 2.6, 0.9),        # important: noun_chunk
        ("today.", 2.7, 3.0, 0.9),
        # Pause > 0.25s (boundary)
        # Sentence 2: "Dr. Smith will present the findings at the conference."
        ("Dr.", 3.5, 3.7, 0.9),           # NER, gap=0.5 from "today."
        ("Smith", 3.8, 4.2, 0.9),         # NER
        ("will", 4.3, 4.5, 0.9),
        ("present", 4.6, 5.0, 0.9),       # verb
        ("the", 5.1, 5.2, 0.9),
        ("findings", 5.3, 5.8, 0.9),      # noun_chunk
        ("at", 5.9, 6.0, 0.9),
        ("the", 6.1, 6.2, 0.9),
        ("conference.", 6.3, 6.9, 0.9),   # NER/noun_chunk
    ]
    return _make_transcription(words)


# ════════════════════════════════════════════════════════════════
# TEST SUITE
# ════════════════════════════════════════════════════════════════


class TestCoherenceResultSchema:
    """Test CoherenceResult schema defaults."""

    def test_defaults(self):
        r = CoherenceResult()
        assert r.sentiment_prosody_score == 50.0
        assert r.emphasis_alignment_score == 50.0
        assert r.pause_semantic_score == 50.0
        assert r.composite_coherence == 50.0

    def test_custom_values(self):
        r = CoherenceResult(
            sentiment_prosody_score=80.0,
            emphasis_alignment_score=70.0,
            pause_semantic_score=90.0,
            composite_coherence=79.5,
        )
        assert r.composite_coherence == 79.5


class TestAnalyzerEdgeCases:
    """Edge cases: too few windows, empty data."""

    def test_fewer_than_3_windows(self, analyzer):
        """Should return defaults when < 3 windows."""
        windows = [_make_window(0, 0, 10), _make_window(1, 10, 20)]
        tx = _make_transcription([("hello", 0, 0.5, 0.9)])
        result = analyzer.analyze(windows, tx)
        assert isinstance(result, CoherenceResult)
        assert result.composite_coherence == 50.0

    def test_zero_windows(self, analyzer):
        result = analyzer.analyze([], TranscriptionResult())
        assert result.composite_coherence == 50.0

    def test_empty_transcription(self, analyzer):
        windows = [_make_window(i, i * 10, (i + 1) * 10) for i in range(5)]
        result = analyzer.analyze(windows, TranscriptionResult())
        assert isinstance(result, CoherenceResult)


class TestSentimentProsody:
    """Tests for sentiment-prosody alignment."""

    def test_aligned_sentiment_energy(self, analyzer, basic_windows, basic_transcription):
        """Positive sentiment + high energy → good alignment."""
        result = analyzer.analyze(basic_windows, basic_transcription)
        assert result.sentiment_prosody_score >= 50.0

    def test_returns_50_with_few_windows(self, analyzer):
        """With < 4 reliable windows, should return neutral 50."""
        windows = [_make_window(i, i * 10, (i + 1) * 10) for i in range(3)]
        # mark two unreliable → only 1 reliable → < 4
        windows[0].reliable = False
        windows[1].reliable = False
        tx = _make_transcription([("hi", 0, 0.5, 0.9)])
        score = analyzer._sentiment_prosody(windows, tx)
        assert score == 50.0

    def test_score_range(self, analyzer, basic_windows, basic_transcription):
        """Score must be in [0, 100]."""
        s = analyzer._sentiment_prosody(basic_windows, basic_transcription)
        assert 0 <= s <= 100

    def test_monotone_energy_gives_lower_correlation(self, analyzer):
        """All same energy → correlation ~0 → score ~50."""
        windows = [
            _make_window(i, i * 10, (i + 1) * 10, rms_mean=0.05,
                         transcript=t)
            for i, t in enumerate([
                "I am excited", "how terrible", "amazing work",
                "what a disaster", "this is great",
            ])
        ]
        tx = _make_transcription([("word", 0, 0.5, 0.9)])
        score = analyzer._sentiment_prosody(windows, tx)
        # With uniform energy, correlation should be near 0
        assert 45 <= score <= 55


class TestEmphasisImportance:
    """Tests for emphasis-importance alignment."""

    def test_basic_score_range(self, analyzer, basic_windows, basic_transcription):
        s = analyzer._emphasis_importance(basic_windows, basic_transcription)
        assert 0 <= s <= 100

    def test_empty_words(self, analyzer):
        tx = TranscriptionResult(full_text="")
        windows = [_make_window(0, 0, 10)]
        s = analyzer._emphasis_importance(windows, tx)
        assert s == 50.0  # no words → default

    def test_no_important_tokens(self, analyzer):
        """If spaCy finds no entities/nouns, return 60 (neutral)."""
        tx = _make_transcription([
            ("oh", 0, 0.2, 0.9),
            ("um", 0.3, 0.5, 0.9),
            ("and", 0.6, 0.8, 0.9),
            ("uh", 0.9, 1.1, 0.9),
        ])
        windows = [_make_window(0, 0, 10)]
        s = analyzer._emphasis_importance(windows, tx)
        # Should return either 50 or 60 (no important words)
        assert s >= 50


class TestPauseSemantic:
    """Tests for pause-semantic synchronization."""

    def test_basic_score_range(self, analyzer, basic_windows, basic_transcription):
        s = analyzer._pause_semantic(basic_windows, basic_transcription)
        assert 0 <= s <= 100

    def test_no_pauses_gives_neutral(self, analyzer):
        """Continuous speech with no gaps → 70 (neutral-good)."""
        words = [
            ("hello", 0.0, 0.3, 0.9),
            ("world", 0.3, 0.6, 0.9),
            ("now", 0.6, 0.9, 0.9),
        ]
        tx = _make_transcription(words)
        windows = [_make_window(i, i * 10, (i + 1) * 10) for i in range(3)]
        s = analyzer._pause_semantic(windows, tx)
        assert s == 70.0

    def test_empty_text_gives_default(self, analyzer):
        tx = TranscriptionResult()
        windows = [_make_window(0, 0, 10)]
        s = analyzer._pause_semantic(windows, tx)
        assert s == 50.0

    def test_boundary_aligned_pauses(self, analyzer, basic_transcription):
        """Pauses at sentence boundaries should score well."""
        windows = [_make_window(i, i * 10, (i + 1) * 10) for i in range(3)]
        score = analyzer._pause_semantic(windows, basic_transcription)
        # The gap at 3.0→3.5 (0.5s) is right at the sentence boundary
        assert score >= 50


class TestCompositeCoherence:
    """Tests for the composite coherence score."""

    def test_composite_is_weighted_average(self, analyzer, basic_windows, basic_transcription):
        result = analyzer.analyze(basic_windows, basic_transcription)
        expected = (
            result.sentiment_prosody_score * 0.35
            + result.emphasis_alignment_score * 0.35
            + result.pause_semantic_score * 0.30
        )
        assert abs(result.composite_coherence - round(expected, 1)) <= 0.2

    def test_all_fields_populated(self, analyzer, basic_windows, basic_transcription):
        result = analyzer.analyze(basic_windows, basic_transcription)
        assert result.sentiment_prosody_score > 0
        assert result.emphasis_alignment_score > 0
        assert result.pause_semantic_score > 0
        assert result.composite_coherence > 0

    def test_result_type(self, analyzer, basic_windows, basic_transcription):
        result = analyzer.analyze(basic_windows, basic_transcription)
        assert isinstance(result, CoherenceResult)

    def test_scores_in_valid_range(self, analyzer, basic_windows, basic_transcription):
        result = analyzer.analyze(basic_windows, basic_transcription)
        for field in [
            result.sentiment_prosody_score,
            result.emphasis_alignment_score,
            result.pause_semantic_score,
            result.composite_coherence,
        ]:
            assert 0 <= field <= 100
