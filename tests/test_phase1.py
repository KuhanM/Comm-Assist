"""
SpeechScore 2.0 — Phase 1 Unit Tests

Validates all base metric extractors using synthetic audio so tests
run quickly without loading Whisper or requiring real recordings.

Run:
    cd Comm-Assist && python -m pytest tests/test_phase1.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from speechscore.config.settings import SpeechScoreConfig
from speechscore.utils.audio_utils import (
    AudioSegment,
    create_windows,
    get_duration,
)
from speechscore.models.schemas import (
    WindowMetrics,
    LanguageMetrics,
    TranscriptionResult,
    WordInfo,
)
from speechscore.analyzers.acoustic import AcousticAnalyzer
from speechscore.analyzers.fluency import FluencyAnalyzer
from speechscore.analyzers.clarity import ClarityAnalyzer


# ────────────────────────────────────────────────────────────────────
# Audio generators
# ────────────────────────────────────────────────────────────────────

def _sine(freq: float = 440, dur: float = 5.0, sr: int = 16000,
          amp: float = 0.5) -> np.ndarray:
    """Pure sine wave — float32, mono."""
    t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
    return amp * np.sin(2 * np.pi * freq * t)


def _silence(dur: float = 1.0, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(sr * dur), dtype=np.float32)


def _speech_like(dur: float = 30.0, sr: int = 16000) -> np.ndarray:
    """Alternating voiced chunks + short pauses."""
    audio = np.zeros(int(sr * dur), dtype=np.float32)
    seg_len = int(2.0 * sr)
    pause_len = int(0.5 * sr)
    pos = 0
    while pos < len(audio):
        end = min(pos + seg_len, len(audio))
        freq = np.random.uniform(100, 300)
        t = np.arange(end - pos) / sr
        audio[pos:end] = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        pos = end + pause_len
    return audio


def _make_segment(audio: np.ndarray, sr: int = 16000,
                  start: float = 0.0) -> AudioSegment:
    return AudioSegment(
        audio=audio,
        sample_rate=sr,
        start_time=start,
        end_time=start + len(audio) / sr,
        window_id=0,
    )


def _word_dicts(n: int, start: float = 0.0, gap: float = 0.3,
                prob: float = 0.9) -> list[dict]:
    """Generate *n* fake word dicts."""
    return [
        {
            "word": f"word{i}",
            "start": start + i * gap,
            "end": start + i * gap + 0.2,
            "probability": prob,
        }
        for i in range(n)
    ]


# ────────────────────────────────────────────────────────────────────
# Audio Utilities
# ────────────────────────────────────────────────────────────────────

class TestAudioUtils:

    def test_duration_1s(self):
        assert get_duration(np.zeros(16000), 16000) == 1.0

    def test_duration_30s(self):
        assert get_duration(np.zeros(16000 * 30), 16000) == 30.0

    def test_create_windows_count(self):
        cfg = SpeechScoreConfig()
        cfg.audio.window_duration = 5.0
        cfg.audio.hop_duration = 2.5
        audio = np.zeros(16000 * 20, dtype=np.float32)
        wins = create_windows(audio, 16000, cfg)
        assert len(wins) > 0
        assert wins[0].window_id == 0
        assert wins[0].start_time == 0.0
        assert wins[0].end_time == 5.0

    def test_create_windows_overlap(self):
        cfg = SpeechScoreConfig()
        cfg.audio.window_duration = 10.0
        cfg.audio.hop_duration = 5.0
        audio = np.zeros(16000 * 30, dtype=np.float32)
        wins = create_windows(audio, 16000, cfg)
        assert len(wins) >= 5
        assert wins[1].start_time == 5.0

    def test_window_audio_length(self):
        cfg = SpeechScoreConfig()
        cfg.audio.window_duration = 5.0
        cfg.audio.hop_duration = 5.0
        audio = np.random.randn(16000 * 15).astype(np.float32)
        wins = create_windows(audio, 16000, cfg)
        assert len(wins[0].audio) == 16000 * 5

    def test_short_trailing_window_dropped(self):
        cfg = SpeechScoreConfig()
        cfg.audio.window_duration = 10.0
        cfg.audio.hop_duration = 10.0
        cfg.audio.min_speech_duration = 2.0
        # 11 seconds → 1 full window + 1 s left (<2 s) → dropped
        audio = np.zeros(16000 * 11, dtype=np.float32)
        wins = create_windows(audio, 16000, cfg)
        assert len(wins) == 1


# ────────────────────────────────────────────────────────────────────
# Acoustic Analyzer
# ────────────────────────────────────────────────────────────────────

class TestAcousticAnalyzer:

    def setup_method(self):
        self.cfg = SpeechScoreConfig()
        self.ana = AcousticAnalyzer(self.cfg)

    def test_speech_rate(self):
        seg = _make_segment(_sine(dur=10.0))
        # 30 words in 10 s → 180 WPM
        words = _word_dicts(30)
        r = self.ana.analyze_window(seg, words)
        assert r["speech_rate_wpm"] == pytest.approx(180.0, abs=0.1)

    def test_pitch_detected(self):
        seg = _make_segment(_sine(freq=200, dur=5.0))
        r = self.ana.analyze_window(seg, [])
        assert r["pitch_mean"] > 0
        assert r["pitch_std"] >= 0

    def test_volume_consistency_pure_tone(self):
        seg = _make_segment(_sine(dur=5.0))
        r = self.ana.analyze_window(seg, [])
        assert r["volume_consistency"] > 0.5

    def test_silent_audio_zeros(self):
        seg = _make_segment(_silence(dur=1.0))
        r = self.ana.analyze_window(seg, [])
        assert r["pitch_mean"] == 0.0
        assert r["speech_rate_wpm"] == 0.0

    def test_global_aggregation(self):
        seg = _make_segment(_sine(freq=200, dur=5.0))
        r1 = self.ana.analyze_window(seg, _word_dicts(10))
        r2 = self.ana.analyze_window(seg, _word_dicts(15))
        g = self.ana.compute_global([r1, r2])
        assert "avg_speech_rate_wpm" in g
        assert g["avg_speech_rate_wpm"] > 0

    def test_global_empty(self):
        assert self.ana.compute_global([]) == {}


# ────────────────────────────────────────────────────────────────────
# Fluency Analyzer
# ────────────────────────────────────────────────────────────────────

class TestFluencyAnalyzer:

    def setup_method(self):
        self.cfg = SpeechScoreConfig()
        self.ana = FluencyAnalyzer(self.cfg)

    def test_filler_detection(self):
        words = [
            {"word": "I",     "start": 0.0, "end": 0.1, "probability": 0.9},
            {"word": "um",    "start": 0.2, "end": 0.4, "probability": 0.8},
            {"word": "think", "start": 0.5, "end": 0.7, "probability": 0.9},
            {"word": "like",  "start": 0.8, "end": 1.0, "probability": 0.9},
            {"word": "uh",    "start": 1.1, "end": 1.3, "probability": 0.7},
        ]
        seg = _make_segment(_sine(dur=2.0))
        r = self.ana.analyze_window(seg, words)
        assert r["filler_count"] >= 3  # um, like, uh

    def test_bigram_filler(self):
        words = [
            {"word": "you",  "start": 0.0, "end": 0.1, "probability": 0.9},
            {"word": "know", "start": 0.2, "end": 0.3, "probability": 0.9},
            {"word": "that", "start": 0.4, "end": 0.5, "probability": 0.9},
        ]
        seg = _make_segment(_sine(dur=1.0))
        r = self.ana.analyze_window(seg, words)
        # "you know" should be detected as a bigram filler
        found = [f["word"] for f in r["filler_words_found"]]
        assert "you know" in found

    def test_pause_detection(self):
        sr = 16000
        audio = np.concatenate([
            _sine(freq=200, dur=2.0, sr=sr, amp=0.5),
            _silence(dur=1.0, sr=sr),
            _sine(freq=250, dur=2.0, sr=sr, amp=0.5),
        ])
        seg = _make_segment(audio, sr=sr)
        r = self.ana.analyze_window(seg, [])
        assert r["pause_count"] >= 1

    def test_phonation_ratio_continuous(self):
        seg = _make_segment(_sine(dur=5.0, amp=0.5))
        r = self.ana.analyze_window(seg, [])
        assert r["phonation_ratio"] > 0.5

    def test_no_words(self):
        seg = _make_segment(_sine(dur=2.0))
        r = self.ana.analyze_window(seg, [])
        assert r["filler_count"] == 0
        assert r["word_count"] == 0

    def test_global_aggregation(self):
        seg = _make_segment(_sine(dur=3.0))
        r1 = self.ana.analyze_window(seg, _word_dicts(5))
        r2 = self.ana.analyze_window(seg, _word_dicts(8))
        g = self.ana.compute_global([r1, r2])
        assert g["total_word_count"] == 13


# ────────────────────────────────────────────────────────────────────
# Clarity Analyzer
# ────────────────────────────────────────────────────────────────────

class TestClarityAnalyzer:

    def setup_method(self):
        self.cfg = SpeechScoreConfig()
        self.ana = ClarityAnalyzer(self.cfg)

    def test_high_confidence(self):
        words = _word_dicts(3, prob=0.95)
        r = self.ana.analyze_window(words)
        assert r["asr_confidence"] > 0.9
        assert r["word_recognition_rate"] == 1.0

    def test_low_confidence(self):
        words = _word_dicts(3, prob=0.2)
        r = self.ana.analyze_window(words)
        assert r["asr_confidence"] < 0.5
        assert r["word_recognition_rate"] == 0.0
        assert len(r["low_confidence_words"]) == 3

    def test_empty_words(self):
        r = self.ana.analyze_window([])
        assert r["asr_confidence"] == 0.0
        assert r["word_recognition_rate"] == 0.0

    def test_mixed(self):
        words = [
            {"word": "clear",  "start": 0, "end": 0.1, "probability": 0.9},
            {"word": "mumble", "start": 0.2, "end": 0.3, "probability": 0.3},
            {"word": "ok",     "start": 0.4, "end": 0.5, "probability": 0.8},
        ]
        r = self.ana.analyze_window(words)
        assert 0.0 < r["word_recognition_rate"] < 1.0
        assert len(r["low_confidence_words"]) == 1

    def test_global_aggregation(self):
        r1 = self.ana.analyze_window(_word_dicts(3, prob=0.9))
        r2 = self.ana.analyze_window(_word_dicts(3, prob=0.4))
        g = self.ana.compute_global([r1, r2])
        assert 0.4 < g["global_asr_confidence"] < 0.9


# ────────────────────────────────────────────────────────────────────
# Schema validation
# ────────────────────────────────────────────────────────────────────

class TestSchemas:

    def test_window_metrics_defaults(self):
        wm = WindowMetrics(window_id=0, start_time=0.0, end_time=10.0)
        assert wm.speech_rate_wpm is None
        assert wm.filler_count == 0

    def test_window_metrics_populated(self):
        wm = WindowMetrics(
            window_id=1,
            start_time=5.0,
            end_time=15.0,
            speech_rate_wpm=145.0,
            pitch_std=32.5,
            volume_consistency=0.82,
        )
        assert wm.speech_rate_wpm == 145.0

    def test_language_metrics_defaults(self):
        lm = LanguageMetrics()
        assert lm.grammar_score is None
        assert lm.total_word_count == 0

    def test_transcription_result(self):
        tr = TranscriptionResult(
            full_text="hello world",
            words=[WordInfo(word="hello", start=0.0, end=0.5, probability=0.9)],
        )
        assert len(tr.words) == 1
        assert tr.words[0].word == "hello"


# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
