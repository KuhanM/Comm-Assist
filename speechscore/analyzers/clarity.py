"""
SpeechScore 2.0 — Clarity Analyzer

Extracts **Clarity** metrics per temporal window:
  1. ASR Confidence          — mean Whisper word-level probability
  2. Word Recognition Rate   — fraction of words above confidence threshold

These are derived entirely from Whisper's output (no extra model needed).
Low confidence regions correlate with mumbled / unclear speech and are
flagged for Phase 2 struggle-point detection.
"""

from __future__ import annotations

import logging

import numpy as np

from speechscore.config.settings import SpeechScoreConfig

logger = logging.getLogger(__name__)

# words below this probability are considered poorly recognised
_RECOGNITION_THRESHOLD = 0.5


class ClarityAnalyzer:
    """Speech clarity feature extractor (per-window)."""

    def __init__(self, config: SpeechScoreConfig) -> None:
        self.config = config

    # ── public API ───────────────────────────────────────────────

    def analyze_window(self, words_in_window: list) -> dict:
        """
        Compute clarity metrics for one temporal window.

        Parameters
        ----------
        words_in_window : word dicts / WordInfo objects with a
                          ``probability`` field from Whisper.

        Returns
        -------
        dict with asr_confidence, word_recognition_rate,
        low_confidence_words, confidence_values.
        """
        if not words_in_window:
            return {
                "asr_confidence": 0.0,
                "word_recognition_rate": 0.0,
                "low_confidence_words": [],
                "confidence_values": [],
            }

        confidences: list[float] = []
        for w in words_in_window:
            prob = (
                w.probability if hasattr(w, "probability")
                else w.get("probability", 0.0)
            )
            confidences.append(float(prob))

        asr_conf = float(np.mean(confidences))

        recognised = sum(1 for c in confidences if c >= _RECOGNITION_THRESHOLD)
        recog_rate = recognised / len(confidences)

        # flag low-confidence words (potential clarity issues)
        low_conf: list[dict] = []
        for w in words_in_window:
            prob = (
                w.probability if hasattr(w, "probability")
                else w.get("probability", 0.0)
            )
            if prob < _RECOGNITION_THRESHOLD:
                word_txt = w.word if hasattr(w, "word") else w.get("word", "")
                start = w.start if hasattr(w, "start") else w.get("start", 0)
                end = w.end if hasattr(w, "end") else w.get("end", 0)
                low_conf.append({
                    "word": word_txt,
                    "confidence": round(prob, 3),
                    "start": start,
                    "end": end,
                })

        return {
            "asr_confidence": round(asr_conf, 4),
            "word_recognition_rate": round(recog_rate, 4),
            "low_confidence_words": low_conf,
            "confidence_values": [round(c, 4) for c in confidences],
        }

    def compute_global(self, window_results: list[dict]) -> dict:
        """Aggregate clarity metrics across all windows."""
        if not window_results:
            return {}

        all_conf: list[float] = []
        all_low: list[dict] = []
        for wr in window_results:
            all_conf.extend(wr.get("confidence_values", []))
            all_low.extend(wr.get("low_confidence_words", []))

        n = len(all_conf) or 1
        return {
            "global_asr_confidence": round(
                float(np.mean(all_conf)) if all_conf else 0.0, 4
            ),
            "global_word_recognition_rate": round(
                sum(1 for c in all_conf if c >= _RECOGNITION_THRESHOLD) / n,
                4,
            ),
            "total_low_confidence_words": len(all_low),
        }
