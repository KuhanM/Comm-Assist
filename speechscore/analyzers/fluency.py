"""
SpeechScore 2.0 — Fluency Analyzer

Extracts **Fluency** metrics per temporal window:
  1. Pause Frequency          — pauses per minute
  2. Mean Pause Duration      — average gap length in seconds
  3. Filler Word Rate         — fillers per 100 words
  4. Phonation Time Ratio     — fraction of window that contains speech

Pause detection uses adaptive energy-based VAD (no external model/Java).
Filler detection matches the Whisper word list against a curated set.

Raw pause timestamp arrays are preserved for Phase 2 struggle-point
detection and cognitive load estimation.
"""

from __future__ import annotations

import logging

import librosa
import numpy as np

from speechscore.config.settings import SpeechScoreConfig
from speechscore.utils.audio_utils import AudioSegment

logger = logging.getLogger(__name__)


class FluencyAnalyzer:
    """Fluency feature extractor (per-window)."""

    def __init__(self, config: SpeechScoreConfig) -> None:
        self.config = config
        # pre-build lookup sets for O(1) matching
        self._single_fillers: set[str] = set()
        self._bigram_fillers: set[str] = set()
        for phrase in config.filler.filler_words:
            parts = phrase.lower().split()
            if len(parts) == 1:
                self._single_fillers.add(parts[0])
            else:
                self._bigram_fillers.add(phrase.lower())

    # ── public API ───────────────────────────────────────────────

    def analyze_window(
        self,
        segment: AudioSegment,
        words_in_window: list,
    ) -> dict:
        """
        Compute fluency features for one temporal window.

        Parameters
        ----------
        segment          : audio slice
        words_in_window  : word dicts/objects from Whisper

        Returns
        -------
        dict with pause, filler, and phonation metrics + raw lists
        """
        window_dur = segment.end_time - segment.start_time

        # 1. Pause detection (energy-based)
        pause = self._detect_pauses(segment)

        # 2. Filler detection (lexical)
        filler = self._detect_fillers(words_in_window)

        # 3. Phonation ratio
        phonation = self._phonation_ratio(segment)

        # 4. Derived rates
        window_min = window_dur / 60.0
        word_count = len(words_in_window)
        pause_freq = pause["pause_count"] / window_min if window_min > 0 else 0.0
        filler_rate = (
            filler["filler_count"] / (word_count / 100.0)
            if word_count > 0
            else 0.0
        )

        return {
            "pause_count": pause["pause_count"],
            "pause_frequency_per_min": round(pause_freq, 2),
            "mean_pause_duration": pause["mean_pause_duration"],
            "max_pause_duration": pause["max_pause_duration"],
            "total_pause_duration": pause["total_pause_duration"],
            "pause_durations": pause["pause_durations"],
            "filler_count": filler["filler_count"],
            "filler_rate_per_100": round(filler_rate, 2),
            "filler_words_found": filler["filler_words_found"],
            "phonation_ratio": phonation,
            "word_count": word_count,
        }

    def compute_global(self, window_results: list[dict]) -> dict:
        """Aggregate fluency metrics across all windows."""
        if not window_results:
            return {}

        total_pauses = sum(wr["pause_count"] for wr in window_results)
        total_fillers = sum(wr["filler_count"] for wr in window_results)
        total_words = sum(wr["word_count"] for wr in window_results)

        all_dur: list[float] = []
        for wr in window_results:
            all_dur.extend(wr.get("pause_durations", []))

        phon = [wr["phonation_ratio"] for wr in window_results]

        return {
            "total_pause_count": total_pauses,
            "global_mean_pause_duration": (
                float(np.mean(all_dur)) if all_dur else 0.0
            ),
            "total_filler_count": total_fillers,
            "global_filler_rate_per_100": round(
                total_fillers / (total_words / 100.0) if total_words > 0 else 0.0,
                2,
            ),
            "avg_phonation_ratio": float(np.mean(phon)) if phon else 0.0,
            "total_word_count": total_words,
        }

    # ── pause detection ──────────────────────────────────────────

    def _detect_pauses(self, segment: AudioSegment) -> dict:
        """
        Energy-based pause detection with adaptive thresholding.

        A pause is a contiguous run of frames whose RMS energy falls
        below ``max(energy_threshold, adaptive_factor × mean_rms)``.
        Only gaps ≥ ``min_pause_duration`` are counted.
        """
        frame_length = 1024
        hop_length = 256

        rms = librosa.feature.rms(
            y=segment.audio,
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]

        frame_times = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=segment.sample_rate,
            hop_length=hop_length,
        )

        # adaptive threshold
        threshold = max(
            self.config.vad.energy_threshold,
            np.mean(rms) * self.config.vad.adaptive_factor,
        )

        is_silent = rms < threshold

        # group consecutive silent frames into pauses
        pauses: list[float] = []
        in_pause = False
        pause_start = 0.0

        for i, silent in enumerate(is_silent):
            t = frame_times[i] if i < len(frame_times) else frame_times[-1]
            if silent and not in_pause:
                in_pause = True
                pause_start = t
            elif not silent and in_pause:
                in_pause = False
                dur = t - pause_start
                if dur >= self.config.vad.min_pause_duration:
                    pauses.append(dur)

        # trailing pause
        if in_pause:
            end_t = len(segment.audio) / segment.sample_rate
            dur = end_t - pause_start
            if dur >= self.config.vad.min_pause_duration:
                pauses.append(dur)

        return {
            "pause_count": len(pauses),
            "mean_pause_duration": round(float(np.mean(pauses)), 3) if pauses else 0.0,
            "max_pause_duration": round(float(np.max(pauses)), 3) if pauses else 0.0,
            "total_pause_duration": round(float(np.sum(pauses)), 3) if pauses else 0.0,
            "pause_durations": [round(p, 3) for p in pauses],
        }

    # ── filler detection ─────────────────────────────────────────

    def _detect_fillers(self, words: list) -> dict:
        """
        Match transcribed words against the filler lexicon.

        Handles both single-word fillers (``um``, ``uh``) and two-word
        phrases (``you know``, ``sort of``).
        """
        fillers_found: list[dict] = []

        # normalise word list
        texts = []
        for w in words:
            txt = w.word if hasattr(w, "word") else w.get("word", "")
            texts.append(txt.lower().strip())

        # single-word
        for i, t in enumerate(texts):
            if t in self._single_fillers:
                w_obj = words[i]
                start = w_obj.start if hasattr(w_obj, "start") else w_obj.get("start", 0)
                end = w_obj.end if hasattr(w_obj, "end") else w_obj.get("end", 0)
                fillers_found.append(
                    {"word": t, "index": i, "start": start, "end": end}
                )

        # bigrams
        for i in range(len(texts) - 1):
            bigram = f"{texts[i]} {texts[i + 1]}"
            if bigram in self._bigram_fillers:
                w0, w1 = words[i], words[i + 1]
                start = w0.start if hasattr(w0, "start") else w0.get("start", 0)
                end = w1.end if hasattr(w1, "end") else w1.get("end", 0)
                fillers_found.append(
                    {"word": bigram, "index": i, "start": start, "end": end}
                )

        return {
            "filler_count": len(fillers_found),
            "filler_words_found": fillers_found,
        }

    # ── phonation ratio ──────────────────────────────────────────

    def _phonation_ratio(self, segment: AudioSegment) -> float:
        """
        Fraction of the window containing speech (not silence).

        phonation_ratio ∈ [0, 1].  Optimal range: 0.60 – 0.80.
        """
        rms = librosa.feature.rms(
            y=segment.audio, frame_length=1024, hop_length=256
        )[0]

        threshold = max(
            self.config.vad.energy_threshold,
            np.mean(rms) * self.config.vad.adaptive_factor,
        )

        speech_frames = int(np.sum(rms >= threshold))
        total_frames = len(rms)

        if total_frames == 0:
            return 0.0

        return round(float(speech_frames / total_frames), 4)
