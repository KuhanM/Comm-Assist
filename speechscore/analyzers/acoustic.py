"""
SpeechScore 2.0 — Acoustic Analyzer

Extracts **Vocal Delivery** metrics per temporal window:
  1. Speech Rate (WPM)         — words per minute
  2. Pitch Variation (F0 SD)   — standard deviation of fundamental frequency
  3. Volume Consistency        — 1 − CV of RMS energy

Also stores raw pitch / RMS arrays for Phase 2 temporal analysis.

Tools: parselmouth (Praat bindings), librosa
"""

from __future__ import annotations

import logging

import librosa
import numpy as np
import parselmouth

from speechscore.config.settings import SpeechScoreConfig
from speechscore.utils.audio_utils import AudioSegment

logger = logging.getLogger(__name__)


class AcousticAnalyzer:
    """Vocal delivery feature extractor (per-window)."""

    def __init__(self, config: SpeechScoreConfig) -> None:
        self.config = config

    # ── public API ───────────────────────────────────────────────

    def analyze_window(
        self,
        segment: AudioSegment,
        words_in_window: list,
    ) -> dict:
        """
        Compute acoustic features for one temporal window.

        Parameters
        ----------
        segment          : audio slice with metadata
        words_in_window  : word dicts/objects with 'word', 'start', 'end'

        Returns
        -------
        dict with keys: speech_rate_wpm, pitch_mean, pitch_std,
                        pitch_range, voiced_fraction,
                        volume_consistency, rms_mean, rms_std,
                        pitch_values (list[float]),
                        rms_values   (list[float])
        """
        results: dict = {}

        # 1. Speech Rate
        window_duration = segment.end_time - segment.start_time
        results["speech_rate_wpm"] = self._speech_rate(
            words_in_window, window_duration
        )

        # 2. Pitch (F0) features
        results.update(self._pitch_features(segment))

        # 3. Volume features
        results.update(self._volume_features(segment))

        return results

    def compute_global(self, window_results: list[dict]) -> dict:
        """Aggregate acoustic metrics across all windows."""
        if not window_results:
            return {}

        wpm_vals = [w["speech_rate_wpm"] for w in window_results]

        all_pitch: list[float] = []
        all_rms: list[float] = []
        for wr in window_results:
            all_pitch.extend(wr.get("pitch_values", []))
            all_rms.extend(wr.get("rms_values", []))

        result = {
            "avg_speech_rate_wpm": float(np.mean(wpm_vals)),
            "speech_rate_std": float(np.std(wpm_vals)),
        }

        if all_pitch:
            result["global_pitch_mean"] = float(np.mean(all_pitch))
            result["global_pitch_std"] = float(np.std(all_pitch))
            result["global_pitch_range"] = float(
                np.max(all_pitch) - np.min(all_pitch)
            )

        if all_rms:
            m, s = float(np.mean(all_rms)), float(np.std(all_rms))
            result["global_volume_consistency"] = max(
                0.0, min(1.0, 1.0 - s / (m + 1e-8))
            )

        return result

    # ── private helpers ──────────────────────────────────────────

    @staticmethod
    def _speech_rate(words: list, window_duration: float) -> float:
        """Words per minute."""
        n = len(words)
        minutes = window_duration / 60.0
        return n / minutes if minutes > 0 else 0.0

    def _pitch_features(self, segment: AudioSegment) -> dict:
        """Extract F0 statistics via Praat (parselmouth)."""
        try:
            snd = parselmouth.Sound(
                segment.audio,
                sampling_frequency=segment.sample_rate,
            )
            pitch_obj = snd.to_pitch(
                time_step=self.config.prosody.time_step,
                pitch_floor=self.config.prosody.f0_min,
                pitch_ceiling=self.config.prosody.f0_max,
            )

            f0_all = pitch_obj.selected_array["frequency"]
            f0_voiced = f0_all[f0_all > 0]  # only voiced frames

            if len(f0_voiced) < 2:
                return self._empty_pitch()

            return {
                "pitch_mean": float(np.mean(f0_voiced)),
                "pitch_std": float(np.std(f0_voiced)),
                "pitch_range": float(np.ptp(f0_voiced)),
                "voiced_fraction": float(len(f0_voiced) / len(f0_all)),
                "pitch_values": f0_voiced.tolist(),
            }

        except Exception as exc:
            logger.warning("Pitch extraction failed: %s", exc)
            return self._empty_pitch()

    @staticmethod
    def _empty_pitch() -> dict:
        return {
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "pitch_range": 0.0,
            "voiced_fraction": 0.0,
            "pitch_values": [],
        }

    @staticmethod
    def _volume_features(segment: AudioSegment) -> dict:
        """RMS energy statistics."""
        rms = librosa.feature.rms(
            y=segment.audio, frame_length=2048, hop_length=512
        )[0]

        if len(rms) == 0 or np.mean(rms) < 1e-10:
            return {
                "volume_consistency": 0.0,
                "rms_mean": 0.0,
                "rms_std": 0.0,
                "rms_values": [],
            }

        m = float(np.mean(rms))
        s = float(np.std(rms))
        consistency = max(0.0, min(1.0, 1.0 - s / (m + 1e-8)))

        return {
            "volume_consistency": consistency,
            "rms_mean": m,
            "rms_std": s,
            "rms_values": rms.tolist(),
        }
