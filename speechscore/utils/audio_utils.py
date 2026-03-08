"""
SpeechScore 2.0 — Audio Utilities

Handles audio loading, resampling, and temporal windowing.
All audio is converted to mono float32 at 16 kHz (Whisper standard).
"""

from __future__ import annotations

from dataclasses import dataclass

import warnings

import librosa
import numpy as np

from speechscore.config.settings import SpeechScoreConfig

# Suppress librosa's audioread deprecation warning — we have ffmpeg installed
# so soundfile will handle most formats; the fallback is harmless.
warnings.filterwarnings("ignore", message=".*audioread.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*", category=UserWarning)


@dataclass
class AudioSegment:
    """A contiguous slice of audio with metadata."""
    audio: np.ndarray       # float32, mono, config.audio.sample_rate
    sample_rate: int
    start_time: float       # seconds (relative to full recording)
    end_time: float         # seconds
    window_id: int


# ────────────────────────────────────────────────────────────────────
# Loading
# ────────────────────────────────────────────────────────────────────

def load_audio(
    file_path: str,
    config: SpeechScoreConfig,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file, convert to mono float32, resample to target SR.

    Supports every format understood by ``soundfile`` / ``ffmpeg``
    (WAV, MP3, M4A, FLAC, OGG, …).

    Returns
    -------
    audio : np.ndarray   — shape ``(n_samples,)``, dtype ``float32``
    sr    : int          — always ``config.audio.sample_rate`` (16 000)
    """
    audio, sr = librosa.load(
        file_path,
        sr=config.audio.sample_rate,
        mono=True,
    )
    return audio.astype(np.float32), sr


def get_duration(audio: np.ndarray, sample_rate: int) -> float:
    """Audio duration in seconds."""
    return len(audio) / sample_rate


# ────────────────────────────────────────────────────────────────────
# Windowing
# ────────────────────────────────────────────────────────────────────

def create_windows(
    audio: np.ndarray,
    sample_rate: int,
    config: SpeechScoreConfig,
) -> list[AudioSegment]:
    """
    Segment audio into overlapping temporal windows.

    Default: 10 s window, 5 s hop → 50 % overlap.
    Windows shorter than ``config.audio.min_speech_duration`` are dropped.

    Parameters
    ----------
    audio       : full recording array
    sample_rate : sampling rate (should already be 16 kHz)
    config      : global configuration

    Returns
    -------
    List of ``AudioSegment`` instances in chronological order.
    """
    window_samples = int(config.audio.window_duration * sample_rate)
    hop_samples = int(config.audio.hop_duration * sample_rate)
    total_samples = len(audio)

    windows: list[AudioSegment] = []
    window_id = 0
    start = 0

    while start < total_samples:
        end = min(start + window_samples, total_samples)
        segment = audio[start:end]

        # drop very short trailing windows
        if len(segment) / sample_rate < config.audio.min_speech_duration:
            break

        windows.append(AudioSegment(
            audio=segment,
            sample_rate=sample_rate,
            start_time=start / sample_rate,
            end_time=end / sample_rate,
            window_id=window_id,
        ))
        window_id += 1
        start += hop_samples

    return windows


# ────────────────────────────────────────────────────────────────────
# Feature-level helpers (used by multiple analyzers)
# ────────────────────────────────────────────────────────────────────

def compute_rms_energy(
    audio: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """RMS energy contour — shape ``(n_frames,)``."""
    return librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]


def compute_zero_crossing_rate(audio: np.ndarray) -> np.ndarray:
    """Zero-crossing rate per frame — shape ``(n_frames,)``."""
    return librosa.feature.zero_crossing_rate(y=audio)[0]
