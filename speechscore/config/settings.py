"""
SpeechScore 2.0 — Configuration Settings

Centralized configuration for all pipeline components.
Optimized defaults for Apple Silicon (M4 Mac Pro, 48GB RAM).
"""

from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    """Audio loading and windowing parameters."""
    sample_rate: int = 16000           # Whisper expects 16 kHz
    window_duration: float = 10.0      # seconds per analysis window
    hop_duration: float = 5.0          # seconds between window starts (50% overlap)
    min_speech_duration: float = 1.0   # skip windows shorter than this


@dataclass
class WhisperConfig:
    """Whisper ASR configuration.

    M4 Mac Pro with 48 GB RAM can comfortably run ``large-v3``.
    Device is auto-detected at runtime (MPS → CUDA → CPU).
    """
    model_name: str = "large-v3"
    device: str = "auto"                # resolved by detect_device()
    language: str = "en"
    word_timestamps: bool = True
    temperature: float = 0.0            # deterministic decoding
    # hallucination filter thresholds
    halluc_logprob: float = -1.0        # segments worse than this are dropped
    halluc_max_compression: float = 2.5 # high compression → repetitive text
    min_word_prob: float = 0.05         # words below this are dropped


@dataclass
class BaselineConfig:
    """Speaker baseline extraction settings."""
    baseline_duration: float = 30.0    # first N seconds for personal baseline


@dataclass
class VADConfig:
    """Energy-based Voice Activity Detection thresholds."""
    energy_threshold: float = 0.01     # absolute RMS floor
    min_pause_duration: float = 0.2    # minimum gap to count as a pause (s)
    min_speech_duration: float = 0.1   # minimum voiced segment (s)
    adaptive_factor: float = 0.15      # fraction-of-mean for adaptive threshold


@dataclass
class FillerConfig:
    """Filler / hedge word detection."""
    filler_words: list = field(default_factory=lambda: [
        # single-word fillers
        "um", "uh", "er", "ah", "hmm",
        "like", "basically", "actually", "literally",
        "right", "well", "okay", "ok",
        # two-word fillers (matched as bigrams)
        "you know", "sort of", "kind of", "i mean",
        "okay so", "so basically",
    ])


@dataclass
class LanguageConfig:
    """NLP / grammar tooling."""
    spacy_model: str = "en_core_web_lg"
    language_tool_lang: str = "en-US"


@dataclass
class ProsodyConfig:
    """Pitch extraction parameters (Praat / parselmouth)."""
    f0_min: float = 75.0               # Hz — floor for pitch tracking
    f0_max: float = 500.0              # Hz — ceiling for pitch tracking
    time_step: float = 0.01            # pitch analysis time step (s)


# ────────────────────────────────────────────────────────────────────
# Root config — instantiate once, share across all modules
# ────────────────────────────────────────────────────────────────────

@dataclass
class SpeechScoreConfig:
    """Master configuration for the entire SpeechScore pipeline."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    filler: FillerConfig = field(default_factory=FillerConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    prosody: ProsodyConfig = field(default_factory=ProsodyConfig)

    def detect_device(self) -> str:
        """Auto-detect best compute device for Apple Silicon / CUDA / CPU."""
        import torch

        if self.whisper.device != "auto":
            return self.whisper.device

        if torch.backends.mps.is_available():
            self.whisper.device = "mps"
        elif torch.cuda.is_available():
            self.whisper.device = "cuda"
        else:
            self.whisper.device = "cpu"

        return self.whisper.device
