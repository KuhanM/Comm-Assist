"""
SpeechScore 2.0 — Whisper Transcription Module

Provides word-level transcription with timestamps and per-word confidence
scores.  Optimised for Apple Silicon via MPS; falls back to CPU gracefully.

Outputs:
  - Full text transcript
  - Word-level timing + probability  (used by Clarity, Fluency, Acoustic)
  - Segment-level log-probs          (used by Clarity)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from speechscore.config.settings import SpeechScoreConfig
from speechscore.models.schemas import (
    TranscriptionResult,
    WordInfo,
    SegmentInfo,
)

if TYPE_CHECKING:
    import whisper as _whisper  # type hints only

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Whisper-based ASR with word-level timestamps."""

    def __init__(self, config: SpeechScoreConfig) -> None:
        self.config = config
        self._model: _whisper.Whisper | None = None

    # ── model management ─────────────────────────────────────────

    def load_model(self) -> None:
        """
        Load Whisper model onto the best available device.

        On an M4 Mac Pro the model is placed on MPS (Metal Performance
        Shaders) for hardware-accelerated inference.  ``large-v3`` needs
        ~3 GB VRAM / RAM which is well within the 48 GB budget.
        """
        import whisper  # deferred import — heavy

        device = self.config.detect_device()
        logger.info(
            "Loading Whisper '%s' on device '%s' …",
            self.config.whisper.model_name,
            device,
        )

        # MPS does not support float64 which Whisper's DTW word-timestamp
        # alignment requires.  Force CPU when word_timestamps are enabled.
        if device == "mps" and self.config.whisper.word_timestamps:
            logger.info(
                "MPS doesn't support float64 (needed for word timestamps) "
                "— loading model on CPU instead"
            )
            device = "cpu"
            self.config.whisper.device = "cpu"

        try:
            self._model = whisper.load_model(
                self.config.whisper.model_name,
                device=device,
            )
        except RuntimeError:
            logger.warning("Device '%s' failed – falling back to CPU", device)
            self.config.whisper.device = "cpu"
            self._model = whisper.load_model(
                self.config.whisper.model_name,
                device="cpu",
            )

        logger.info("Whisper model loaded successfully")

    # ── transcription ────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Transcribe a full audio array and return structured results.

        Parameters
        ----------
        audio : np.ndarray, float32, mono, 16 kHz

        Returns
        -------
        TranscriptionResult with word-level timing and probabilities.
        """
        if self._model is None:
            self.load_model()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        use_fp16 = self.config.whisper.device not in ("cpu",)

        result = self._model.transcribe(
            audio,
            language=self.config.whisper.language,
            word_timestamps=self.config.whisper.word_timestamps,
            fp16=use_fp16,
            verbose=False,
            temperature=self.config.whisper.temperature,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )

        # ── hallucination filter thresholds ──
        logprob_thresh = self.config.whisper.halluc_logprob
        min_wp = self.config.whisper.min_word_prob

        # ── parse segments (with multi-layer hallucination filter) ──
        segments: list[SegmentInfo] = []
        kept_seg_ids: set[int] = set()
        raw_segs = result.get("segments", [])
        seen_texts: set[str] = set()

        for seg in raw_segs:
            avg_lp = seg.get("avg_logprob", 0.0)
            comp = seg.get("compression_ratio", 0.0)
            text = seg.get("text", "").strip()
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)

            # Hallucination detection (multi-condition):
            # 1. Very low confidence (logprob < -1.0)
            is_low_logprob = avg_lp < logprob_thresh
            # 2. Zero-duration segment (collapsed timestamps)
            is_zero_dur = (seg_end - seg_start) < 0.05
            # 3. Repeated text (exact duplicate of a previous segment)
            text_norm = text.lower().strip()
            is_repeat = text_norm in seen_texts and len(text_norm) > 5
            # 4. Empty text
            is_empty = len(text_norm) == 0

            is_halluc = is_low_logprob or is_zero_dur or is_repeat or is_empty

            if text_norm:
                seen_texts.add(text_norm)

            if is_halluc:
                reasons = []
                if is_low_logprob:
                    reasons.append(f"logprob={avg_lp:.2f}")
                if is_zero_dur:
                    reasons.append("zero-duration")
                if is_repeat:
                    reasons.append("repeated-text")
                if is_empty:
                    reasons.append("empty")
                logger.warning(
                    "Dropping hallucinated segment %d [%.1f-%.1fs] (%s): '%s'",
                    seg["id"], seg_start, seg_end,
                    ", ".join(reasons), text[:60],
                )
                continue
            kept_seg_ids.add(seg["id"])
            segments.append(SegmentInfo(
                id=seg["id"],
                text=text,
                start=seg_start,
                end=seg_end,
                avg_logprob=avg_lp,
                no_speech_prob=seg.get("no_speech_prob", 0.0),
                compression_ratio=comp,
            ))

        n_dropped = len(raw_segs) - len(segments)
        if n_dropped:
            logger.info(
                "Hallucination filter: dropped %d/%d segments",
                n_dropped, len(raw_segs),
            )

        # ── parse words (only from kept segments, dedup collapsed) ──
        words: list[WordInfo] = []
        for seg in raw_segs:
            if seg["id"] not in kept_seg_ids:
                continue
            prev_ts = None
            for w in seg.get("words", []):
                wp = w.get("probability", 0.0)
                ws = w["start"]
                we = w["end"]
                # skip near-zero probability words
                if wp < min_wp:
                    continue
                # skip collapsed-timestamp duplicates (Whisper artefact)
                ts = (round(ws, 3), round(we, 3))
                if ts == prev_ts and (we - ws) < 0.01:
                    continue
                prev_ts = ts
                words.append(WordInfo(
                    word=w["word"].strip(),
                    start=ws,
                    end=we,
                    probability=wp,
                ))

        dups_dropped = sum(
            len(s.get("words", []))
            for s in raw_segs if s["id"] in kept_seg_ids
        ) - len(words)
        if dups_dropped > 0:
            logger.info(
                "Word filter: dropped %d low-prob/collapsed words",
                dups_dropped,
            )

        # ── duration ──
        duration = 0.0
        if words:
            duration = words[-1].end
        elif segments:
            duration = segments[-1].end

        # ── rebuild full_text from kept segments only ──
        full_text = " ".join(s.text for s in segments).strip()

        return TranscriptionResult(
            full_text=full_text,
            words=words,
            segments=segments,
            language=result.get("language", "en"),
            duration=duration,
        )

    # ── window helpers ───────────────────────────────────────────

    def get_words_in_timerange(
        self,
        transcription: TranscriptionResult,
        start_time: float,
        end_time: float,
    ) -> list[WordInfo]:
        """Return words whose midpoint falls inside [start, end]."""
        return [
            w for w in transcription.words
            if w.start >= start_time and w.end <= end_time
        ]

    def get_transcript_for_window(
        self,
        transcription: TranscriptionResult,
        start_time: float,
        end_time: float,
    ) -> str:
        """Concatenated text for a temporal window."""
        words = self.get_words_in_timerange(
            transcription, start_time, end_time
        )
        return " ".join(w.word for w in words)
