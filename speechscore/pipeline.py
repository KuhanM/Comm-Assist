"""
SpeechScore 2.0 — Phase 1 Pipeline

End-to-end orchestrator:
  Audio file → Whisper transcription → Temporal windows
             → Per-window metrics (Acoustic + Fluency + Clarity)
             → Full-transcript Language analysis
             → Global aggregation → SpeechAnalysisResult

Design note
-----------
Every per-window metric is stored *individually* so that Phase 2 modules
(temporal.py, baseline.py, cognitive.py) can consume the time-series
directly without re-computing anything.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from speechscore.config.settings import SpeechScoreConfig
from speechscore.models.schemas import (
    SpeechAnalysisResult,
    WindowMetrics,
)
from speechscore.utils.audio_utils import load_audio, get_duration, create_windows
from speechscore.analyzers.transcription import WhisperTranscriber
from speechscore.analyzers.acoustic import AcousticAnalyzer
from speechscore.analyzers.fluency import FluencyAnalyzer
from speechscore.analyzers.clarity import ClarityAnalyzer
from speechscore.analyzers.language import LanguageAnalyzer
from speechscore.analyzers.temporal import TemporalAnalyzer
from speechscore.analyzers.baseline import BaselineExtractor
from speechscore.analyzers.adaptive_scorer import AdaptiveScorer
from speechscore.analyzers.cognitive import CognitiveAnalyzer
from speechscore.analyzers.coherence import CoherenceAnalyzer
from speechscore.analyzers.listener import ListenerPredictor
from speechscore.analyzers.scoring import compute_composite

logger = logging.getLogger(__name__)


class SpeechScorePipeline:
    """
    Phase 1 pipeline — base metric extraction.

    Usage::

        pipeline = SpeechScorePipeline()
        pipeline.initialize()          # loads Whisper (heavy, once)
        result = pipeline.analyze("speech.wav")
        print(result.model_dump_json(indent=2))

    The returned ``SpeechAnalysisResult`` contains:
      * ``transcription``   — full text + word-level timestamps / probabilities
      * ``window_metrics``  — list[WindowMetrics] with 12 per-window base metrics
      * ``global_acoustic`` — aggregated vocal delivery stats
      * ``global_fluency``  — aggregated fluency stats
      * ``global_clarity``  — aggregated clarity stats
      * ``language_metrics``— grammar, vocabulary, complexity (full-transcript)
    """

    def __init__(self, config: SpeechScoreConfig | None = None) -> None:
        self.config = config or SpeechScoreConfig()

        self.transcriber = WhisperTranscriber(self.config)
        self.acoustic = AcousticAnalyzer(self.config)
        self.fluency = FluencyAnalyzer(self.config)
        self.clarity = ClarityAnalyzer(self.config)
        self.language = LanguageAnalyzer(self.config)
        self.temporal = TemporalAnalyzer()
        self.baseline_extractor = BaselineExtractor(self.config)
        self.adaptive_scorer = AdaptiveScorer()
        self.cognitive = CognitiveAnalyzer()
        self.coherence = CoherenceAnalyzer()
        self.listener = ListenerPredictor()

        self._initialised = False

    # ── lifecycle ────────────────────────────────────────────────

    def initialize(self) -> None:
        """Pre-load heavy models (Whisper).  Call once before first analyze()."""
        logger.info("Initialising SpeechScore pipeline …")
        t0 = time.time()

        device = self.config.detect_device()
        logger.info("Compute device: %s", device)

        self.transcriber.load_model()
        self._initialised = True
        logger.info("Pipeline ready (%.1f s)", time.time() - t0)

    def close(self) -> None:
        """Release external resources (LanguageTool server, etc.)."""
        self.language.close()

    # ── main entry point ─────────────────────────────────────────

    def analyze(self, audio_path: str) -> SpeechAnalysisResult:
        """
        Run the full Phase 1 analysis on an audio file.

        Parameters
        ----------
        audio_path : path to a WAV / MP3 / M4A / FLAC file.

        Returns
        -------
        SpeechAnalysisResult  (serialisable via ``.model_dump()``).
        """
        audio_path = str(Path(audio_path).resolve())
        logger.info("▶ Analysing: %s", audio_path)
        t_total = time.time()

        # ── Step 1: load audio ──
        self._step("1/12 Loading audio")
        audio, sr = load_audio(audio_path, self.config)
        duration = get_duration(audio, sr)
        logger.info("  %.1f s, %d Hz", duration, sr)

        # ── Step 2: transcribe ──
        self._step("2/12 Transcribing (Whisper)")
        t0 = time.time()
        transcription = self.transcriber.transcribe(audio)
        logger.info(
            "  %d words, %d segments (%.1f s)",
            len(transcription.words),
            len(transcription.segments),
            time.time() - t0,
        )

        # ── Step 3: create windows ──
        self._step("3/12 Creating temporal windows")
        windows = create_windows(audio, sr, self.config)
        logger.info(
            "  %d windows (%.0f s window, %.0f s hop)",
            len(windows),
            self.config.audio.window_duration,
            self.config.audio.hop_duration,
        )

        # ── Step 4: per-window analysis ──
        self._step("4/12 Per-window analysis")
        all_wm: list[WindowMetrics] = []
        acoustic_buf: list[dict] = []
        fluency_buf: list[dict] = []
        clarity_buf: list[dict] = []

        for win in windows:
            words = self.transcriber.get_words_in_timerange(
                transcription, win.start_time, win.end_time
            )
            text = self.transcriber.get_transcript_for_window(
                transcription, win.start_time, win.end_time
            )

            ac = self.acoustic.analyze_window(win, words)
            fl = self.fluency.analyze_window(win, words)
            cl = self.clarity.analyze_window(words)

            acoustic_buf.append(ac)
            fluency_buf.append(fl)
            clarity_buf.append(cl)

            wm = WindowMetrics(
                window_id=win.window_id,
                start_time=win.start_time,
                end_time=win.end_time,
                # vocal delivery
                speech_rate_wpm=ac["speech_rate_wpm"],
                pitch_mean=ac["pitch_mean"],
                pitch_std=ac["pitch_std"],
                volume_consistency=ac["volume_consistency"],
                pitch_range=ac.get("pitch_range"),
                voiced_fraction=ac.get("voiced_fraction"),
                rms_mean=ac.get("rms_mean"),
                rms_std=ac.get("rms_std"),
                # fluency
                pause_count=fl["pause_count"],
                pause_frequency_per_min=fl["pause_frequency_per_min"],
                mean_pause_duration=fl["mean_pause_duration"],
                max_pause_duration=fl.get("max_pause_duration"),
                filler_count=fl["filler_count"],
                filler_rate_per_100=fl["filler_rate_per_100"],
                phonation_ratio=fl["phonation_ratio"],
                # clarity
                asr_confidence=cl["asr_confidence"],
                word_recognition_rate=cl["word_recognition_rate"],
                # raw
                word_count=fl["word_count"],
                transcript=text,
            )

            # ── reliability flagging ──
            flags: list[str] = []
            if ac["speech_rate_wpm"] and ac["speech_rate_wpm"] > 240:
                flags.append("extreme_wpm")
            if cl["asr_confidence"] is not None and cl["asr_confidence"] < 0.50:
                flags.append("very_low_asr")
            if fl["word_count"] < 3:
                flags.append("too_few_words")
            if flags:
                wm.reliable = False
                wm.reliability_flags = flags

            all_wm.append(wm)

            logger.debug(
                "  win %02d  WPM=%3.0f  F0σ=%5.1f  fillers=%d  ASR=%.2f",
                win.window_id,
                ac["speech_rate_wpm"],
                ac["pitch_std"],
                fl["filler_count"],
                cl["asr_confidence"],
            )

        # ── Step 5: language analysis (full transcript) ──
        self._step("5/12 Language analysis")
        t0 = time.time()
        lang = self.language.analyze(transcription.full_text)
        logger.info("  done (%.1f s)", time.time() - t0)

        # ── Step 6: temporal dynamics analysis (Phase 2 ⭐ NOVEL) ──
        self._step("6/12 Temporal dynamics analysis")
        t0 = time.time()
        temporal = self.temporal.analyze(all_wm)
        logger.info("  done (%.1f s)", time.time() - t0)

        # ── Step 7: global aggregation ──
        self._step("7/12 Global aggregation")
        g_ac = self.acoustic.compute_global(acoustic_buf)
        g_fl = self.fluency.compute_global(fluency_buf)
        g_cl = self.clarity.compute_global(clarity_buf)

        # ── Step 8: speaker baseline extraction (Phase 2 ⭐ NOVEL) ──
        self._step("8/12 Speaker baseline extraction")
        t0 = time.time()
        baseline = self.baseline_extractor.extract(all_wm)
        logger.info(
            "  %d baseline windows (%.0f s), WPM=%.0f±%.1f",
            baseline.windows_used,
            baseline.baseline_duration,
            baseline.speech_rate_mean,
            baseline.speech_rate_std,
        )

        # ── Step 9: adaptive scoring (Phase 2 ⭐ NOVEL) ──
        self._step("9/12 Speaker-adaptive scoring")
        adaptive = self.adaptive_scorer.score(all_wm, baseline)
        logger.info(
            "  composite=%.1f/100, consistency=%.1f/100, SR delta=%+.1f%%",
            adaptive.overall_adaptive_score,
            adaptive.consistency_score,
            adaptive.speech_rate_delta_pct,
        )

        # ── Step 10: cognitive strain index (Phase 2 ⭐ NOVEL) ──
        self._step("10/12 Cognitive strain analysis")
        t0 = time.time()
        cognitive = self.cognitive.analyze(all_wm, baseline)
        logger.info(
            "  mean CSI=%.1f, max=%.1f, struggle points=%d/%d (%.0f%%)",
            cognitive.mean_csi,
            cognitive.max_csi,
            cognitive.struggle_count,
            len(all_wm),
            cognitive.struggle_pct,
        )

        result = SpeechAnalysisResult(
            audio_file=audio_path,
            duration=duration,
            sample_rate=sr,
            transcription=transcription,
            window_metrics=all_wm,
            global_acoustic=g_ac,
            global_fluency=g_fl,
            global_clarity=g_cl,
            language_metrics=lang,
            temporal_metrics=temporal,
            adaptive_score=adaptive,
            cognitive_strain=cognitive,
            total_windows=len(windows),
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # ── Step 11: multi-modal coherence ⭐ NOVEL ──
        self._step("11/12 Multi-modal coherence analysis")
        t0 = time.time()
        coh = self.coherence.analyze(all_wm, transcription)
        result.coherence = coh
        logger.info(
            "  sentiment-prosody=%.1f, emphasis=%.1f, pause-semantic=%.1f → composite=%.1f (%.1f s)",
            coh.sentiment_prosody_score, coh.emphasis_alignment_score,
            coh.pause_semantic_score, coh.composite_coherence,
            time.time() - t0,
        )

        # ── Step 12: listener prediction ──
        self._step("12/12 Listener prediction")
        t0 = time.time()
        lp = self.listener.predict(result)
        result.listener_prediction = lp
        logger.info(
            "  comp=%.0f eng=%.0f trust=%.0f ret=%.0f attn=%.0f → overall=%.1f (%.1f s)",
            lp.comprehension, lp.engagement, lp.trust,
            lp.retention, lp.attention_sustainability,
            lp.overall_listener_score, time.time() - t0,
        )

        # ── Composite scoring (Day 7) ──
        result.composite = compute_composite(result)
        logger.info(
            "  ▸ Composite: %.1f/100 (%s)",
            result.composite.composite_score,
            result.composite.grade,
        )

        elapsed = time.time() - t_total
        logger.info("■ Analysis complete (%.1f s)", elapsed)
        self._log_summary(result)
        return result

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _step(msg: str) -> None:
        logger.info("── %s ──", msg)

    @staticmethod
    def _log_summary(r: SpeechAnalysisResult) -> None:
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info("Duration       : %.1f s", r.duration)
        logger.info("Windows        : %d", r.total_windows)
        logger.info("Words          : %d", len(r.transcription.words))

        ga = r.global_acoustic
        if ga:
            logger.info(
                "Avg WPM        : %.0f  (σ %.1f)",
                ga.get("avg_speech_rate_wpm", 0),
                ga.get("speech_rate_std", 0),
            )
            logger.info("Global F0 σ    : %.1f Hz", ga.get("global_pitch_std", 0))

        gf = r.global_fluency
        if gf:
            logger.info("Total pauses   : %d", gf.get("total_pause_count", 0))
            logger.info("Total fillers  : %d", gf.get("total_filler_count", 0))
            logger.info(
                "Phonation      : %.1f%%",
                gf.get("avg_phonation_ratio", 0) * 100,
            )

        gc = r.global_clarity
        if gc:
            logger.info("ASR confidence : %.2f", gc.get("global_asr_confidence", 0))

        lm = r.language_metrics
        if lm:
            logger.info("Grammar        : %s", lm.grammar_score)
            logger.info("Vocab TTR      : %s", lm.vocabulary_richness)
            logger.info("Sent. complex. : %s", lm.sentence_complexity)

        tm = r.temporal_metrics
        if tm:
            ct = tm.confidence_trajectory
            logger.info("Confidence     : %s (slope=%.3f, p=%.3f)",
                        ct.direction, ct.slope, ct.p_value)
            wi = tm.warmup_index
            logger.info("Warmup         : %.0f s (window %d)",
                        wi.warmup_seconds, wi.warmup_window)
            fd = tm.fatigue_detection
            logger.info("Fatigue        : %.1f/100 (significant=%s)",
                        fd.fatigue_score, fd.significant)
            ea = tm.engagement_arc
            logger.info("Engagement arc : %.1f/100 (%s)",
                        ea.score, ea.shape)

        ad = r.adaptive_score
        if ad:
            logger.info("Adaptive score : %.1f/100", ad.overall_adaptive_score)
            logger.info("Consistency    : %.1f/100", ad.consistency_score)
            logger.info("SR vs baseline : %+.1f%%", ad.speech_rate_delta_pct)
            logger.info("Pitch stab.    : %.2fx baseline", ad.pitch_stability_ratio)
            for am in ad.adaptive_metrics:
                logger.info(
                    "  %-30s z=%+5.2f  %s",
                    am.metric_name, am.z_score, am.deviation_label,
                )

        cs = r.cognitive_strain
        if cs:
            logger.info("Cogn. strain   : mean=%.1f, max=%.1f", cs.mean_csi, cs.max_csi)
            logger.info("Struggle pts   : %d/%d (%.0f%%)",
                        cs.struggle_count, len(r.window_metrics), cs.struggle_pct)
            for sp in cs.struggle_points:
                logger.info(
                    "  ⚠ win %d (%.0f–%.0fs) CSI=%.0f  cause=%s",
                    sp.window_id, sp.start_time, sp.end_time,
                    sp.csi_score, sp.primary_cause,
                )

        cp = r.composite
        if cp:
            logger.info("COMPOSITE      : %.1f/100 (%s)", cp.composite_score, cp.grade)
            for cs_item in cp.category_scores:
                logger.info("  %-22s %5.1f", cs_item.category, cs_item.score)

        coh = r.coherence
        if coh:
            logger.info("Coherence      : %.1f/100 (SP=%.1f, EI=%.1f, PS=%.1f)",
                        coh.composite_coherence,
                        coh.sentiment_prosody_score,
                        coh.emphasis_alignment_score,
                        coh.pause_semantic_score)

        lp = r.listener_prediction
        if lp:
            logger.info("Listener pred. : %.1f/100 (comp=%.0f eng=%.0f trust=%.0f ret=%.0f attn=%.0f)",
                        lp.overall_listener_score,
                        lp.comprehension, lp.engagement, lp.trust,
                        lp.retention, lp.attention_sustainability)

        logger.info("=" * 60)
