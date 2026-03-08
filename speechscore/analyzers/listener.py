"""
SpeechScore 2.0 — Listener Prediction Module

Predicts the *listener experience* from measured speech metrics.
All predictions are derived — no new feature extraction is needed.

Five listener-experience dimensions:

  1. **Comprehension** — How easily the listener can follow.
     f(speech_rate, vocabulary, sentence_complexity, grammar, clarity)

  2. **Engagement** — How captivated the listener stays.
     f(pitch_variation, energy_dynamics, engagement_arc, filler_rate)

  3. **Trust / Credibility** — Perceived speaker confidence.
     f(adaptive_consistency, cognitive_strain, fluency, coherence)

  4. **Retention** — How much the listener remembers afterwards.
     = comprehension × engagement (multiplicative interaction)

  5. **Attention Sustainability** — How long the listener stays engaged.
     f(fatigue_score, engagement_arc_shape, temporal_consistency)

Each dimension is scored 0–100 via calibrated piecewise maps.
"""

from __future__ import annotations

import logging

import numpy as np

from speechscore.models.schemas import (
    SpeechAnalysisResult,
    ListenerPrediction,
)

logger = logging.getLogger(__name__)


def _piecewise(value: float, bp: list[tuple[float, float]]) -> float:
    """Piecewise-linear interpolation (same helper as scoring.py)."""
    if value <= bp[0][0]:
        return float(bp[0][1])
    if value >= bp[-1][0]:
        return float(bp[-1][1])
    for i in range(len(bp) - 1):
        v0, s0 = bp[i]
        v1, s1 = bp[i + 1]
        if v0 <= value <= v1:
            t = (value - v0) / (v1 - v0) if v1 > v0 else 0.0
            return s0 + t * (s1 - s0)
    return 50.0


class ListenerPredictor:
    """
    Predict listener experience from SpeechAnalysisResult.

    Usage::

        predictor = ListenerPredictor()
        prediction = predictor.predict(result)
    """

    def predict(self, result: SpeechAnalysisResult) -> ListenerPrediction:
        """
        Compute all 5 listener dimensions + overall.

        Parameters
        ----------
        result : fully-populated SpeechAnalysisResult.

        Returns
        -------
        ListenerPrediction with per-dimension scores.
        """
        comp = self._comprehension(result)
        eng = self._engagement(result)
        trust = self._trust(result)
        ret = self._retention(comp, eng)
        attn = self._attention_sustainability(result)

        overall = (
            comp * 0.25
            + eng * 0.25
            + trust * 0.20
            + ret * 0.15
            + attn * 0.15
        )

        return ListenerPrediction(
            comprehension=round(comp, 1),
            engagement=round(eng, 1),
            trust=round(trust, 1),
            retention=round(ret, 1),
            attention_sustainability=round(attn, 1),
            overall_listener_score=round(overall, 1),
            details={
                "comprehension_inputs": self._last_comp_details,
                "engagement_inputs": self._last_eng_details,
                "trust_inputs": self._last_trust_details,
            },
        )

    # ────────────────────────────────────────────────────────
    # 1.  COMPREHENSION
    # ────────────────────────────────────────────────────────

    _last_comp_details: dict = {}

    def _comprehension(self, r: SpeechAnalysisResult) -> float:
        """
        Predicted ease of understanding.

        Inputs:
          - Speech rate (too fast → hard to follow)
          - Vocabulary richness (moderate is best)
          - Sentence complexity (too nested → hard)
          - Grammar score (errors confuse)
          - ASR confidence proxy for articulation clarity
        """
        ga = r.global_acoustic or {}
        gc = r.global_clarity or {}
        lm = r.language_metrics

        wpm = ga.get("avg_speech_rate_wpm", 135.0)
        asr = gc.get("global_asr_confidence", 0.85)

        s_rate = _piecewise(wpm, [
            (60, 40), (90, 70), (120, 95), (140, 100),
            (160, 85), (190, 60), (230, 30),
        ])
        s_asr = _piecewise(asr, [
            (0.4, 20), (0.6, 40), (0.75, 70), (0.85, 85),
            (0.9, 95), (0.95, 100),
        ])

        if lm:
            ttr = lm.vocabulary_richness if lm.vocabulary_richness is not None else 0.5
            s_vocab = _piecewise(ttr, [
                (0.1, 30), (0.3, 60), (0.45, 90), (0.55, 100),
                (0.65, 85), (0.8, 60),
            ])
            cx = lm.sentence_complexity if lm.sentence_complexity is not None else 1.5
            s_complex = _piecewise(cx, [
                (0.5, 40), (1.0, 70), (1.5, 95), (2.0, 100),
                (2.5, 80), (3.5, 50),
            ])
            gs = (lm.grammar_score if lm.grammar_score is not None else 0.85) * 100
            s_grammar = min(100, gs)
        else:
            s_vocab = 70.0
            s_complex = 70.0
            s_grammar = 70.0

        score = (
            s_rate * 0.30
            + s_asr * 0.25
            + s_vocab * 0.15
            + s_complex * 0.15
            + s_grammar * 0.15
        )

        self._last_comp_details = {
            "speech_rate": round(s_rate, 1),
            "clarity": round(s_asr, 1),
            "vocabulary": round(s_vocab, 1),
            "complexity": round(s_complex, 1),
            "grammar": round(s_grammar, 1),
        }

        return score

    # ────────────────────────────────────────────────────────
    # 2.  ENGAGEMENT
    # ────────────────────────────────────────────────────────

    _last_eng_details: dict = {}

    def _engagement(self, r: SpeechAnalysisResult) -> float:
        """
        Predicted listener attention/captivation.

        Inputs:
          - Pitch variation (monotone → disengaging)
          - Energy dynamics (flat energy → boring)
          - Engagement arc score (from temporal)
          - Filler rate (excessive fillers → disengaging)
        """
        ga = r.global_acoustic or {}
        gf = r.global_fluency or {}

        f0_std = ga.get("global_pitch_std", 30.0)
        filler = gf.get("global_filler_rate_per_100", 2.0)

        s_pitch = _piecewise(f0_std, [
            (5, 15), (10, 35), (20, 70), (30, 90),
            (45, 100), (70, 85), (100, 60),
        ])
        s_filler = _piecewise(filler, [
            (0, 100), (1, 100), (3, 80), (5, 60),
            (10, 30), (20, 10),
        ])

        # Energy dynamics: std of per-window RMS
        wm = r.window_metrics
        if wm:
            rms_vals = [w.rms_mean for w in wm if w.rms_mean is not None]
            if rms_vals and np.std(rms_vals) > 1e-8:
                # Coefficient of variation — more variation = more engaging
                cv = float(np.std(rms_vals) / np.mean(rms_vals))
                s_energy = _piecewise(cv, [
                    (0, 20), (0.05, 40), (0.15, 75), (0.3, 100),
                    (0.5, 90), (0.8, 60),
                ])
            else:
                s_energy = 40.0
        else:
            s_energy = 50.0

        # Engagement arc from temporal
        tm = r.temporal_metrics
        if tm:
            s_arc = tm.engagement_arc.score
        else:
            s_arc = 50.0

        score = (
            s_pitch * 0.30
            + s_energy * 0.25
            + s_arc * 0.25
            + s_filler * 0.20
        )

        self._last_eng_details = {
            "pitch_variation": round(s_pitch, 1),
            "energy_dynamics": round(s_energy, 1),
            "engagement_arc": round(s_arc, 1),
            "filler_freedom": round(s_filler, 1),
        }

        return score

    # ────────────────────────────────────────────────────────
    # 3.  TRUST / CREDIBILITY
    # ────────────────────────────────────────────────────────

    _last_trust_details: dict = {}

    def _trust(self, r: SpeechAnalysisResult) -> float:
        """
        Perceived speaker credibility.

        Inputs:
          - Adaptive consistency (deviation from own baseline → uncertain)
          - Cognitive strain (high strain → looks unsure)
          - Fluency (smooth delivery → trustworthy)
          - Coherence (words match delivery → authentic)
        """
        # Adaptive consistency
        if r.adaptive_score:
            s_consist = r.adaptive_score.consistency_score
        else:
            s_consist = 60.0

        # Cognitive strain (lower = more trustworthy)
        if r.cognitive_strain:
            s_csi = max(0, 100 - r.cognitive_strain.mean_csi)
        else:
            s_csi = 70.0

        # Fluency — phonation ratio + filler rate
        gf = r.global_fluency or {}
        phonation = gf.get("avg_phonation_ratio", 0.7)
        filler = gf.get("global_filler_rate_per_100", 2.0)
        s_phon = _piecewise(phonation, [
            (0.3, 20), (0.5, 50), (0.65, 80), (0.75, 95), (0.85, 100),
        ])
        s_filler = _piecewise(filler, [
            (0, 100), (2, 90), (4, 70), (7, 45), (12, 20),
        ])
        s_fluency = (s_phon + s_filler) / 2.0

        # Coherence (if available)
        if r.coherence:
            s_coh = r.coherence.composite_coherence
        else:
            s_coh = 60.0

        score = (
            s_consist * 0.30
            + s_csi * 0.25
            + s_fluency * 0.25
            + s_coh * 0.20
        )

        self._last_trust_details = {
            "consistency": round(s_consist, 1),
            "low_strain": round(s_csi, 1),
            "fluency": round(s_fluency, 1),
            "coherence": round(s_coh, 1),
        }

        return score

    # ────────────────────────────────────────────────────────
    # 4.  RETENTION
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _retention(comprehension: float, engagement: float) -> float:
        """
        Predicted information retention.

        Multiplicative: if either comprehension or engagement is low,
        retention drops sharply.
        """
        # Geometric mean scaled to 0–100
        geo = (comprehension * engagement) ** 0.5
        return min(100.0, geo)

    # ────────────────────────────────────────────────────────
    # 5.  ATTENTION SUSTAINABILITY
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _attention_sustainability(r: SpeechAnalysisResult) -> float:
        """
        How long the listener stays engaged.

        Inputs:
          - Fatigue score (speaker fading → listener loses interest)
          - Engagement arc shape (declining → dropping attention)
          - Temporal consistency (erratic → attention wanders)
        """
        tm = r.temporal_metrics
        if not tm:
            return 60.0

        # Fatigue resistance (lower fatigue → better attention)
        s_fatigue = max(0, 100 - tm.fatigue_detection.fatigue_score)

        # Arc shape bonus
        shape = tm.engagement_arc.shape
        arc_map = {
            "ideal": 95,
            "rising": 85,
            "u-shaped": 70,
            "flat": 60,
            "declining": 35,
        }
        s_arc = arc_map.get(shape, 55)

        # Temporal consistency — lower pitch-std variance = more consistent
        pstds = tm.confidence_trajectory.per_window_pitch_std
        if len(pstds) >= 3:
            cv = float(np.std(pstds) / max(np.mean(pstds), 1e-8))
            s_consist = _piecewise(cv, [
                (0, 100), (0.2, 90), (0.5, 70), (0.8, 50), (1.2, 30),
            ])
        else:
            s_consist = 60.0

        score = (
            s_fatigue * 0.40
            + s_arc * 0.35
            + s_consist * 0.25
        )

        return score
