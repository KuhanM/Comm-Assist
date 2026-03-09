"""
SpeechScore 2.0 — Composite Scoring Engine  (Day 7)

Takes the full SpeechAnalysisResult and produces a weighted composite
score (0–100) with per-category sub-scores, letter grade, and summary.

Category weights (from spec §5.2, normalised over 9 implemented categories):
  Vocal Delivery     15.0%    (15/100)
  Fluency            15.0%    (15/100)
  Clarity            10.0%    (10/100)
  Language           10.0%    (10/100)
  Temporal Dynamics  15.0%    (15/100)
  Cognitive Load      5.0%    ( 5/100)
  Speaker-Adaptive   10.0%    (10/100)
  Coherence          10.0%    (10/100)  ⭐ NEW
  Listener Score     10.0%    (10/100)  ⭐ NEW

Total: 100%
"""

from __future__ import annotations

import logging

import numpy as np

from speechscore.models.schemas import (
    SpeechAnalysisResult,
    CategoryScore,
    CompositeScoreResult,
)

logger = logging.getLogger(__name__)


# ── Weights (spec §5.2: total = 100) ──────────────────────────

_W_TOTAL = 100.0

CATEGORY_WEIGHTS: dict[str, float] = {
    "vocal_delivery":    15.0 / _W_TOTAL,
    "fluency":           15.0 / _W_TOTAL,
    "clarity":           10.0 / _W_TOTAL,
    "language":          10.0 / _W_TOTAL,
    "temporal_dynamics": 15.0 / _W_TOTAL,
    "cognitive_load":     5.0 / _W_TOTAL,
    "speaker_adaptive":  10.0 / _W_TOTAL,
    "coherence":         10.0 / _W_TOTAL,
    "listener_score":    10.0 / _W_TOTAL,
}


# ── Piecewise-linear interpolation ──────────────────────────────

def _piecewise(value: float, bp: list[tuple[float, float]]) -> float:
    """
    Linearly interpolate a score from sorted (value, score) breakpoints.

    Values below the first breakpoint return its score; values above the
    last breakpoint return the last score.
    """
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


# ── Per-category scorers ─────────────────────────────────────────

def _score_vocal_delivery(r: SpeechAnalysisResult) -> tuple[float, dict]:
    """Vocal Delivery: speech rate, pitch variation, volume consistency."""
    ga = r.global_acoustic or {}
    wpm = ga.get("avg_speech_rate_wpm", 135.0)
    f0_std = ga.get("global_pitch_std", 30.0)
    vol_con = ga.get("global_volume_consistency", 0.7)

    s_wpm = _piecewise(wpm, [
        (60, 20), (90, 50), (120, 90), (135, 100),
        (150, 90), (180, 60), (220, 30), (260, 10),
    ])
    s_f0 = _piecewise(f0_std, [
        (5, 20), (15, 60), (20, 85), (30, 100),
        (50, 90), (70, 60), (100, 30),
    ])
    s_vol = _piecewise(vol_con, [
        (0, 0), (0.3, 30), (0.5, 60), (0.7, 85),
        (0.85, 100), (1.0, 100),
    ])

    score = (s_wpm + s_f0 + s_vol) / 3.0
    return score, {
        "speech_rate": round(s_wpm, 1),
        "pitch_variation": round(s_f0, 1),
        "volume_consistency": round(s_vol, 1),
    }


def _score_fluency(r: SpeechAnalysisResult) -> tuple[float, dict]:
    """Fluency: pause frequency, pause duration, filler rate, phonation."""
    gf = r.global_fluency or {}

    # Pause frequency — computed from window metrics for accuracy
    wm = r.window_metrics
    if wm:
        pf_vals = [w.pause_frequency_per_min or 0.0 for w in wm]
        pause_freq = float(np.mean(pf_vals))
    else:
        pause_freq = 18.0

    pause_dur = gf.get("global_mean_pause_duration", 0.4)
    filler_rate = gf.get("global_filler_rate_per_100", 2.0)
    phonation = gf.get("avg_phonation_ratio", 0.7)

    # Note: our pause detector counts ALL energy dips (micro-pauses
    # included), so "optimal <8/min" from the spec doesn't apply
    # directly — calibrated breakpoints for our detector:
    s_pf = _piecewise(pause_freq, [
        (0, 100), (10, 100), (20, 90), (30, 75),
        (40, 60), (50, 45), (60, 30), (80, 15),
    ])
    s_pd = _piecewise(pause_dur, [
        (0, 80), (0.25, 95), (0.5, 100), (0.75, 85),
        (1.0, 60), (2.0, 20),
    ])
    s_fr = _piecewise(filler_rate, [
        (0, 100), (1, 100), (3, 85), (5, 60), (10, 30), (20, 10),
    ])
    s_ph = _piecewise(phonation, [
        (0, 0), (0.3, 20), (0.5, 50), (0.6, 80),
        (0.7, 95), (0.8, 100), (0.9, 90), (1.0, 80),
    ])

    score = (s_pf + s_pd + s_fr + s_ph) / 4.0
    return score, {
        "pause_frequency": round(s_pf, 1),
        "pause_duration": round(s_pd, 1),
        "filler_rate": round(s_fr, 1),
        "phonation_ratio": round(s_ph, 1),
    }


def _score_clarity(r: SpeechAnalysisResult) -> tuple[float, dict]:
    """Clarity: ASR confidence, word recognition rate."""
    gc = r.global_clarity or {}
    asr = gc.get("global_asr_confidence", 0.85)
    wrr = gc.get("global_word_recognition_rate", 0.90)

    s_asr = _piecewise(asr, [
        (0, 0), (0.5, 20), (0.7, 50), (0.8, 75),
        (0.85, 85), (0.9, 95), (0.95, 100), (1.0, 100),
    ])
    s_wrr = _piecewise(wrr, [
        (0, 0), (0.7, 30), (0.85, 60), (0.9, 80),
        (0.95, 95), (1.0, 100),
    ])

    score = (s_asr + s_wrr) / 2.0
    return score, {
        "asr_confidence": round(s_asr, 1),
        "word_recognition": round(s_wrr, 1),
    }


def _score_language(r: SpeechAnalysisResult) -> tuple[float, dict]:
    """Language: grammar, vocabulary richness, sentence complexity."""
    lm = r.language_metrics
    if not lm:
        return 50.0, {}

    grammar = (lm.grammar_score if lm.grammar_score is not None else 0.85) * 100.0
    ttr = lm.vocabulary_richness if lm.vocabulary_richness is not None else 0.5
    complexity = lm.sentence_complexity if lm.sentence_complexity is not None else 2.0

    s_grammar = min(100, grammar)
    s_ttr = _piecewise(ttr, [
        (0, 10), (0.2, 30), (0.4, 70), (0.5, 90),
        (0.6, 100), (0.7, 90), (0.85, 60), (1.0, 40),
    ])
    s_complex = _piecewise(complexity, [
        (0.5, 30), (1.0, 60), (1.5, 85), (2.0, 100),
        (2.5, 90), (3.0, 70), (4.0, 40),
    ])

    score = (s_grammar + s_ttr + s_complex) / 3.0
    return score, {
        "grammar": round(s_grammar, 1),
        "vocabulary_richness": round(s_ttr, 1),
        "sentence_complexity": round(s_complex, 1),
    }


def _score_temporal(r: SpeechAnalysisResult) -> tuple[float, dict]:
    """Temporal Dynamics: confidence, warmup, fatigue, engagement arc."""
    tm = r.temporal_metrics
    if not tm:
        return 50.0, {}

    # 1. Confidence trajectory
    ct = tm.confidence_trajectory
    _conf_map = {
        ("increasing", True):  95,
        ("increasing", False): 75,
        ("stable", True):      70,
        ("stable", False):     70,
        ("decreasing", False): 55,
        ("decreasing", True):  30,
    }
    s_conf = _conf_map.get(
        (ct.direction, ct.p_value < 0.05), 60
    )

    # 2. Warmup — shorter relative to duration = better
    wi = tm.warmup_index
    if wi.change_point_detected and r.duration > 0:
        ratio = wi.warmup_seconds / r.duration
        s_warmup = max(0, min(100, 100 * (1 - ratio * 2)))
    else:
        s_warmup = 80.0  # no warmup needed = good

    # 3. Fatigue resistance (lower fatigue_score = better)
    fd = tm.fatigue_detection
    s_fatigue = max(0, 100 - fd.fatigue_score)

    # 4. Engagement arc (already scored 0–100)
    ea = tm.engagement_arc
    s_arc = ea.score

    score = (s_conf + s_warmup + s_fatigue + s_arc) / 4.0
    return score, {
        "confidence_trajectory": round(float(s_conf), 1),
        "warmup": round(float(s_warmup), 1),
        "fatigue_resistance": round(float(s_fatigue), 1),
        "engagement_arc": round(s_arc, 1),
    }


def _score_cognitive(r: SpeechAnalysisResult) -> tuple[float, dict]:
    """Cognitive Load: CSI (lower = better), struggle freedom."""
    cs = r.cognitive_strain
    if not cs:
        return 80.0, {}

    # Lower CSI = better
    s_csi = max(0.0, 100.0 - cs.mean_csi)

    # Fewer struggle points = better (25% struggle → 50 score)
    s_struggle = max(0.0, 100.0 - cs.struggle_pct * 2.0)

    score = s_csi * 0.6 + s_struggle * 0.4
    return score, {
        "strain_index": round(s_csi, 1),
        "struggle_freedom": round(s_struggle, 1),
    }


def _score_adaptive(r: SpeechAnalysisResult) -> tuple[float, dict]:
    """Speaker-Adaptive: adaptive score + consistency."""
    ad = r.adaptive_score
    if not ad:
        return 50.0, {}

    score = (ad.overall_adaptive_score + ad.consistency_score) / 2.0
    return score, {
        "adaptive_score": round(ad.overall_adaptive_score, 1),
        "consistency": round(ad.consistency_score, 1),
    }


def _score_coherence(r: SpeechAnalysisResult) -> tuple[float, dict]:
    """Multi-Modal Coherence: sentiment-prosody, emphasis, pause-semantic."""
    coh = r.coherence
    if not coh:
        return 50.0, {}

    return coh.composite_coherence, {
        "sentiment_prosody": round(coh.sentiment_prosody_score, 1),
        "emphasis_alignment": round(coh.emphasis_alignment_score, 1),
        "pause_semantic": round(coh.pause_semantic_score, 1),
    }


def _score_listener(r: SpeechAnalysisResult) -> tuple[float, dict]:
    """Listener Prediction: overall predicted listener experience."""
    lp = r.listener_prediction
    if not lp:
        return 50.0, {}

    return lp.overall_listener_score, {
        "comprehension": round(lp.comprehension, 1),
        "engagement": round(lp.engagement, 1),
        "trust": round(lp.trust, 1),
        "retention": round(lp.retention, 1),
        "attention": round(lp.attention_sustainability, 1),
    }


# ── Grade mapping ───────────────────────────────────────────────

def _grade(score: float) -> str:
    """Map composite score to letter grade."""
    if score >= 90:
        return "A+"
    if score >= 85:
        return "A"
    if score >= 80:
        return "B+"
    if score >= 75:
        return "B"
    if score >= 70:
        return "C+"
    if score >= 65:
        return "C"
    if score >= 55:
        return "D"
    return "F"


# ── Summary generator ──────────────────────────────────────────

def _summary(cats: list[CategoryScore]) -> str:
    """Generate a one-sentence narrative summary from category scores."""
    strengths = sorted(
        [c for c in cats if c.score >= 80],
        key=lambda c: c.score,
        reverse=True,
    )
    weaknesses = sorted(
        [c for c in cats if c.score < 60],
        key=lambda c: c.score,
    )

    parts: list[str] = []
    if strengths:
        names = ", ".join(
            c.category.replace("_", " ").title() for c in strengths[:3]
        )
        parts.append(f"Strong in {names}")
    if weaknesses:
        names = ", ".join(
            c.category.replace("_", " ").title() for c in weaknesses[:3]
        )
        parts.append(f"Improvement needed in {names}")
    if parts:
        return ". ".join(parts) + "."
    return "Balanced performance across all categories."


# ── Scorer registry ─────────────────────────────────────────────

_SCORERS = {
    "vocal_delivery":    _score_vocal_delivery,
    "fluency":           _score_fluency,
    "clarity":           _score_clarity,
    "language":          _score_language,
    "temporal_dynamics": _score_temporal,
    "cognitive_load":    _score_cognitive,
    "speaker_adaptive":  _score_adaptive,
    "coherence":         _score_coherence,
    "listener_score":    _score_listener,
}


# ── Main entry point ────────────────────────────────────────────

def compute_composite(result: SpeechAnalysisResult) -> CompositeScoreResult:
    """
    Compute the final weighted composite score.

    Parameters
    ----------
    result : a fully-populated SpeechAnalysisResult from the pipeline.

    Returns
    -------
    CompositeScoreResult with per-category scores, composite, grade,
    and narrative summary.
    """
    cats: list[CategoryScore] = []
    weighted_sum = 0.0

    for cat_name, scorer in _SCORERS.items():
        raw_score, details = scorer(result)
        weight = CATEGORY_WEIGHTS[cat_name]
        weighted = raw_score * weight
        cats.append(CategoryScore(
            category=cat_name,
            score=round(raw_score, 1),
            weight=round(weight, 4),
            weighted=round(weighted, 2),
            details=details,
        ))
        weighted_sum += weighted

    composite = round(weighted_sum, 1)
    g = _grade(composite)
    s = _summary(cats)

    logger.info("Composite score: %.1f (%s)", composite, g)
    return CompositeScoreResult(
        composite_score=composite,
        grade=g,
        category_scores=cats,
        summary=s,
    )
