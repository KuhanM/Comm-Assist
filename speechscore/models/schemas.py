"""
SpeechScore 2.0 — Data Schemas

Pydantic models that define the structured output of each analysis stage.
All per-window metrics are stored individually so that Phase 2 temporal
analysis can operate over the time series directly.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────────────────
# Transcription
# ────────────────────────────────────────────────────────────────────

class WordInfo(BaseModel):
    """A single word with timing and confidence from Whisper."""
    word: str
    start: float
    end: float
    probability: float = 0.0


class SegmentInfo(BaseModel):
    """A Whisper segment (roughly one sentence / phrase)."""
    id: int
    text: str
    start: float
    end: float
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 0.0


class TranscriptionResult(BaseModel):
    """Complete Whisper transcription output."""
    full_text: str = ""
    words: list[WordInfo] = Field(default_factory=list)
    segments: list[SegmentInfo] = Field(default_factory=list)
    language: str = "en"
    duration: float = 0.0


# ────────────────────────────────────────────────────────────────────
# Per-Window Metrics
# ────────────────────────────────────────────────────────────────────

class WindowMetrics(BaseModel):
    """
    All 12 per-window base metrics (Language metrics are global, not windowed).

    Categories:
      Vocal Delivery  — speech_rate_wpm, pitch_std, volume_consistency
      Fluency         — pause_frequency_per_min, mean_pause_duration,
                        filler_rate_per_100, phonation_ratio
      Clarity         — asr_confidence, word_recognition_rate
      [Prosody raw]   — pitch_mean, pitch_range, voiced_fraction (used in Phase 2)
    """

    # ── window identity ──
    window_id: int
    start_time: float                             # seconds
    end_time: float                               # seconds

    # ── Vocal Delivery (3 metrics) ──
    speech_rate_wpm: Optional[float] = None
    pitch_mean: Optional[float] = None
    pitch_std: Optional[float] = None             # F0 SD
    volume_consistency: Optional[float] = None

    # ── Fluency (4 metrics) ──
    pause_count: int = 0
    pause_frequency_per_min: Optional[float] = None
    mean_pause_duration: Optional[float] = None
    filler_count: int = 0
    filler_rate_per_100: Optional[float] = None
    phonation_ratio: Optional[float] = None

    # ── Clarity (2 metrics) ──
    asr_confidence: Optional[float] = None
    word_recognition_rate: Optional[float] = None

    # ── Raw temporal data (for Phase 2) ──
    pitch_range: Optional[float] = None
    voiced_fraction: Optional[float] = None
    rms_mean: Optional[float] = None
    rms_std: Optional[float] = None
    max_pause_duration: Optional[float] = None
    word_count: int = 0
    transcript: str = ""

    # ── Reliability (Phase 2) ──
    reliable: bool = True                         # False if Whisper may have hallucinated
    reliability_flags: list[str] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# Language Metrics (computed on full transcript, not per-window)
# ────────────────────────────────────────────────────────────────────

class LanguageMetrics(BaseModel):
    """Grammar, vocabulary, and syntactic complexity scores."""
    grammar_score: Optional[float] = None
    grammar_error_count: int = 0
    vocabulary_richness: Optional[float] = None   # Type-Token Ratio
    unique_word_count: int = 0
    total_word_count: int = 0
    sentence_complexity: Optional[float] = None   # clauses / sentences
    sentence_count: int = 0
    clause_count: int = 0


# ────────────────────────────────────────────────────────────────────
# Temporal Dynamics Metrics  ⭐ NOVEL (Phase 2)
# ────────────────────────────────────────────────────────────────────

class ConfidenceTrajectory(BaseModel):
    """Slope of pitch stability over temporal windows."""
    slope: float = 0.0                    # negative = gaining confidence
    r_squared: float = 0.0               # goodness of fit
    p_value: float = 1.0                  # statistical significance
    direction: str = "stable"             # "increasing" | "decreasing" | "stable"
    interpretation: str = ""
    per_window_pitch_std: list[float] = Field(default_factory=list)


class WarmupIndex(BaseModel):
    """Time to reach stable fluent state."""
    warmup_seconds: float = 0.0           # seconds to stabilise
    warmup_window: int = 0                # window index where stable
    pre_warmup_mean_wpm: float = 0.0
    post_warmup_mean_wpm: float = 0.0
    change_point_detected: bool = False
    speech_rate_trajectory: list[float] = Field(default_factory=list)


class FatigueDetection(BaseModel):
    """First-half vs second-half metric degradation."""
    fatigue_score: float = 0.0            # 0 = no fatigue, 100 = severe
    significant: bool = False             # any metric degraded significantly?
    first_half_means: dict = Field(default_factory=dict)
    second_half_means: dict = Field(default_factory=dict)
    degraded_metrics: list[str] = Field(default_factory=list)
    metric_details: dict = Field(default_factory=dict)  # per-metric p-value & effect


class EngagementArc(BaseModel):
    """Delivery energy curve vs ideal narrative arc."""
    score: float = 0.0                    # 0–100
    correlation: float = 0.0             # Pearson r with ideal template
    p_value: float = 1.0
    shape: str = "flat"                   # "ideal" | "u-shaped" | "declining" | "flat" | "rising"
    energy_trajectory: list[float] = Field(default_factory=list)
    ideal_template: list[float] = Field(default_factory=list)


class TemporalMetrics(BaseModel):
    """All temporal dynamics metrics (Phase 2, Day 4)."""
    confidence_trajectory: ConfidenceTrajectory = Field(
        default_factory=ConfidenceTrajectory
    )
    warmup_index: WarmupIndex = Field(default_factory=WarmupIndex)
    fatigue_detection: FatigueDetection = Field(default_factory=FatigueDetection)
    engagement_arc: EngagementArc = Field(default_factory=EngagementArc)


# ────────────────────────────────────────────────────────────────────
# Speaker Baseline  ⭐ NOVEL (Phase 2, Day 5)
# ────────────────────────────────────────────────────────────────────

class SpeakerBaseline(BaseModel):
    """
    Personal baseline extracted from the first N seconds of speech.

    This captures the speaker's *natural* communication style before
    nervousness / fatigue / content difficulty has an effect.  All
    subsequent metrics can be expressed relative to this baseline.
    """

    # seconds of audio used for baseline extraction
    baseline_duration: float = 0.0
    windows_used: int = 0

    # ── central tendency ──
    speech_rate_mean: float = 0.0          # WPM
    speech_rate_std: float = 0.0
    pitch_mean: float = 0.0                # Hz (F0 mean)
    pitch_std_mean: float = 0.0            # mean within-window F0 SD
    pitch_std_std: float = 0.0             # SD of within-window F0 SD
    volume_mean: float = 0.0               # mean RMS
    volume_std: float = 0.0                # SD of RMS
    pause_freq_mean: float = 0.0           # pauses per minute
    pause_freq_std: float = 0.0
    phonation_mean: float = 0.0            # phonation ratio
    phonation_std: float = 0.0
    filler_rate_mean: float = 0.0          # fillers per 100 words
    filler_rate_std: float = 0.0


class AdaptiveMetricScore(BaseModel):
    """A single metric scored relative to the speaker's own baseline."""
    metric_name: str = ""
    raw_value: float = 0.0                 # global aggregate
    baseline_value: float = 0.0            # from baseline windows
    z_score: float = 0.0                   # (raw − baseline_mean) / baseline_std
    percentile: float = 50.0               # estimated percentile (from z)
    deviation_label: str = "typical"       # "low" | "typical" | "high" | "extreme"
    interpretation: str = ""


class AdaptiveScoreResult(BaseModel):
    """
    Speaker-Adaptive Normalization result  ⭐ NOVEL

    Instead of comparing against population norms (which penalises
    natural speaking-style differences), each metric is expressed as
    a z-score relative to the speaker's own baseline.
    """

    baseline: SpeakerBaseline = Field(default_factory=SpeakerBaseline)
    adaptive_metrics: list[AdaptiveMetricScore] = Field(default_factory=list)

    # composite scores
    overall_adaptive_score: float = 0.0     # 0–100
    speech_rate_delta_pct: float = 0.0      # +8% = speaking 8% faster than baseline
    pitch_stability_ratio: float = 1.0      # >1 = less stable than baseline
    consistency_score: float = 0.0          # 0–100, how consistent vs baseline


# ────────────────────────────────────────────────────────────────────
# Cognitive Strain Index  ⭐ NOVEL (Phase 2, Day 6 — Contribution 3)
# ────────────────────────────────────────────────────────────────────

class StrugglePoint(BaseModel):
    """A single window flagged as high cognitive strain."""
    window_id: int = 0
    start_time: float = 0.0               # seconds
    end_time: float = 0.0                 # seconds
    csi_score: float = 0.0                # 0–100
    primary_cause: str = ""               # most-weighted strain indicator
    strain_breakdown: dict = Field(default_factory=dict)  # per-indicator contribution
    transcript_snippet: str = ""          # what was being said


class CognitiveStrainResult(BaseModel):
    """
    Cognitive Strain Index (CSI) — per-window and aggregate  ⭐ NOVEL

    Combines five physiological/linguistic strain indicators into a
    single per-window score that estimates how hard the speaker is
    working to produce fluent speech.

    Indicators (from spec):
      Pause excess      — 25%  — processing time needed
      Filler excess     — 25%  — word-finding difficulty
      SR deviation      — 20%  — uncertainty or rushing
      Pitch instability — 15%  — stress / discomfort
      Hesitation pattern— 15%  — planning difficulty
    """

    # per-window CSI scores (same length as window_metrics)
    per_window_csi: list[float] = Field(default_factory=list)

    # aggregate
    mean_csi: float = 0.0                  # 0–100
    max_csi: float = 0.0                   # 0–100
    min_csi: float = 0.0                   # 0–100
    std_csi: float = 0.0

    # struggle points — windows exceeding the strain threshold
    struggle_threshold: float = 60.0       # CSI > this → struggle point
    struggle_points: list[StrugglePoint] = Field(default_factory=list)
    struggle_count: int = 0
    struggle_pct: float = 0.0             # % of windows that are struggle points

    # per-indicator global means (for radar chart / breakdown)
    indicator_means: dict = Field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────
# Composite Score  (Day 7 — Scoring Integration)
# ────────────────────────────────────────────────────────────────────

class CategoryScore(BaseModel):
    """Score for one assessment category."""
    category: str = ""
    score: float = 0.0           # 0–100
    weight: float = 0.0          # 0–1 (normalised)
    weighted: float = 0.0        # score × weight
    details: dict = Field(default_factory=dict)  # sub-metric scores


class CompositeScoreResult(BaseModel):
    """Final weighted composite score with category breakdown."""
    composite_score: float = 0.0        # 0–100
    grade: str = ""                      # A+, A, B+, B, C+, C, D, F
    category_scores: list[CategoryScore] = Field(default_factory=list)
    summary: str = ""


# ────────────────────────────────────────────────────────────────────
# Multi-Modal Coherence  ⭐ NOVEL (Phase 2 stretch)
# ────────────────────────────────────────────────────────────────────

class CoherenceResult(BaseModel):
    """
    Cross-modal alignment between speech content and delivery.

    Three sub-metrics:
      1. Sentiment-Prosody — does vocal energy match text sentiment?
      2. Emphasis-Importance — are key words acoustically emphasised?
      3. Pause-Semantic — do pauses fall at syntactic boundaries?
    """
    sentiment_prosody_score: float = 50.0     # 0–100
    emphasis_alignment_score: float = 50.0    # 0–100
    pause_semantic_score: float = 50.0        # 0–100
    composite_coherence: float = 50.0         # weighted combo


# ────────────────────────────────────────────────────────────────────
# V2 Novel: Multiscale Entropy  ⭐ NOVEL (V2-1)
# ────────────────────────────────────────────────────────────────────

class ChannelEntropySchema(BaseModel):
    """Per-channel MSE result (Pydantic model for serialisation)."""
    channel: str = ""
    sample_entropy_by_scale: list[float] = Field(default_factory=list)
    complexity_index: float = 0.0
    ci_normalised: float = 0.0                  # 0–100
    profile_class: str = "monotonous"
    series_length: int = 0
    series_std: float = 0.0


class MultiscaleEntropySchema(BaseModel):
    """Pydantic schema mirroring MultiscaleEntropyResult for serialisation."""
    channels: list[ChannelEntropySchema] = Field(default_factory=list)
    composite_complexity: float = 50.0          # 0–100
    profile_class: str = "monotonous"
    interpretation: str = ""
    scales_used: int = 0
    min_series_length: int = 0


# ────────────────────────────────────────────────────────────────────
# V2 Novel: Recurrence Quantification Analysis  ⭐ NOVEL (V2-2)
# ────────────────────────────────────────────────────────────────────

class ChannelRQASchema(BaseModel):
    """Per-channel RQA result (Pydantic model for serialisation)."""
    channel: str = ""
    recurrence_rate: float = 0.0
    determinism: float = 0.0
    laminarity: float = 0.0
    trapping_time: float = 0.0
    max_diagonal: int = 0
    entropy_diagonal: float = 0.0
    n_embedded: int = 0
    radius: float = 0.0


class RecurrenceSchema(BaseModel):
    """Pydantic schema mirroring RecurrenceResult for serialisation."""
    channels: list[ChannelRQASchema] = Field(default_factory=list)
    predictability_score: float = 50.0          # 0–100
    consistency_score: float = 50.0             # 0–100
    fluidity_score: float = 50.0                # 0–100
    composite_rqa: float = 50.0                 # 0–100
    interpretation: str = ""
    embedding_dim: int = 2
    delay: int = 1


# ────────────────────────────────────────────────────────────────────
# V2 Novel: Information-Theoretic Coherence  ⭐ NOVEL (V2-3)
# ────────────────────────────────────────────────────────────────────

class ChannelPairInfoSchema(BaseModel):
    """Per-pair IT coherence result (Pydantic model for serialisation)."""
    channel_x: str = ""
    channel_y: str = ""
    mutual_information: float = 0.0
    normalised_mi: float = 0.0
    transfer_entropy_x_to_y: float = 0.0
    transfer_entropy_y_to_x: float = 0.0
    dominant_direction: str = "none"
    coupling_strength: str = "none"
    series_length: int = 0


class InfoTheoreticCoherenceSchema(BaseModel):
    """Pydantic schema mirroring InfoTheoreticCoherenceResult for serialisation."""
    channel_pairs: list[ChannelPairInfoSchema] = Field(default_factory=list)
    nonlinear_coherence: float = 50.0           # 0–100
    directional_flow: float = 50.0              # 0–100
    composite_it_coherence: float = 50.0        # 0–100
    interpretation: str = ""
    k_neighbours: int = 3


# ────────────────────────────────────────────────────────────────────
# Listener Prediction  (Phase 2 stretch)
# ────────────────────────────────────────────────────────────────────

class ListenerPrediction(BaseModel):
    """
    Predicted listener experience derived from speech metrics.

    All scores are 0–100 where higher = better listener experience.
    """
    comprehension: float = 50.0          # predicted understanding ease
    engagement: float = 50.0             # predicted attention level
    trust: float = 50.0                  # perceived speaker credibility
    retention: float = 50.0              # predicted information retention
    attention_sustainability: float = 50.0  # how long listener stays engaged
    overall_listener_score: float = 50.0
    details: dict = Field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────
# Complete Analysis Result
# ────────────────────────────────────────────────────────────────────

class SpeechAnalysisResult(BaseModel):
    """
    Full analysis output (Phase 1 + Phase 2).

    Contains:
      - Raw transcription (words + segments)
      - Per-window metrics for all 12 windowed base metrics
      - Global aggregated metrics per category
      - Language metrics (full-transcript scope)
      - Temporal dynamics metrics (Phase 2)  ⭐ NOVEL
    """

    # ── Metadata ──
    audio_file: str = ""
    duration: float = 0.0
    sample_rate: int = 16000

    # ── Transcription ──
    transcription: TranscriptionResult = Field(
        default_factory=TranscriptionResult
    )

    # ── Per-window ──
    window_metrics: list[WindowMetrics] = Field(default_factory=list)

    # ── Global aggregated (dicts keyed by metric name) ──
    global_acoustic: dict = Field(default_factory=dict)
    global_fluency: dict = Field(default_factory=dict)
    global_clarity: dict = Field(default_factory=dict)

    # ── Language ──
    language_metrics: Optional[LanguageMetrics] = None

    # ── Phase 2: Temporal Dynamics ⭐ NOVEL ──
    temporal_metrics: Optional[TemporalMetrics] = None

    # ── Phase 2: Speaker-Adaptive Normalization ⭐ NOVEL ──
    adaptive_score: Optional[AdaptiveScoreResult] = None

    # ── Phase 2: Cognitive Strain Index ⭐ NOVEL ──
    cognitive_strain: Optional[CognitiveStrainResult] = None

    # ── Day 7: Composite Score ──
    composite: Optional[CompositeScoreResult] = None

    # ── Stretch: Multi-Modal Coherence ⭐ NOVEL ──
    coherence: Optional[CoherenceResult] = None

    # ── Stretch: Listener Prediction ──
    listener_prediction: Optional[ListenerPrediction] = None

    # ── V2 Novel: Multiscale Entropy ⭐ NOVEL ──
    multiscale_entropy: Optional[MultiscaleEntropySchema] = None

    # ── V2 Novel: Recurrence Quantification Analysis ⭐ NOVEL ──
    recurrence_analysis: Optional[RecurrenceSchema] = None

    # ── V2 Novel: Information-Theoretic Coherence ⭐ NOVEL ──
    info_theoretic_coherence: Optional[InfoTheoreticCoherenceSchema] = None

    # ── Summary ──
    total_windows: int = 0
    analysis_timestamp: str = ""
