#!/usr/bin/env python3
"""
SpeechScore 2.0 — CLI Runner

Usage:
    python run_analysis.py <audio_file>
    python run_analysis.py speech.wav
    python run_analysis.py recording.mp3

Supported formats: WAV, MP3, M4A, FLAC, OGG
"""

import sys
import json
import logging

from speechscore.config.settings import SpeechScoreConfig
from speechscore.pipeline import SpeechScorePipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_analysis.py <audio_file>")
        print("Example: python run_analysis.py speech.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  SpeechScore 2.0 — Full Analysis")
    print("=" * 60)
    print(f"  File: {audio_path}")
    print()

    # initialize pipeline (loads Whisper — takes ~10s first time)
    config = SpeechScoreConfig()
    pipeline = SpeechScorePipeline(config)
    pipeline.initialize()

    # run analysis
    result = pipeline.analyze(audio_path)

    # print results
    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)

    print(f"\n  Duration        : {result.duration:.1f} s")
    print(f"  Windows         : {result.total_windows}")
    print(f"  Words           : {len(result.transcription.words)}")

    # ── Composite Score ──
    cp = result.composite
    if cp:
        print(f"\n{'─' * 60}")
        print(f"  ⭐ COMPOSITE SCORE: {cp.composite_score:.1f}/100  ({cp.grade})")
        print(f"{'─' * 60}")
        print(f"  {cp.summary}")
        print()
        print(f"  {'Category':<25} {'Score':>6} {'Weight':>7} {'Contrib':>8}")
        print(f"  {'─'*25} {'─'*6} {'─'*7} {'─'*8}")
        for cs in cp.category_scores:
            name = cs.category.replace('_', ' ').title()
            print(f"  {name:<25} {cs.score:>5.1f}  {cs.weight*100:>5.1f}%  {cs.weighted:>7.1f}")

    # ── Transcript ──
    print(f"\n{'─' * 60}")
    print("  TRANSCRIPT")
    print(f"{'─' * 60}")
    text = result.transcription.full_text
    # wrap at 70 chars
    for i in range(0, len(text), 70):
        print(f"  {text[i:i+70]}")

    # ── Vocal Delivery ──
    ga = result.global_acoustic
    if ga:
        print(f"\n{'─' * 60}")
        print("  VOCAL DELIVERY")
        print(f"{'─' * 60}")
        print(f"  Avg Speech Rate  : {ga.get('avg_speech_rate_wpm', 0):.0f} WPM"
              f"  (optimal: 120–150)")
        print(f"  Speech Rate σ    : {ga.get('speech_rate_std', 0):.1f} WPM")
        print(f"  Pitch Mean       : {ga.get('global_pitch_mean', 0):.1f} Hz")
        print(f"  Pitch Variation  : {ga.get('global_pitch_std', 0):.1f} Hz"
              f"  (optimal: 20–50)")
        print(f"  Volume Consist.  : {ga.get('global_volume_consistency', 0):.2f}"
              f"  (optimal: >0.70)")

    # ── Fluency ──
    gf = result.global_fluency
    if gf:
        print(f"\n{'─' * 60}")
        print("  FLUENCY")
        print(f"{'─' * 60}")
        print(f"  Total Pauses     : {gf.get('total_pause_count', 0)}")
        print(f"  Avg Pause Dur.   : {gf.get('global_mean_pause_duration', 0):.2f} s"
              f"  (optimal: <0.5)")
        print(f"  Total Fillers    : {gf.get('total_filler_count', 0)}")
        print(f"  Filler Rate      : {gf.get('global_filler_rate_per_100', 0):.1f}"
              f" per 100 words  (optimal: <3)")
        print(f"  Phonation Ratio  : {gf.get('avg_phonation_ratio', 0)*100:.1f}%"
              f"  (optimal: 60–80%)")

    # ── Clarity ──
    gc = result.global_clarity
    if gc:
        print(f"\n{'─' * 60}")
        print("  CLARITY")
        print(f"{'─' * 60}")
        print(f"  ASR Confidence   : {gc.get('global_asr_confidence', 0):.2f}"
              f"  (optimal: >0.80)")
        print(f"  Word Recog. Rate : {gc.get('global_word_recognition_rate', 0):.2f}"
              f"  (optimal: >0.90)")
        print(f"  Low Conf. Words  : {gc.get('total_low_confidence_words', 0)}")

    # ── Language ──
    lm = result.language_metrics
    if lm:
        print(f"\n{'─' * 60}")
        print("  LANGUAGE QUALITY")
        print(f"{'─' * 60}")
        gs = f"{lm.grammar_score:.2f}" if lm.grammar_score is not None else "N/A (Java needed)"
        print(f"  Grammar Score    : {gs}  (optimal: >0.85)")
        print(f"  Grammar Errors   : {lm.grammar_error_count}")
        print(f"  Vocab Richness   : {lm.vocabulary_richness:.2f}"
              f"  (optimal: 0.40–0.70)")
        print(f"  Unique Words     : {lm.unique_word_count} / {lm.total_word_count}")
        print(f"  Sent. Complexity : {lm.sentence_complexity:.2f}"
              f" clauses/sent  (optimal: 1.5–2.5)")

    # ── Per-Window Timeline ──
    if result.window_metrics:
        print(f"\n{'─' * 60}")
        print("  PER-WINDOW TIMELINE")
        print(f"{'─' * 60}")
        print(f"  {'Win':>3}  {'Time':>10}  {'WPM':>5}  {'F0σ':>5}"
              f"  {'Pauses':>6}  {'Fill':>4}  {'ASR':>5}  {'Phon':>5}")
        print(f"  {'───':>3}  {'──────────':>10}  {'─────':>5}  {'─────':>5}"
              f"  {'──────':>6}  {'────':>4}  {'─────':>5}  {'─────':>5}")
        for wm in result.window_metrics:
            t0 = f"{wm.start_time:.0f}-{wm.end_time:.0f}s"
            wpm = f"{wm.speech_rate_wpm:.0f}" if wm.speech_rate_wpm else "—"
            f0 = f"{wm.pitch_std:.1f}" if wm.pitch_std else "—"
            asr = f"{wm.asr_confidence:.2f}" if wm.asr_confidence else "—"
            pho = f"{wm.phonation_ratio:.2f}" if wm.phonation_ratio else "—"
            flag = " ⚠" if not wm.reliable else ""
            print(f"  {wm.window_id:>3}  {t0:>10}  {wpm:>5}  {f0:>5}"
                  f"  {wm.pause_count:>6}  {wm.filler_count:>4}  {asr:>5}  {pho:>5}{flag}")
        # show reliability warnings
        unreliable = [wm for wm in result.window_metrics if not wm.reliable]
        if unreliable:
            print(f"\n  ⚠ {len(unreliable)} unreliable window(s) flagged:")
            for wm in unreliable:
                print(f"    win {wm.window_id}: {', '.join(wm.reliability_flags)}")

    # ── Temporal Dynamics ⭐ NOVEL ──
    tm = result.temporal_metrics
    if tm:
        ct = tm.confidence_trajectory
        print(f"\n{'─' * 60}")
        print("  ⭐ TEMPORAL DYNAMICS (Novel)")
        print(f"{'─' * 60}")

        print(f"\n  ▸ Confidence Trajectory")
        print(f"    Direction      : {ct.direction}")
        print(f"    Slope          : {ct.slope:.4f} Hz/window")
        print(f"    R²             : {ct.r_squared:.4f}")
        print(f"    p-value        : {ct.p_value:.4f}"
              f"  {'✓ significant' if ct.p_value < 0.05 else '✗ not significant'}")
        print(f"    Interpretation : {ct.interpretation}")
        print(f"    F0σ trajectory : {ct.per_window_pitch_std}")

        wi = tm.warmup_index
        print(f"\n  ▸ Fluency Warmup Index")
        print(f"    Warmup time    : {wi.warmup_seconds:.0f} s  (window {wi.warmup_window})")
        print(f"    Pre-warmup WPM : {wi.pre_warmup_mean_wpm:.0f}")
        print(f"    Post-warmup WPM: {wi.post_warmup_mean_wpm:.0f}")
        print(f"    Change detected: {'Yes' if wi.change_point_detected else 'No'}")
        print(f"    WPM trajectory : {wi.speech_rate_trajectory}")

        fd = tm.fatigue_detection
        print(f"\n  ▸ Fatigue Detection")
        print(f"    Fatigue score  : {fd.fatigue_score:.1f} / 100"
              f"  (0=fresh, 100=severe)")
        print(f"    Significant    : {'Yes' if fd.significant else 'No'}")
        if fd.degraded_metrics:
            print(f"    Degraded       : {', '.join(fd.degraded_metrics)}")
        print(f"    1st half means : {fd.first_half_means}")
        print(f"    2nd half means : {fd.second_half_means}")
        for mname, mdet in fd.metric_details.items():
            marker = "⚠" if mdet["degraded"] else " "
            print(f"    {marker} {mname:25s}  d={mdet['cohens_d']:+.3f}  "
                  f"p={mdet['p_value']:.4f}")

        ea = tm.engagement_arc
        print(f"\n  ▸ Engagement Arc")
        print(f"    Score          : {ea.score:.1f} / 100")
        print(f"    Shape          : {ea.shape}")
        print(f"    Correlation    : {ea.correlation:.4f}  (p={ea.p_value:.4f})")
        print(f"    Energy curve   : {ea.energy_trajectory}")

    # ── Speaker-Adaptive Normalization ⭐ NOVEL ──
    ad = result.adaptive_score
    if ad:
        bl = ad.baseline
        print(f"\n{'─' * 60}")
        print("  ⭐ SPEAKER-ADAPTIVE NORMALIZATION (Novel)")
        print(f"{'─' * 60}")

        print(f"\n  ▸ Speaker Baseline (first {bl.baseline_duration:.0f} s,"
              f" {bl.windows_used} windows)")
        print(f"    Speech Rate    : {bl.speech_rate_mean:.0f} ± {bl.speech_rate_std:.1f} WPM")
        print(f"    Pitch (F0)     : {bl.pitch_mean:.1f} ± mean F0 SD"
              f" {bl.pitch_std_mean:.1f} Hz")
        print(f"    Volume (RMS)   : {bl.volume_mean:.4f} ± {bl.volume_std:.4f}")
        print(f"    Pause Freq     : {bl.pause_freq_mean:.1f} ± {bl.pause_freq_std:.1f} /min")
        print(f"    Phonation      : {bl.phonation_mean:.2f} ± {bl.phonation_std:.3f}")
        print(f"    Filler Rate    : {bl.filler_rate_mean:.1f} ± {bl.filler_rate_std:.1f} /100w")

        print(f"\n  ▸ Adaptive Scores")
        print(f"    Overall Score  : {ad.overall_adaptive_score:.1f} / 100")
        print(f"    Consistency    : {ad.consistency_score:.1f} / 100")
        print(f"    SR vs baseline : {ad.speech_rate_delta_pct:+.1f}%")
        print(f"    Pitch stab.    : {ad.pitch_stability_ratio:.2f}x baseline")

        print(f"\n  ▸ Per-Metric Adaptive Z-Scores")
        print(f"    {'Metric':30s}  {'Raw':>8}  {'Baseline':>8}"
              f"  {'z':>6}  {'Label':>10}")
        print(f"    {'─'*30}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*10}")
        for am in ad.adaptive_metrics:
            print(f"    {am.metric_name:30s}  {am.raw_value:8.3f}"
                  f"  {am.baseline_value:8.3f}  {am.z_score:+6.2f}"
                  f"  {am.deviation_label:>10}")
        for am in ad.adaptive_metrics:
            if am.deviation_label != "typical":
                print(f"    → {am.interpretation}")

    # ── Cognitive Strain Index ⭐ NOVEL ──
    cs = result.cognitive_strain
    if cs:
        print(f"\n{'─' * 60}")
        print("  ⭐ COGNITIVE STRAIN INDEX (Novel)")
        print(f"{'─' * 60}")

        print(f"\n  ▸ Aggregate")
        print(f"    Mean CSI       : {cs.mean_csi:.1f} / 100  (0=relaxed, 100=severe)")
        print(f"    Max CSI        : {cs.max_csi:.1f}")
        print(f"    Min CSI        : {cs.min_csi:.1f}")
        print(f"    Std CSI        : {cs.std_csi:.1f}")

        print(f"\n  ▸ Per-Window CSI Timeline")
        for i, csi_val in enumerate(cs.per_window_csi):
            bar_len = int(csi_val / 2)  # 50 chars = 100
            bar = "█" * bar_len
            flag = " ⚠" if csi_val > cs.struggle_threshold else ""
            print(f"    win {i:2d}  {csi_val:5.1f}  {bar}{flag}")

        print(f"\n  ▸ Strain Indicators (global means)")
        for name, mean_val in cs.indicator_means.items():
            print(f"    {name:25s}  {mean_val:5.1f} / 100")

        print(f"\n  ▸ Struggle Points  ({cs.struggle_count} of"
              f" {len(cs.per_window_csi)} windows,"
              f" {cs.struggle_pct:.0f}%)")
        if cs.struggle_points:
            for sp in cs.struggle_points:
                print(f"    ⚠ [{sp.start_time:.0f}–{sp.end_time:.0f}s]"
                      f"  CSI={sp.csi_score:.0f}"
                      f"  cause: {sp.primary_cause}")
                if sp.transcript_snippet:
                    snip = sp.transcript_snippet[:80]
                    print(f"      \"{snip}...\"")
                for ind_name, ind_val in sp.strain_breakdown.items():
                    marker = "●" if ind_val > 50 else "○"
                    print(f"        {marker} {ind_name:25s}  {ind_val:5.1f}")
        else:
            print("    None — speaker maintained low cognitive load throughout.")

    print(f"\n{'=' * 60}")
    print()

    # ── Coherence ⭐ NOVEL ──
    coh = result.coherence
    if coh:
        print(f"\n{'─' * 60}")
        print("  ⭐ MULTI-MODAL COHERENCE (Novel)")
        print(f"{'─' * 60}")
        print(f"  Composite        : {coh.composite_coherence:.1f} / 100")
        print(f"  Sentiment-Prosody: {coh.sentiment_prosody_score:.1f}"
              f"  (text emotion ↔ vocal energy)")
        print(f"  Emphasis-Importance: {coh.emphasis_alignment_score:.1f}"
              f"  (key-word emphasis)")
        print(f"  Pause-Semantic   : {coh.pause_semantic_score:.1f}"
              f"  (pause placement)")

    # ── Listener Prediction ──
    lp = result.listener_prediction
    if lp:
        print(f"\n{'─' * 60}")
        print("  LISTENER PREDICTION")
        print(f"{'─' * 60}")
        print(f"  Overall          : {lp.overall_listener_score:.1f} / 100")
        print(f"  Comprehension    : {lp.comprehension:.1f}"
              f"  (ease of understanding)")
        print(f"  Engagement       : {lp.engagement:.1f}"
              f"  (listener attention)")
        print(f"  Trust            : {lp.trust:.1f}"
              f"  (speaker credibility)")
        print(f"  Retention        : {lp.retention:.1f}"
              f"  (information retention)")
        print(f"  Attention Sust.  : {lp.attention_sustainability:.1f}"
              f"  (staying engaged)")

    print(f"\n{'=' * 60}")
    print()

    # ── Save full JSON ──
    out_path = audio_path.rsplit(".", 1)[0] + "_analysis.json"
    with open(out_path, "w") as f:
        json.dump(result.model_dump(), f, indent=2, default=str)
    print(f"  Full results saved to: {out_path}")
    print()

    pipeline.close()


if __name__ == "__main__":
    main()
