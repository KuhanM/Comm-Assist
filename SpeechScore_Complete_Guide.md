# SpeechScore 2.0 — Complete Guide

## Temporal-Adaptive Multi-Modal Communication Assessment Framework

> **One-line summary:** SpeechScore 2.0 analyses a spoken audio recording and produces a 0–100 composite score across 9 categories, using 4 **novel research contributions** that go beyond anything existing tools offer.

---

## Table of Contents

1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [How Existing Tools Work (and Their Limitations)](#2-how-existing-tools-work-and-their-limitations)
3. [What Makes SpeechScore 2.0 Novel](#3-what-makes-speechscore-20-novel)
4. [Architecture Overview](#4-architecture-overview)
5. [The 12-Step Pipeline](#5-the-12-step-pipeline)
6. [All Metrics Explained — The Full Taxonomy](#6-all-metrics-explained--the-full-taxonomy)
   - [Category 1: Vocal Delivery](#category-1-vocal-delivery-weight-15)
   - [Category 2: Fluency](#category-2-fluency-weight-15)
   - [Category 3: Clarity](#category-3-clarity-weight-10)
   - [Category 4: Language Quality](#category-4-language-quality-weight-10)
   - [Category 5: Temporal Dynamics ⭐ NOVEL](#category-5-temporal-dynamics--novel-weight-15)
   - [Category 6: Cognitive Strain ⭐ NOVEL](#category-6-cognitive-strain--novel-weight-5)
   - [Category 7: Speaker-Adaptive Normalization ⭐ NOVEL](#category-7-speaker-adaptive-normalization--novel-weight-10)
   - [Category 8: Multi-Modal Coherence ⭐ NOVEL](#category-8-multi-modal-coherence--novel-weight-10)
   - [Category 9: Listener Prediction](#category-9-listener-prediction-weight-10)
7. [Composite Scoring & Grading](#7-composite-scoring--grading)
8. [End-to-End Example with Real Output](#8-end-to-end-example-with-real-output)
9. [Comparison with Existing Approaches](#9-comparison-with-existing-approaches)
10. [Applications](#10-applications)
11. [Technical Stack](#11-technical-stack)
12. [Academic References](#12-academic-references)

---

## 1. What Problem Does This Solve?

When someone gives a speech, presentation, or interview, we intuitively sense whether they're "good" or "bad" — but we can't explain *why* precisely. Current speech assessment tools give you either:

- **A single score** with no actionable breakdown, or
- **Static averages** that miss how the speaker *changed over time*

**SpeechScore 2.0** solves both problems. Given any audio recording, it produces:

1. A **composite score** (0–100) with a letter grade (A+ through F)
2. **9 category sub-scores** so you know *exactly* where to improve
3. **Temporal analysis** — how you changed from minute 1 to minute 5
4. **Speaker-adaptive** normalization — scored against *your own* baseline, not a population average
5. **Cognitive strain mapping** — pinpoints the exact moments you struggled and *why*
6. **Multi-modal coherence** — whether your words and voice *agree*
7. **Listener prediction** — how your audience will *experience* your speech

---

## 2. How Existing Tools Work (and Their Limitations)

| Tool / Approach | What it does | Limitation |
|---|---|---|
| **Praat / Parselmouth** | Acoustic feature extraction (F0, intensity, formants) | Raw numbers only. No interpretation, no scoring. |
| **LIWC** | Psycholinguistic text analysis | Text-only. Ignores how words are *spoken*. |
| **OpenSMILE** | 6,000+ acoustic features (eGeMAPS) | Feature dump — no composite score, no temporal analysis. Too many features for interpretability. |
| **Speechace / ELSA** | Pronunciation scoring for language learners | Focused on non-native pronunciation, not communication quality. |
| **Grammarly / LanguageTool** | Grammar and writing quality | Text-only. Designed for written language, not spontaneous speech. |
| **Manual rubrics** (Toastmasters, debate scoring) | Human-rated categories | Subjective, inconsistent, not automated, expensive. |

### What they all miss

1. **No temporal analysis** — They average everything. A speaker who starts terribly but finishes strong gets the same score as one who is consistent throughout. The *trajectory* of communication quality is invisible.

2. **No speaker adaptation** — A naturally fast speaker (180 WPM baseline) who slows to 150 WPM is actually adjusting well. A naturally slow speaker (100 WPM) at 150 WPM is rushing. Population-based norms can't distinguish these.

3. **No multi-modal coherence** — Text and audio are analysed separately. Nobody checks whether the *emotional content* of words matches the *vocal delivery* (saying "I'm thrilled!" in a flat monotone).

4. **No cognitive load estimation** — Existing tools can count pauses and fillers, but don't combine them into a *cognitive strain* model that identifies *why* the speaker struggled (word-finding difficulty? Information overload? Fatigue?).

---

## 3. What Makes SpeechScore 2.0 Novel

SpeechScore 2.0 introduces **4 novel research contributions** — techniques not found in any existing speech assessment tool:

### Novel Contribution 1: Temporal Dynamics Analysis

**The idea:** Instead of computing a single average WPM or average pitch stability, we track these metrics *across overlapping time windows* and analyse the **trajectory**.

- **Confidence Trajectory** — Is pitch getting more or less stable over time? (Statistical regression with p-values)
- **Fluency Warmup Index** — When does the speaker settle into a rhythm? (CUSUM change-point detection)
- **Fatigue Detection** — Does the second half degrade vs. the first half? (Welch's t-test + Cohen's d effect size)
- **Engagement Arc** — Does the speaker's energy follow a narrative arc? (Template matching against ideal/rising/u-shaped/declining/flat patterns)

**Why it matters:** A job interview candidate who is nervous for the first 30 seconds but then speaks confidently should not be penalised the same as someone who is shaky throughout. Temporal analysis captures the *story* of the speech.

**Prior art gap:** Mixdorff et al. (2018) studied F0 stability but only as a static average. Reagan et al. (2016) proposed narrative arcs for *text* stories. Nobody has applied temporal trajectory analysis to *live speech assessment*.

### Novel Contribution 2: Speaker-Adaptive Normalization

**The idea:** Use the first 30 seconds of your speech to build a *personal baseline* (your own mean ± standard deviation for each metric). Then score the rest of your speech as z-scores *relative to your own baseline*.

```
z = (metric_value − your_baseline_mean) / your_baseline_std
```

**Why it matters:** 
- A woman with F0 = 220 Hz and a man with F0 = 120 Hz are not "better" or "worse" — they have different baselines
- A naturally expressive speaker (high pitch variation) is different from a naturally calm one
- An accented speaker with lower ASR confidence should be scored on their *own* clarity trajectory, not an idealised native speaker

**Prior art gap:** Levelt (1989) documented intra-speaker variation. Cohen (1988) formalised z-score interpretation. But no speech assessment tool uses the speaker's *own early speech* as a personalised baseline for scoring.

### Novel Contribution 3: Cognitive Strain Index (CSI)

**The idea:** Combine 6 physiological and linguistic indicators into a per-window "cognitive load" score, then identify **struggle points** — specific moments where the speaker was under high mental load.

The 6 indicators:
1. **Pause excess** (20%) — More pauses than your baseline → need processing time
2. **Filler excess** (20%) — More "um"/"uh" than your baseline → word-finding difficulty
3. **Speech rate deviation** (15%) — Speeding up (rushing) or slowing down (struggling)
4. **Pitch instability** (10%) — Voice cracking/wavering → stress
5. **Hesitation pattern** (15%) — Pauses AND fillers co-occurring → planning difficulty
6. **Clarity strain** (20%) — ASR confidence dropping → articulatory breakdown

Each indicator is normalised to [0, 100] using a sigmoid activation relative to the speaker's baseline. The composite CSI is their weighted sum.

**The "struggle point" timeline:** When CSI exceeds a threshold (default: 40), we flag that window as a struggle point and identify the **primary cause** (e.g., "you struggled at 1:30–1:40, primarily due to filler excess, suggesting word-finding difficulty").

**Why it matters:** Instead of just saying "you used too many fillers," we can say "you had a cognitive strain spike at 1:30 caused by simultaneous filler excess and speech rate deviation — you were trying to find the right word while maintaining pace."

**Prior art gap:** Sweller (1988) proposed Cognitive Load Theory for learning. Goldman-Eisler (1968) linked pauses to cognitive planning. Lickley (2015) catalogued disfluency types. But nobody has combined these into a *real-time, multi-indicator, speaker-adaptive* cognitive strain timeline for speech assessment.

### Novel Contribution 4: Multi-Modal Coherence

**The idea:** Cross-reference **what the speaker says** (text semantics) with **how they say it** (acoustic prosody) to measure *alignment*.

Three sub-metrics:
1. **Sentiment-Prosody Coherence** — Does vocal energy match text emotion? (Pearson correlation between VADER sentiment and RMS energy per window)
2. **Emphasis-Importance Alignment** — Are key words (named entities, nouns, verbs) acoustically emphasised? (Elongation > 1.2× mean duration, or preceded by a > 0.15s pause)
3. **Pause-Semantic Synchronization** — Do pauses fall at sentence/clause boundaries or mid-phrase? (spaCy dependency parsing + ±0.5s tolerance matching)

**Why it matters:** A politician who says "I am deeply concerned" in a cheerful, upbeat voice has low coherence — the message doesn't match the delivery. This misalignment erodes trust even if both text and voice are individually "fine."

**Prior art gap:** Bänziger & Scherer (2005) studied prosody-emotion alignment in laboratory conditions. Pierrehumbert (1980) developed F0 emphasis theory. Goldman-Eisler (1972) linked pauses to syntactic planning. Nobody has integrated all three into an automated coherence metric for speech assessment.

---

## 4. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     AUDIO FILE (.wav/.mp3/.m4a)                  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 1: Load Audio (librosa, 16kHz mono)                        │
│  Step 2: Transcribe (Whisper large-v3 + hallucination filter)    │
│  Step 3: Create Temporal Windows (10s window, 5s hop, 50% overlap)│
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 4: Per-Window Feature Extraction                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                  │
│  │  Acoustic   │  │  Fluency   │  │  Clarity   │                  │
│  │ (Praat/F0)  │  │ (pauses,   │  │ (ASR conf, │                  │
│  │ WPM, vol.   │  │  fillers,  │  │  word      │                  │
│  │ consistency │  │  phonation)│  │  recog.)   │                  │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                  │
│        └───────────────┼───────────────┘                          │
│                        ▼                                          │
│           list[WindowMetrics] — 12 metrics per window             │
└──────────────────────┬───────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────────────────┐
        ▼              ▼                          ▼
┌──────────────┐ ┌───────────────┐ ┌──────────────────────┐
│ Step 5:      │ │ Step 7:       │ │ Step 8:              │
│ Language     │ │ Global        │ │ Baseline Extraction  │
│ Analysis     │ │ Aggregation   │ │ (first 30s → mean±σ) │
│ (grammar,    │ │ (means, σ     │ │ ⭐ NOVEL             │
│  TTR, clause)│ │  across       │ └──────────┬───────────┘
│ [full text]  │ │  windows)     │            │
└──────┬───────┘ └───────┬───────┘            │
       │                 │                     │
       ▼                 ▼                     ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 6:  Temporal Dynamics ⭐ NOVEL                               │
│ Step 9:  Speaker-Adaptive Scoring ⭐ NOVEL                        │
│ Step 10: Cognitive Strain Index ⭐ NOVEL                          │
│ Step 11: Multi-Modal Coherence ⭐ NOVEL                           │
│ Step 12: Listener Prediction (derived from all above)             │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Composite Scoring Engine                                        │
│  9 categories × piecewise-linear normalisation → weighted sum    │
│  → 0–100 score → letter grade (A+ through F)                     │
└──────────────────────────────────────────────────────────────────┘
```

### The Windowing Strategy

Why overlapping windows? Consider a 2-minute speech:

```
Audio:  |========================================|
Win 0:  [0────10s]
Win 1:      [5────15s]          ← 50% overlap
Win 2:          [10────20s]
Win 3:              [15────25s]
...
Win 22:                                [105──111s]
```

Each 10-second window produces 12 metrics. The 50% overlap ensures smooth transitions — no abrupt jumps at window boundaries. This creates a **time series** for every metric, enabling temporal analysis.

---

## 5. The 12-Step Pipeline

| Step | Name | Module | Input | Output |
|------|------|--------|-------|--------|
| 1 | Load Audio | `audio_utils.py` | File path | Raw waveform (16kHz mono) |
| 2 | Transcribe | `transcription.py` | Waveform | Words + segments + probabilities |
| 3 | Create Windows | `audio_utils.py` | Waveform | 10s overlapping audio slices |
| 4 | Per-Window Analysis | `acoustic.py` + `fluency.py` + `clarity.py` | Each window | 12 metrics per window |
| 5 | Language Analysis | `language.py` | Full transcript | Grammar, vocab richness, complexity |
| 6 | Temporal Dynamics ⭐ | `temporal.py` | Window time series | Confidence, warmup, fatigue, arc |
| 7 | Global Aggregation | Built-in | All window metrics | Means + standard deviations |
| 8 | Baseline Extraction ⭐ | `baseline.py` | First 6 windows | Personal mean ± σ per metric |
| 9 | Adaptive Scoring ⭐ | `adaptive_scorer.py` | All windows + baseline | Z-scores per metric |
| 10 | Cognitive Strain ⭐ | `cognitive.py` | All windows + baseline | CSI per window + struggle points |
| 11 | Coherence ⭐ | `coherence.py` | Windows + transcription | 3 coherence sub-scores |
| 12 | Listener Prediction | `listener.py` | Full result | 5 listener dimensions |

---

## 6. All Metrics Explained — The Full Taxonomy

### Category 1: Vocal Delivery (Weight: 15%)

These measure *how the voice sounds* — independent of what words are being said.

| Metric | What It Measures | How It's Computed | Optimal Range | Why It Matters |
|--------|-----------------|-------------------|---------------|----------------|
| **Speech Rate (WPM)** | Speed of delivery | Word count ÷ window duration × 60 | 120–150 WPM | Too fast → audience can't follow. Too slow → boring. |
| **Pitch Variation (F0 SD)** | Vocal expressiveness | Standard deviation of fundamental frequency (Praat) | 20–50 Hz | Low → monotone, disengaging. High → anxious/erratic. |
| **Volume Consistency** | Steadiness of loudness | 1 − (σ of RMS energy ÷ mean RMS energy) | > 0.70 | Low → volume swings wildly; hard to listen to. |

**Scoring example:** A speaker at 140 WPM with F0 SD = 30 Hz and volume consistency = 0.85 would score ~93/100 for vocal delivery — right in the sweet spot.

**How speech rate scoring works (piecewise-linear):**
```
WPM    Score
 60  →  20    (too slow)
 90  →  50
120  →  90
135  →  100   (optimal)
150  →  90
180  →  60    (too fast)
220  →  30    (much too fast)
260  →  10
```

### Category 2: Fluency (Weight: 15%)

These measure *how smoothly* the speaker delivers — pauses, fillers, and silence.

| Metric | What It Measures | How It's Computed | Optimal Range | Why It Matters |
|--------|-----------------|-------------------|---------------|----------------|
| **Pause Frequency** | How often the speaker stops | Energy-based VAD: frames below adaptive threshold → pause (≥0.2s) | < 20/min | Excessive pauses → choppy, uncertain delivery |
| **Mean Pause Duration** | Average length of each pause | Mean of all detected pause durations | 0.25–0.5s | Very short = natural breathing. Very long = struggling. |
| **Filler Rate** | How many "um", "uh", "like" etc. | Lexical matching against curated filler lexicon | < 3 per 100 words | Fillers signal word-finding difficulty and reduce credibility |
| **Phonation Ratio** | What proportion of time the speaker is actually speaking | Voiced frames ÷ total frames | 60–80% | Low → too much silence. Very high → no pauses (unnatural). |

**The filler lexicon includes:**
- Single words: `um`, `uh`, `er`, `ah`, `hmm`, `like`, `basically`, `actually`, `literally`, `right`, `well`, `okay`
- Bigrams: `you know`, `sort of`, `kind of`, `i mean`, `okay so`, `so basically`

### Category 3: Clarity (Weight: 10%)

These measure *how clearly* the speaker articulates — can an AI transcription system understand them?

| Metric | What It Measures | How It's Computed | Optimal Range | Why It Matters |
|--------|-----------------|-------------------|---------------|----------------|
| **ASR Confidence** | Overall transcription confidence | Mean of Whisper's per-word probability scores | > 0.85 | Low confidence = Whisper struggled → speaker was unclear |
| **Word Recognition Rate** | Fraction of clearly spoken words | % of words with probability ≥ 0.5 | > 0.95 | Identifies the proportion of "clear" vs "mumbled" words |

**Insight:** Whisper assigns a probability (0–1) to every word it transcribes. Words spoken clearly get probabilities of 0.9+. Mumbled or unclear words might get 0.3–0.5. We use this as a *proxy for articulatory clarity* — it's not measuring pronunciation correctness (like accent), but how clearly the sound reaches the listener.

### Category 4: Language Quality (Weight: 10%)

These measure the *linguistic content* — grammar, vocabulary, and sentence structure.

| Metric | What It Measures | How It's Computed | Optimal Range | Why It Matters |
|--------|-----------------|-------------------|---------------|----------------|
| **Grammar Score** | Grammatical correctness | 1 − (grammar/typo/spelling errors ÷ sentence count), via LanguageTool | > 0.85 | Errors confuse listeners and reduce credibility |
| **Vocabulary Richness (TTR)** | Word diversity | Unique lemmas ÷ total alphabetic tokens (spaCy) | 0.4–0.6 | Low → repetitive. Very high → possibly unfocused. |
| **Sentence Complexity** | Structural sophistication | Average clause count per sentence (spaCy dependency parsing) | 1.5–2.5 | Simple = easy to follow. Too complex = hard to track. |

**Note:** These are computed on the *full transcript*, not per-window, because grammar and vocabulary analysis need broader context to be meaningful.

### Category 5: Temporal Dynamics ⭐ NOVEL (Weight: 15%)

This is the **primary novel contribution**. Instead of static averages, we analyse how metrics *change over time*.

#### 5a. Confidence Trajectory

| Aspect | Detail |
|--------|--------|
| **What** | Linear trend of pitch stability (F0 SD) over time |
| **Insight** | Lower F0 SD = more controlled voice = higher confidence |
| **Method** | OLS regression: pitch_std ~ window_index |
| **Output** | Slope, R², p-value, direction (increasing/stable/decreasing) |
| **Interpretation** | Negative slope → pitch is stabilising → speaker is *gaining* confidence |

**Example:** A slope of −0.334 Hz/window with p=0.033 means the speaker's voice became significantly more stable over time — they grew more confident. This is classified as "increasing confidence."

#### 5b. Fluency Warmup Index

| Aspect | Detail |
|--------|--------|
| **What** | The point where the speaker settles into a stable speech rate |
| **Method** | CUSUM (Cumulative Sum) change-point detection on WPM time series |
| **Output** | Warmup time (seconds), pre/post warmup mean WPM, whether change was significant |
| **Interpretation** | Shorter warmup relative to speech duration = better |

**How CUSUM works:**
1. Compute the overall mean speech rate
2. Calculate cumulative deviations from that mean
3. The peak of the cumulative sum = the change point
4. Test whether pre-change and post-change distributions differ significantly (Welch's t-test, p < 0.05)

**Example:** Warmup at 15s in a 120s speech = speaker needed 12.5% of the time to settle in. Score: `100 × (1 − 0.125 × 2) = 75`.

#### 5c. Fatigue Detection

| Aspect | Detail |
|--------|--------|
| **What** | Whether the second half of the speech degrades compared to the first half |
| **Method** | Welch's t-test + Cohen's d effect size on 4 metrics |
| **Metrics tested** | Pause frequency (↑ = fatigue), phonation ratio (↓ = fatigue), ASR confidence (↓ = fatigue), speech rate (any change = instability) |
| **Output** | Fatigue score (0–100), statistical significance, degraded metrics list |

**The fatigue score** is the mean normalised effect size across all metrics that showed degradation, scaled to [0, 100]. The interpretation:
- 0–20: No fatigue detected
- 20–40: Mild changes in second half
- 40–60: Moderate fatigue effects
- 60–80: Significant degradation
- 80–100: Severe fatigue

**Fallback for short speeches:** When there are fewer than 15 windows per half (not enough for a t-test), we use Cohen's d effect size directly: d > 0.5 = medium effect → flagged as fatigue.

#### 5d. Engagement Arc

| Aspect | Detail |
|--------|--------|
| **What** | Whether the speaker's energy curve follows a narrative arc |
| **Method** | Normalise RMS energy curve, correlate against 5 ideal templates |
| **Templates** | Ideal (╱╲), Rising (╱), U-shaped (╲╱), Declining (╲), Flat (—) |
| **Output** | Shape name, best-fit correlation, score (0–100) |

**The 5 templates:**
```
Ideal:     ╱╲    (build up → climax → wind down — best for presentations)
Rising:    ╱     (starts quiet, ends strong — good for building momentum)
U-shaped:  ╲╱    (dips in middle → recovers — acceptable in longer talks)
Flat:      ───   (monotone energy — boring but consistent)
Declining: ╲     (starts strong, fades — worst pattern)
```

**Score calcuation:** The best-matching template gets a correlation score, then:
- "ideal" match: `score = 60 + r × 40` (max 100)
- "rising": `score = 50 + r × 40` (max 90)
- "u-shaped": `score = 40 + r × 35` (max 75)
- "flat": `score = 30 + r × 25` (max 55)
- "declining": `score = 10 + r × 30` (max 40)

### Category 6: Cognitive Strain ⭐ NOVEL (Weight: 5%)

The Cognitive Strain Index (CSI) estimates *mental load* per window, then identifies **struggle points**.

#### The 6 Indicators

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cognitive Strain Index                         │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐        │
│  │ Pause Excess │  │ Filler Excess│  │ Speech Rate Dev. │        │
│  │     20%      │  │     20%      │  │       15%        │        │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘        │
│         │                 │                    │                   │
│  ┌──────┴───────┐  ┌──────┴────────┐  ┌───────┴────────┐        │
│  │   Pitch     │  │  Hesitation  │  │  Clarity       │        │
│  │ Instability │  │   Pattern    │  │   Strain       │        │
│  │    10%      │  │     15%      │  │     20%        │        │
│  └─────────────┘  └──────────────┘  └────────────────┘        │
│                                                                   │
│  Each indicator: 0 = at baseline → 100 = extreme deviation        │
│  CSI = weighted sum (0–100)                                       │
│  Struggle Point = window where CSI > 40                           │
└─────────────────────────────────────────────────────────────────┘
```

**How each indicator is computed:**

Every indicator uses a **sigmoid activation** relative to the speaker's personal baseline:

$$score = \frac{100}{1 + e^{-k(z - m)}}$$

Where $z$ = the z-score (how many standard deviations from baseline), $k$ = 3.0 (sensitivity), $m$ = 1.5 (midpoint at which score ≈ 50).

This means:
- At baseline (z = 0) → score ≈ 0 (no strain)
- At z = 1.5 → score ≈ 50 (moderate strain)
- At z = 3.0 → score ≈ 90 (severe strain)

#### Struggle Point Detection

When a window's CSI exceeds the threshold (40), we create a **Struggle Point** record:
- Which window (time range)
- CSI score
- **Primary cause** (the indicator with the highest value)
- Full breakdown of all 6 indicators
- Transcript snippet of what was being said

**Example Struggle Point:**
```
⚠ [45–55s]  CSI = 44  cause: speech_rate_dev
  "Along with development, I also enjoy learning..."
    ○ pause_excess          0.0
    ● filler_excess        73.9
    ● speech_rate_dev      99.5
    ○ pitch_instability     0.0
    ○ hesitation_pattern    0.0
    ● clarity_strain       73.4
```

**Reading this:** At 45–55 seconds, the speaker's cognitive load spiked. The primary cause was speech rate deviation (99.5/100) — they were speaking much faster or slower than their baseline. Filler excess (73.9) and clarity strain (73.4) also contributed. This suggests the speaker was *rushing through a thought while losing articulatory control*.

### Category 7: Speaker-Adaptive Normalization ⭐ NOVEL (Weight: 10%)

#### How the Baseline Works

```
Audio:  |===========|==================================|
        ↑ First 30s ↑          Rest of speech
        ← BASELINE →
        
        For each metric, compute:
          mean = average over baseline windows
          std  = standard deviation over baseline windows
        
        Then for the full speech:
          z = (overall_mean − baseline_mean) / baseline_std
```

**Baseline metrics captured:**
| Metric | Baseline Field | Why |
|--------|---------------|-----|
| Speech Rate (WPM) | `speech_rate_mean` ± `speech_rate_std` | Personal speaking speed |
| Pitch Stability (F0 SD) | `pitch_std_mean` ± `pitch_std_std` | Natural voice variability |
| Volume (RMS) | `volume_mean` ± `volume_std` | Personal loudness level |
| Pause Frequency | `pause_freq_mean` ± `pause_freq_std` | Natural pause rhythm |
| Phonation Ratio | `phonation_mean` ± `phonation_std` | Speaking vs silence balance |
| Filler Rate | `filler_rate_mean` ± `filler_rate_std` | Natural filler usage |

#### Z-Score Interpretation

| z-score | Label | Meaning |
|---------|-------|---------|
| |z| < 0.5 | **Typical** | Within normal personal range |
| 0.5 ≤ |z| < 1.5 | **Moderate** | Noticeable change from baseline |
| 1.5 ≤ |z| < 2.5 | **High** | Significant departure from norm |
| |z| ≥ 2.5 | **Extreme** | Major change — likely noteworthy |

**Direction matters:**
- Pitch stability z = −1.0 (lower F0 SD) → "moderate improvement" (voice got more controlled)
- Filler rate z = +1.13 (higher fillers) → "moderate degradation" (more fillers than usual)

#### Composite Adaptive Score

The composite adaptive score maps the z-vector to 0–100:
1. **Winsorise** all z-scores to ±3 (prevents outliers from dominating)
2. For metrics where deviation direction matters: penalise bad direction, reward good
3. Combine via weighted sigmoid → 0–100

**Consistency Score:** Measures how *stable* the speaker was throughout — computed from the variance of per-window z-scores. Low variance = high consistency.

### Category 8: Multi-Modal Coherence ⭐ NOVEL (Weight: 10%)

This is the only category that crosses the **text ↔ audio** boundary.

#### 8a. Sentiment-Prosody Coherence (35% of coherence)

```
Per window:
  Text sentiment (VADER compound score, −1 to +1)
      ↕ correlated with ↕
  Vocal energy (normalised RMS z-scores)

Then: Pearson correlation → map |r| to 0–100
  r = 0   → 50  (no alignment)
  |r| = 1 → 100 (perfect alignment)
```

**Example:** If a speaker says positive things with high energy and negative things with low energy, r ≈ +0.7 → score ≈ 85. If there is no relationship between sentiment and energy, r ≈ 0 → score ≈ 50.

**Why absolute correlation?** Both positive and negative correlation indicate coherence. A sad story told with low, quiet energy (negative correlation) is *coherent*. Only zero correlation indicates misalignment.

#### 8b. Emphasis-Importance Alignment (35% of coherence)

```
Step 1: Identify important words using spaCy NLP:
   - Named entities ("Python", "Google", "AI")
   - Noun chunk heads ("development", "technology")
   - Content verbs ("build", "explore", "improve")

Step 2: For each important word, check for acoustic emphasis:
   - Duration > 1.2× average word duration (elongation)
   - Preceding gap > 0.15s (emphasising pause)

Step 3: Score = % of important words that got emphasis
```

**Example:** If spaCy identifies 20 important words and 12 of them were acoustically emphasised → score = 60/100.

#### 8c. Pause-Semantic Synchronization (30% of coherence)

```
Step 1: Find all pauses > 0.25s (from word timestamps)

Step 2: Find sentence/clause boundaries (spaCy parsing)

Step 3: For each pause, check if it's within ±0.5s of a 
        syntactic boundary

Step 4: Score = % of pauses at boundaries
```

**Why it matters:** Pauses mid-sentence ("I went to... the store") signal struggle. Pauses between sentences ("I went to the store. [pause] Then I...") are natural and professional.

#### Composite Coherence

$$\text{Coherence} = SP \times 0.35 + EI \times 0.35 + PS \times 0.30$$

### Category 9: Listener Prediction (Weight: 10%)

This category is **entirely derived** — no new features are extracted. It combines all previous metrics to predict how the *listener* will experience the speech.

#### The 5 Dimensions

| Dimension | What It Predicts | Key Inputs | Weights |
|-----------|-----------------|------------|---------|
| **Comprehension** | How easily the listener can follow | Speech rate, ASR confidence, vocabulary, complexity, grammar | 30%, 25%, 15%, 15%, 15% |
| **Engagement** | How captivated the listener stays | Pitch variation, energy dynamics, engagement arc, filler rate | 30%, 25%, 25%, 20% |
| **Trust** | Perceived speaker credibility | Adaptive consistency, cognitive strain, fluency, coherence | 30%, 25%, 25%, 20% |
| **Retention** | How much the listener remembers | Geometric mean of comprehension × engagement | multiplicative |
| **Attention Sustainability** | How long the listener stays engaged | Fatigue score, engagement arc shape, temporal consistency | 40%, 35%, 25% |

#### Overall Listener Score

$$\text{Listener} = \text{comp} \times 0.25 + \text{eng} \times 0.25 + \text{trust} \times 0.20 + \text{ret} \times 0.15 + \text{attn} \times 0.15$$

**The retention formula is multiplicative on purpose:**

$$\text{Retention} = \sqrt{\text{comprehension} \times \text{engagement}}$$

This means if *either* comprehension or engagement is low, retention drops sharply. You can't remember something you didn't understand, and you can't remember something you weren't paying attention to. The geometric mean captures this interaction.

---

## 7. Composite Scoring & Grading

### Category Weights

| Category | Weight | Rationale |
|----------|--------|-----------|
| Vocal Delivery | 15% | Core physical delivery |
| Fluency | 15% | Smoothness of expression |
| Temporal Dynamics | 15% | How delivery evolves (novel contribution, high weight) |
| Clarity | 10% | Articulatory clarity |
| Language | 10% | Content quality |
| Speaker Adaptive | 10% | Personal consistency |
| Coherence | 10% | Text-voice alignment |
| Listener Score | 10% | Audience experience |
| Cognitive Load | 5% | Lower weight because it's partly captured by other categories |

**Total: 100%**

### Scoring Method

Each category uses **piecewise-linear interpolation** — a set of (value, score) breakpoints connected by straight lines. This avoids the parametric assumptions of a normal distribution and allows us to encode domain knowledge directly.

For example, the speech rate scorer:
```python
s_wpm = piecewise(wpm, [
    (60, 20),   # 60 WPM → score 20
    (90, 50),   # 90 WPM → score 50
    (120, 90),  # 120 WPM → score 90
    (135, 100), # 135 WPM → score 100 (optimal)
    (150, 90),  # 150 WPM → score 90
    (180, 60),  # 180 WPM → score 60
    (220, 30),  # 220 WPM → score 30
])
```

### Grade Mapping

| Score Range | Grade |
|-------------|-------|
| ≥ 90 | A+ |
| ≥ 85 | A |
| ≥ 80 | B+ |
| ≥ 75 | B |
| ≥ 70 | C+ |
| ≥ 65 | C |
| ≥ 55 | D |
| < 55 | F |

### Score Interpretation

| Grade | Meaning |
|-------|---------|
| **A+ / A** | Exceptional communicator — engaging, clear, well-paced, stable |
| **B+ / B** | Good communicator — some areas for refinement but solid overall |
| **C+ / C** | Average — noticeable weaknesses but gets the message across |
| **D** | Below average — multiple areas need significant work |
| **F** | Poor — major issues across multiple dimensions |

---

## 8. End-to-End Example with Real Output

### Input

**Audio file:** `Audio_Kuhan.m4a` — a 111-second self-introduction recording  
**Ground truth content:** Speaker introduces themselves, talks about interest in technology, chess, hackathons, favourite food, and closes.

### Step-by-Step Processing

**Step 1–2: Load & Transcribe**
```
Duration: 111.3 seconds, 16000 Hz
Whisper large-v3 produced 51 raw segments
Hallucination filter dropped 42/51 segments:
  - 16 "zero-duration" (0s segments)
  - 20 "repeated-text" (same phrase looping)
  - 6 "empty" segments
Result: 9 clean segments, 159 words
```

**Step 3: Create Windows**
```
23 windows (10s window, 5s hop), covering 0–111.3s
```

**Step 4: Per-Window Analysis (sample)**
```
Window 0 (0–10s):  WPM=132  F0σ=16.0  fillers=0  ASR=0.97
Window 5 (25–35s): WPM=144  F0σ=32.1  fillers=1  ASR=0.94
Window 12 (60–70s): WPM=0   F0σ=33.2  fillers=0  ASR=0.00  ← silent gap
```

**Step 5: Language Analysis**
```
Grammar Score:       0.91  (1 error in 11 sentences)
Vocabulary Richness: 0.55  (TTR — moderate diversity)
Sentence Complexity: 2.36  (avg 2.36 clauses per sentence)
```

**Step 6: Temporal Dynamics**
```
Confidence:     INCREASING (slope=−0.334, p=0.033)
   → Pitch became more stable over time
Warmup:         60s (window 12) — took half the speech to settle
Fatigue:        100/100 (significant=True)
   → Second half showed degradation
Engagement Arc: "rising" (score=50/100)
   → Energy increased but didn't follow ideal arc
```

**Steps 8–10: Adaptive + Cognitive**
```
Baseline: first 35s (6 windows), WPM=125±7.0

Adaptive z-scores:
  Speech Rate:   z = −6.81  EXTREME  (much slower than baseline)
  Pitch Stab.:   z = −1.03  moderate (voice got more controlled)
  Volume:        z = +0.49  typical
  Filler Rate:   z = +1.13  moderate (more fillers than baseline)

Cognitive Strain: mean=27.1/100
  6 struggle points out of 23 windows (26%):
  ⚠ [45–55s]  CSI=44  (speech_rate_dev + filler_excess)
  ⚠ [70–80s]  CSI=44  (speech_rate_dev)
  ⚠ [75–85s]  CSI=55  (filler_excess)
```

**Steps 11–12: Coherence + Listener**
```
Coherence: 55.1/100
  Sentiment-Prosody: 51.7  (weak text-voice correlation)
  Emphasis-Importance: 46.0  (key words not emphasised)
  Pause-Semantic: 69.4  (pauses mostly at boundaries — good)

Listener Prediction: 70.2/100
  Comprehension:     82.1  (easy to follow)
  Engagement:        67.0  (moderately engaging)
  Trust:             70.4  (credible)
  Retention:         74.2  (moderate retention)
  Attention Sust.:   51.9  (attention may wander)
```

### Final Composite Score

```
╔════════════════════════════════════════════════════════════╗
║  ⭐ COMPOSITE SCORE: 64.2 / 100  (Grade: D)              ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Category                Score  Weight  Contribution       ║
║  ─────────────────────  ─────  ──────  ────────────       ║
║  Vocal Delivery          57.5   15.0%      8.6            ║
║  Fluency                 77.8   15.0%     11.7            ║
║  Clarity                 91.0   10.0%      9.1            ║
║  Language                92.7   10.0%      9.3            ║
║  Temporal Dynamics       36.2   15.0%      5.4            ║
║  Cognitive Load          43.7    5.0%      2.2            ║
║  Speaker Adaptive        53.4   10.0%      5.3            ║
║  Coherence               55.1   10.0%      5.5            ║
║  Listener Score          70.2   10.0%      7.0            ║
║                                                            ║
║  Summary: Strong in Language, Clarity.                     ║
║  Improvement needed in Temporal Dynamics, Cognitive Load,  ║
║  Speaker Adaptive.                                         ║
╚════════════════════════════════════════════════════════════╝
```

### How to Read This

**Strengths (≥ 80):**
- **Clarity: 91** — The speaker enunciates clearly; Whisper has high confidence in every word
- **Language: 93** — Good grammar, diverse vocabulary, appropriately complex sentences

**Moderate (60–80):**
- **Fluency: 78** — Decent but some pauses and fillers
- **Listener: 70** — Audience would follow but might lose attention

**Weaknesses (< 60):**
- **Temporal Dynamics: 36** — The speech didn't have a good arc; fatigue was detected
- **Cognitive Load: 44** — 6 struggle points identified
- **Coherence: 55** — Words and voice don't always align

**Actionable feedback for this speaker:**
1. "Work on maintaining consistent energy throughout — your second half showed fatigue"
2. "Practise emphasising key words with slight pauses before them"
3. "Your warmup period was 60 seconds — try to settle into your rhythm within the first 15 seconds"
4. "At 45–55 seconds, you experienced a strain spike from speech rate deviation — try to maintain a steady pace when transitioning between topics"

---

## 9. Comparison with Existing Approaches

### Feature-by-Feature Comparison

| Capability | Praat | OpenSMILE | Speechace | LIWC | **SpeechScore 2.0** |
|---|:---:|:---:|:---:|:---:|:---:|
| Acoustic features (F0, energy) | ✅ | ✅ | ✅ | ❌ | ✅ |
| Fluency (pauses, fillers) | Partial | Partial | ❌ | ❌ | ✅ |
| ASR-based clarity | ❌ | ❌ | ✅ | ❌ | ✅ |
| Grammar & vocabulary | ❌ | ❌ | ❌ | Partial | ✅ |
| **Temporal trajectory** | ❌ | ❌ | ❌ | ❌ | ⭐ **Yes** |
| **Speaker-adaptive scoring** | ❌ | ❌ | ❌ | ❌ | ⭐ **Yes** |
| **Cognitive strain mapping** | ❌ | ❌ | ❌ | ❌ | ⭐ **Yes** |
| **Multi-modal coherence** | ❌ | ❌ | ❌ | ❌ | ⭐ **Yes** |
| **Listener experience prediction** | ❌ | ❌ | ❌ | ❌ | ✅ |
| Single composite score | ❌ | ❌ | ✅ | ❌ | ✅ |
| Actionable feedback | ❌ | ❌ | Partial | ❌ | ✅ |
| Works with any speaker | ✅ | ✅ | Limited | ✅ | ✅ |
| No training data needed | ✅ | ✅ | ❌ | ✅ | ✅ |

### Key Differentiators

1. **SpeechScore 2.0 is unsupervised.** There is no training dataset, no labelled examples, no model fitting. All scoring is based on signal processing, statistical tests, and domain-knowledge breakpoints. This means it works for any speaker, any accent, any topic — without bias from training data.

2. **Every score is interpretable and traceable.** Unlike black-box ML models, every number can be traced back to a specific acoustic feature, time window, and mathematical operation. You can explain *why* any score is what it is.

3. **The speaker is compared to themselves.** The adaptive normalization eliminates population-level bias. A naturally fast speaker (180 WPM baseline) who slows to 160 WPM is *adapting well*, not "speaking too fast."

4. **Temporal granularity.** The 10s/5s windowing strategy produces a time series for every metric, enabling trajectory analysis, change-point detection, and temporal degradation testing.

### What Makes This Publishable (Q1/Q2)?

For an academic publication in venues like INTERSPEECH, ACL, or IEEE SPL, reviewers look for:

1. **Novel contributions** — 4 clearly defined novel techniques that don't exist in prior art
2. **Statistical rigour** — Welch's t-test, Cohen's d, linear regression with p-values, Pearson correlation
3. **Reproducibility** — No trained models, no proprietary data, deterministic pipeline
4. **Interpretability** — Every metric is explainable (unlike end-to-end neural approaches)
5. **Practical value** — Direct applications in education, hiring, clinical assessment

---

## 10. Applications

### Education & Training

| Use Case | How SpeechScore Helps |
|----------|----------------------|
| **Public speaking courses** | Students get objective, repeatable feedback on presentations |
| **Language learning** | Track fluency improvement over weeks/months (adaptive baseline evolves) |
| **Debate coaching** | Engagement arc analysis shows if speakers maintain energy |
| **Teacher training** | Cognitive strain mapping helps identify when teachers lose students |

### Professional Development

| Use Case | How SpeechScore Helps |
|----------|----------------------|
| **Interview preparation** | Practise and get scored; track warmup time reduction |
| **Sales training** | Coherence analysis ensures passion matches pitch content |
| **Executive coaching** | Listener prediction scores identify trust and credibility gaps |
| **Podcast/YouTube creators** | Engagement arc analysis helps optimise content pacing |

### Healthcare & Clinical

| Use Case | How SpeechScore Helps |
|----------|----------------------|
| **Speech therapy** | Track patient progress quantitatively (adaptive scoring handles individual variation) |
| **Cognitive assessment** | CSI tracks cognitive load — potential early indicator of cognitive decline |
| **Parkinson's/ALS monitoring** | Temporal analysis of pitch stability and phonation ratio over sessions |
| **Mental health screening** | Fatigue detection and pitch trends may correlate with mood state |

### Research

| Use Case | How SpeechScore Helps |
|----------|----------------------|
| **Psycholinguistics** | Cognitive strain decomposition provides new variables for study |
| **Communication studies** | Multi-modal coherence enables cross-modal alignment research |
| **Political speech analysis** | Compare temporal dynamics across speakers, speeches, contexts |
| **Second language acquisition** | Track fluency warmup evolution as proficiency improves |

### HR & Recruitment

| Use Case | How SpeechScore Helps |
|----------|----------------------|
| **Structured interviews** | Objective speech quality metrics alongside content evaluation |
| **Bias reduction** | Speaker-adaptive scoring means accents, gender, and speaking styles don't bias the score |
| **Candidate benchmarking** | Compare composite scores across candidates fairly |

---

## 11. Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **ASR** | OpenAI Whisper `large-v3` | Speech-to-text with word-level timestamps and probabilities |
| **Pitch analysis** | Parselmouth (Praat bindings) | F0 extraction, voiced frame detection |
| **Audio processing** | librosa | Loading, resampling, RMS energy, windowing |
| **NLP** | spaCy `en_core_web_lg` | POS tagging, NER, dependency parsing, sentence segmentation |
| **Grammar** | language-tool-python | Rule-based grammar/spelling error detection |
| **Sentiment** | vaderSentiment | Lexicon-based text sentiment scoring |
| **Statistics** | scipy.stats | t-tests, linear regression, Pearson correlation, normal CDF |
| **Data models** | Pydantic v2 | Typed schemas, JSON serialisation |
| **Config** | Python dataclasses | Centralised, typed configuration |
| **Testing** | pytest (184 tests) | Unit tests across all 7 test files |

### Key Design Decisions

1. **CPU for Whisper** — Although the M4 Mac has MPS (GPU), Whisper's DTW-based word timestamp alignment requires float64 which MPS doesn't support. We force CPU for correctness.

2. **`condition_on_previous_text=False`** — Critical Whisper setting. When set to `True` (the default), Whisper can enter repetition loops on certain audio, fabricating the same phrase dozens of times. Setting to `False` prevents this hallucination mode.

3. **`temperature=0.0`** — Deterministic decoding ensures reproducible transcriptions.

4. **Multi-layer hallucination filter** — Even with the above settings, a safety net catches: zero-duration segments, repeated-text (exact duplicate > 5 chars), empty segments, and low log-probability (< −1.0) segments.

5. **No ML training** — The entire system is unsupervised. All scoring logic uses analytical functions (piecewise-linear, sigmoid, statistical tests). This means no training bias and full interpretability.

---

## 12. Academic References

| Reference | Used Where | Contribution |
|-----------|-----------|--------------|
| Mixdorff et al. (2018) INTERSPEECH | Confidence Trajectory | F0 stability as a confidence indicator |
| Reagan et al. (2016) EPJ | Engagement Arc | Narrative arc shapes in stories |
| Vonnegut (1995) | Engagement Arc | Original narrative arc taxonomy |
| Killick et al. (2012) JASA | Warmup Index | PELT/CUSUM change-point detection |
| Lefter et al. (2021) Speech Communication | Fatigue Detection | Speech degradation over time |
| Sweller (1988) | Cognitive Strain Index | Cognitive Load Theory |
| Goldman-Eisler (1968, 1972) | CSI + Pause-Semantic | Pauses as cognitive planning indicators |
| Lickley (2015) Handbook of Pragmatics | CSI | Fluency/disfluency taxonomy |
| Cohen (1988) | Adaptive Scoring | Z-score and effect size interpretation |
| Levelt (1989) | Adaptive Scoring | Intra-speaker variation |
| Bänziger & Scherer (2005) | Sentiment-Prosody | Prosody-emotion alignment |
| Pierrehumbert (1980) | Emphasis-Importance | F0 theory and emphasis marking |
| Welch (1947) | Fatigue Detection | Welch's t-test for unequal variances |

---

## Quick-Start Summary

```
Input:  Any audio file (WAV, MP3, M4A, FLAC)
Output: 0–100 composite score with letter grade
        9 category sub-scores
        Per-window time series (12 metrics × N windows)
        Struggle point timeline with root causes
        Listener experience prediction

Novelty:
  1. Temporal Dynamics    — trajectory, not averages
  2. Speaker-Adaptive     — scored vs. yourself, not the population
  3. Cognitive Strain     — per-window decomposed mental load
  4. Multi-Modal Coherence — text ↔ voice alignment

No training data. No black box. Every score is explainable.
```

```bash
# Run it
cd Comm-Assist
source .venv/bin/activate
python run_analysis.py path/to/your/audio.wav
```

---

*SpeechScore 2.0 — Making communication assessment objective, adaptive, and explainable.*
