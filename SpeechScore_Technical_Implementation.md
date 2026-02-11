# SpeechScore 2.0: Technical Implementation Guide
## Temporal-Adaptive Multi-Modal Communication Assessment Framework

---

## 1. System Overview

**SpeechScore 2.0** is a novel communication evaluation system that goes beyond basic metric aggregation by introducing:
- **Temporal Dynamics Analysis** - How metrics evolve over time
- **Speaker Baseline Adaptation** - Personalized benchmarks per speaker
- **Multi-Modal Coherence Scoring** - Prosody-semantic alignment
- **Cognitive Load Estimation** - Mental strain detection
- **Listener-Centric Prediction** - Predicted comprehension & engagement

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WEB APPLICATION                             │
│                    (React Frontend + FastAPI)                       │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   NOVEL PROCESSING PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│  1. Audio Input + Transcription (Whisper)                           │
│  2. LAYER 1: Baseline Extraction (first 30 seconds)                 │
│  3. LAYER 2: Temporal Feature Extraction (windowed analysis)        │
│  4. LAYER 3: Multi-Modal Coherence Analysis                         │
│  5. LAYER 4: Cognitive Load Estimation                              │
│  6. LAYER 5: Listener-Centric Prediction                            │
│  7. Adaptive Weighted Scoring Engine                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Novel Contributions

### 2.1 Key Innovations

| Innovation | Description | Research Gap Addressed |
|------------|-------------|----------------------|
| **Temporal Dynamics** | Analyze HOW metrics change over time, not just averages | Most systems use static aggregate values |
| **Baseline Adaptation** | Personalize benchmarks to individual speaker | One-size-fits-all benchmarks are unfair |
| **Multi-Modal Coherence** | Measure alignment between prosody and semantics | Audio and text analyzed separately |
| **Cognitive Load Estimation** | Infer speaker's mental strain from acoustic cues | No existing metric for speaker struggle |
| **Listener Prediction** | Predict how audience would perceive the speech | Speaker-centric vs. listener-centric |

### 2.2 Paper Title (Suggested)

> **"Beyond Acoustic Metrics: A Temporal Multi-Modal Framework for Communication Assessment with Listener-Centric Prediction"**

---

## 3. Complete Metrics Framework (27 Total)

### 3.1 Base Metrics (15 Metrics)

#### Category 1: Vocal Delivery (3 metrics, 15% weight)
| Metric | Formula | Optimal Range | Tool |
|--------|---------|---------------|------|
| Speech Rate (WPM) | `word_count / (duration / 60)` | 120-150 WPM | Whisper |
| Pitch Variation (F0 SD) | `std(f0)` | 20-50 Hz | parselmouth |
| Volume Consistency | `1 - (std(RMS) / mean(RMS))` | >0.7 | librosa |

#### Category 2: Fluency (4 metrics, 15% weight)
| Metric | Formula | Optimal Range | Tool |
|--------|---------|---------------|------|
| Pause Frequency | `pause_count / minutes` | <8/min | VAD |
| Mean Pause Duration | `mean(pause_durations)` | 0.25-0.75s | librosa |
| Filler Word Rate | `fillers / (words / 100)` | <3/100 | Whisper |
| Phonation Time Ratio | `speech_time / total_time` | 60-80% | VAD |

#### Category 3: Clarity (2 metrics, 10% weight)
| Metric | Formula | Optimal Range | Tool |
|--------|---------|---------------|------|
| ASR Confidence | `mean(word_confidences)` | >0.85 | Whisper |
| Word Recognition Rate | `recognized / total` | >0.95 | Whisper |

#### Category 4: Language (3 metrics, 10% weight)
| Metric | Formula | Optimal Range | Tool |
|--------|---------|---------------|------|
| Grammar Score | `1 - (errors / sentences)` | >0.9 | LanguageTool |
| Vocabulary Richness | `unique_words / total_words` | >0.4 | spaCy |
| Sentence Complexity | `clauses / sentences` | 1.5-2.5 | spaCy |

#### Category 5: Prosody (3 metrics, 10% weight)
| Metric | Formula | Optimal Range | Tool |
|--------|---------|---------------|------|
| Intonation Variation | `range(pitch_slope)` | Variable | parselmouth |
| Energy Variation | `std(energy_contour)` | Moderate | librosa |
| Emotional Tone | Classification confidence | Context-dep | HuggingFace |

---

### 3.2 Novel Metrics (12 Metrics) ⭐ NEW

#### Category 6: Temporal Dynamics (4 metrics, 15% weight) ⭐ NOVEL
| Metric | Description | What It Captures |
|--------|-------------|------------------|
| **Confidence Trajectory** | Slope of pitch stability over time windows | Does speaker become more/less confident? |
| **Engagement Arc Score** | Similarity of energy curve to ideal narrative arc | Does delivery follow engaging pattern? |
| **Fluency Warmup Index** | Time to achieve stable fluent state | How quickly does speaker settle in? |
| **Fatigue Detection Score** | Metric degradation in latter half vs. first half | Does speaker tire over time? |

#### Category 7: Adaptive Baseline (3 metrics, 10% weight) ⭐ NOVEL
| Metric | Description | What It Captures |
|--------|-------------|------------------|
| **Personal Speech Rate Delta** | Deviation from speaker's own baseline WPM | Speaking faster/slower than normal? |
| **Personal Pitch Stability** | Variance ratio compared to baseline | More/less stable than speaker's norm? |
| **Adaptive Benchmark Score** | Overall score using personalized benchmarks | Fair assessment relative to individual |

#### Category 8: Multi-Modal Coherence (3 metrics, 10% weight) ⭐ NOVEL
| Metric | Description | What It Captures |
|--------|-------------|------------------|
| **Sentiment-Prosody Coherence** | Correlation of text sentiment + audio emotion | Does voice match content emotion? |
| **Emphasis-Importance Alignment** | Acoustic emphasis on key words/entities | Are important words highlighted? |
| **Pause-Semantic Synchronization** | Pause alignment with sentence boundaries | Are pauses well-placed for meaning? |

#### Category 9: Cognitive Load (2 metrics, 5% weight) ⭐ NOVEL
| Metric | Description | What It Captures |
|--------|-------------|------------------|
| **Cognitive Strain Index (CSI)** | Composite: pause + filler + rate deviation | Overall mental load indicator |
| **Struggle Point Detection** | Time segments with high cognitive load | Where does speaker struggle? |

---

## 4. Listener-Centric Predictions ⭐ NOVEL

### 4.1 Predicted Outcomes

| Prediction | Based On | Output |
|------------|----------|--------|
| **Comprehension Score** | Speech rate, vocabulary, complexity, grammar | % content listener would understand |
| **Engagement Score** | Pitch variation, energy dynamics, arc pattern | Predicted listener attention level |
| **Trust/Credibility Score** | Confidence indicators, fluency, coherence | Perceived speaker credibility |
| **Retention Estimate** | Comprehension × engagement factors | % content listener would remember |
| **Attention Sustainability** | Engagement decay rate calculation | Minutes before attention drops 50% |

### 4.2 Listener Report Components

- Overall Listener Score (weighted combination)
- Predicted Listener Response (positive/mixed/challenging)
- Specific Improvement Recommendations
- Attention span prediction

---

## 5. Adaptive Scoring System

### 5.1 Two-Stage Scoring

**Stage 1: Extract Speaker Baseline (First 30 seconds)**
- Establish speaker's natural speech rate
- Capture typical pitch patterns
- Record baseline pause/filler frequency
- Use these as personalized reference points

**Stage 2: Adaptive Evaluation**
- Compare full speech to speaker's OWN baseline
- Adjust optimal ranges per individual
- Generate both absolute and relative scores

### 5.2 Final Weight Distribution

| Category | Weight | Type |
|----------|--------|------|
| Vocal Delivery | 15% | Base |
| Fluency | 15% | Base |
| Clarity | 10% | Base |
| Language | 10% | Base |
| Prosody | 10% | Base |
| **Temporal Dynamics** | 15% | Novel |
| **Coherence** | 10% | Novel |
| **Cognitive Load** | 5% | Novel |
| **Listener Prediction** | 10% | Novel |

---

## 6. Temporal Analysis Approach

### 6.1 Windowed Analysis
- Divide audio into overlapping windows (10s window, 5s hop)
- Calculate metrics per window
- Analyze trajectories across windows

### 6.2 Key Temporal Insights

| Insight | Method |
|---------|--------|
| Confidence trend | Linear regression on pitch stability |
| Energy arc | DTW similarity to ideal narrative curve |
| Warmup detection | First window where metrics stabilize |
| Fatigue detection | First-half vs. second-half comparison |

---

## 7. Multi-Modal Coherence Analysis

### 7.1 Sentiment-Prosody Alignment
- Extract text sentiment (positive/negative) per segment
- Extract audio emotion per segment
- Calculate correlation between the two
- High correlation = coherent communication

### 7.2 Emphasis-Importance Matching
- Identify important words (NER, keywords)
- Check if these words have:
  - Higher pitch than surrounding words
  - Longer duration than average
  - Preceding pause for emphasis
- Calculate alignment percentage

### 7.3 Pause-Semantic Sync
- Map pause locations
- Map sentence/clause boundaries
- Calculate overlap within tolerance
- High sync = natural phrasing

---

## 8. Cognitive Load Estimation

### 8.1 Strain Indicators
| Indicator | Weight | What It Shows |
|-----------|--------|---------------|
| Pause excess | 25% | Processing time needed |
| Filler excess | 25% | Word-finding difficulty |
| Speech rate deviation | 20% | Uncertainty or rushing |
| Pitch instability | 15% | Stress/discomfort |
| Hesitation patterns | 15% | Planning difficulty |

### 8.2 Struggle Point Detection
- Calculate CSI per window
- Flag windows exceeding threshold
- Output timestamps and likely causes
- Enable targeted improvement feedback

---

## 9. Project Structure

```
speechscore/
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── requirements.txt
│   ├── config/
│   │   └── config.py
│   ├── analyzers/
│   │   ├── transcription.py       # Whisper STT
│   │   ├── acoustic.py            # Pitch, volume, rate
│   │   ├── fluency.py             # Pauses, fillers
│   │   ├── language.py            # Grammar, vocabulary
│   │   ├── emotion.py             # Emotional tone
│   │   ├── temporal.py            # ⭐ Temporal dynamics
│   │   ├── baseline.py            # ⭐ Baseline adaptation
│   │   ├── coherence.py           # ⭐ Multi-modal coherence
│   │   └── cognitive.py           # ⭐ Cognitive load
│   ├── scoring/
│   │   ├── normalizer.py
│   │   ├── weighted_scorer.py
│   │   └── adaptive_scorer.py     # ⭐ Baseline-adaptive
│   ├── prediction/
│   │   └── listener_model.py      # ⭐ Listener prediction
│   └── utils/
│       ├── windowing.py           # ⭐ Temporal windowing
│       ├── cost_tracker.py
│       └── cache_manager.py
├── frontend/
│   └── (React components)
└── README.md
```

---

## 10. Dependencies

| Category | Tools |
|----------|-------|
| **Speech-to-Text** | OpenAI Whisper (local) |
| **Audio Analysis** | librosa, parselmouth |
| **Voice Activity Detection** | webrtcvad |
| **NLP** | spaCy, NLTK |
| **Grammar** | LanguageTool |
| **Emotion** | HuggingFace transformers |
| **Backend** | FastAPI, uvicorn |
| **Frontend** | React |

**Note: All tools are open-source and run locally. No LLM API required.**

---

## 11. API Response Schema

```json
{
  "composite_score": 78.5,
  "category_scores": {
    "vocal_delivery": 82.3,
    "fluency": 75.0,
    "clarity": 88.2,
    "language": 72.5,
    "prosody": 76.8,
    "temporal_dynamics": 80.1,
    "coherence": 74.5,
    "cognitive_load": 85.0
  },
  "novel_metrics": {
    "confidence_trajectory": "increasing",
    "engagement_arc_score": 78.0,
    "fluency_warmup_index": 15.0,
    "fatigue_detection_score": 88.0,
    "sentiment_prosody_coherence": 72.5,
    "emphasis_alignment": 68.0,
    "pause_semantic_sync": 81.0,
    "cognitive_strain_index": 22.5
  },
  "listener_predictions": {
    "predicted_comprehension": 82.0,
    "predicted_engagement": 75.5,
    "predicted_trust": 79.0,
    "attention_sustainability_minutes": 8.5
  },
  "baseline_comparison": {
    "speech_rate_vs_baseline": "+8%",
    "confidence_vs_baseline": "improved"
  },
  "struggle_points": [
    {"time": "2:15-2:30", "cause": "complex_content"}
  ]
}
```

---

## 12. Technical Constraints

| Constraint | Specification |
|------------|---------------|
| Audio Format | WAV, MP3, M4A |
| Max Duration | 10 minutes |
| Language | English |
| Processing | Post-recording (not real-time) |
| Hardware | CPU-compatible (no GPU required) |
| LLM Dependency | None (all local models) |

---

## 13. Summary of Novel Contributions

1. **Temporal Dynamics Analysis** - First system to analyze metric trajectories over time
2. **Speaker Baseline Adaptation** - Personalized benchmarks per individual speaker
3. **Multi-Modal Coherence Scoring** - Novel alignment measure between prosody and semantics
4. **Cognitive Load Estimation** - First metric for speaker mental strain from acoustic cues
5. **Listener-Centric Prediction** - Shift from speaker-focused to audience-focused evaluation
6. **Integrated Framework** - Unified system combining all innovations with weighted scoring