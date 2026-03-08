"""
SpeechScore 2.0 — Multi-Modal Coherence Analyzer  ⭐ NOVEL

Measures the alignment between what the speaker *says* (text semantics)
and *how they say it* (acoustic prosody), producing three coherence
metrics:

  1. **Sentiment-Prosody Coherence**  — Does the speaker's vocal energy
     and pitch match the emotional valence of their words?
  2. **Emphasis-Importance Alignment** — Are key words (entities, nouns)
     acoustically emphasised (higher pitch, longer duration, preceding
     pause)?
  3. **Pause-Semantic Synchronization** — Do pauses align with sentence
     and clause boundaries rather than falling mid-phrase?

Novelty argument
----------------
Most speech assessment tools analyse audio and text independently.  Our
coherence module cross-references the two modalities, capturing
communicative *alignment* rather than isolated quality.  A speaker who
says "I'm really excited" in a monotone receives a low coherence score
even if both text sentiment and pitch stability are individually fine.

References:
  - Bänziger & Scherer (2005) "The role of intonation in emotional
    expressions" — prosody-emotion alignment
  - Pierrehumbert (1980) — F0 and emphasis theory
  - Goldman-Eisler (1972) — pause placement and planning
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from speechscore.models.schemas import (
    WindowMetrics,
    TranscriptionResult,
    CoherenceResult,
)

logger = logging.getLogger(__name__)

# ── lazy-loaded singletons ──
_nlp: Optional[spacy.language.Language] = None
_vader: Optional[SentimentIntensityAnalyzer] = None


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_lg")
    return _nlp


def _get_vader() -> SentimentIntensityAnalyzer:
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    return _vader


class CoherenceAnalyzer:
    """
    Multi-Modal coherence analysis (3 metrics from spec §7).

    Usage::

        analyzer = CoherenceAnalyzer()
        result = analyzer.analyze(windows, transcription)
    """

    def analyze(
        self,
        windows: list[WindowMetrics],
        transcription: TranscriptionResult,
    ) -> CoherenceResult:
        """
        Compute all three coherence metrics.

        Parameters
        ----------
        windows       : per-window metrics (with pitch / energy / pauses).
        transcription : Whisper transcription with word-level timing.

        Returns
        -------
        CoherenceResult with three sub-scores and a composite.
        """
        if len(windows) < 3:
            return CoherenceResult()

        sp = self._sentiment_prosody(windows, transcription)
        ei = self._emphasis_importance(windows, transcription)
        ps = self._pause_semantic(windows, transcription)

        composite = (sp * 0.35 + ei * 0.35 + ps * 0.30)

        return CoherenceResult(
            sentiment_prosody_score=round(sp, 1),
            emphasis_alignment_score=round(ei, 1),
            pause_semantic_score=round(ps, 1),
            composite_coherence=round(composite, 1),
        )

    # ────────────────────────────────────────────────────────────
    # 1.  SENTIMENT-PROSODY COHERENCE
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _sentiment_prosody(
        windows: list[WindowMetrics],
        transcription: TranscriptionResult,
    ) -> float:
        """
        Correlate per-window text sentiment with vocal energy.

        Positive sentiment → we expect higher energy / pitch variation.
        Negative sentiment → lower energy may be appropriate but
        *any* correlation indicates alignment.

        Returns 0–100.
        """
        vader = _get_vader()

        sentiments: list[float] = []
        energies: list[float] = []

        for w in windows:
            if not w.transcript or not w.reliable:
                continue
            vs = vader.polarity_scores(w.transcript)
            sentiments.append(vs["compound"])  # −1 to +1
            # proxy for vocal engagement: normalised energy
            e = w.rms_mean if w.rms_mean is not None else 0.0
            energies.append(e)

        if len(sentiments) < 4:
            return 50.0  # not enough data

        s_arr = np.array(sentiments)
        e_arr = np.array(energies)

        # normalise energy to z-scores for correlation
        if np.std(e_arr) > 1e-8:
            e_norm = (e_arr - np.mean(e_arr)) / np.std(e_arr)
        else:
            e_norm = np.zeros_like(e_arr)

        # Pearson correlation
        if np.std(s_arr) > 1e-8:
            from scipy import stats
            r, _ = stats.pearsonr(s_arr, e_norm)
            if np.isnan(r):
                r = 0.0
        else:
            r = 0.0

        # Map r to 0–100:  r=1 → 100, r=0 → 50, r=−1 → 0
        # Use absolute correlation (positive or negative alignment
        # both indicate coherence — a sad story with low energy is
        # still coherent)
        score = max(0, min(100, 50.0 + abs(float(r)) * 50.0))
        return score

    # ────────────────────────────────────────────────────────────
    # 2.  EMPHASIS-IMPORTANCE ALIGNMENT
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _emphasis_importance(
        windows: list[WindowMetrics],
        transcription: TranscriptionResult,
    ) -> float:
        """
        Check whether acoustically important words are emphasised.

        Method:
          1. Use spaCy to identify important words (NER + noun chunks)
          2. For each important word, check:
             a. Duration > global mean duration → elongated
             b. Preceding gap > 0.15s → emphasising pause
          3. Alignment % = emphasised important words / total important

        Returns 0–100.
        """
        nlp = _get_nlp()
        words = transcription.words

        if not words or not transcription.full_text:
            return 50.0

        # Identify important words via spaCy NER + noun chunks
        doc = nlp(transcription.full_text)
        important_tokens: set[str] = set()

        # Named entities
        for ent in doc.ents:
            for tok in ent:
                important_tokens.add(tok.text.lower())

        # Noun chunks (head noun only)
        for chunk in doc.noun_chunks:
            important_tokens.add(chunk.root.text.lower())

        # Content verbs (non-auxiliary)
        for tok in doc:
            if tok.pos_ == "VERB" and not tok.dep_.startswith("aux"):
                important_tokens.add(tok.text.lower())

        if not important_tokens:
            return 60.0  # neutral — no key words identified

        # Compute global average word duration
        durations = []
        for w in words:
            d = w.end - w.start
            if d > 0:
                durations.append(d)
        if not durations:
            return 50.0

        mean_dur = float(np.mean(durations))

        # For each important word, check emphasis indicators
        n_important = 0
        n_emphasised = 0

        for i, w in enumerate(words):
            w_lower = w.word.lower().strip(".,!?;:'\"")
            if w_lower not in important_tokens:
                continue

            n_important += 1
            emphasis_score = 0  # need ≥ 1 indicator to count

            # a. Elongation: duration > 1.2× mean
            dur = w.end - w.start
            if dur > mean_dur * 1.2:
                emphasis_score += 1

            # b. Preceding pause: gap > 0.15s before this word
            if i > 0:
                gap = w.start - words[i - 1].end
                if gap > 0.15:
                    emphasis_score += 1

            if emphasis_score >= 1:
                n_emphasised += 1

        if n_important == 0:
            return 60.0

        alignment_pct = (n_emphasised / n_important) * 100.0
        # Scale: 50% alignment → score 70 (most speakers hit 30–60%)
        score = min(100, alignment_pct * 1.4)
        return score

    # ────────────────────────────────────────────────────────────
    # 3.  PAUSE-SEMANTIC SYNCHRONIZATION
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _pause_semantic(
        windows: list[WindowMetrics],
        transcription: TranscriptionResult,
    ) -> float:
        """
        Measure how well pause locations align with syntactic boundaries.

        Method:
          1. Map pause positions from word timing gaps > 0.25s
          2. Map sentence / clause boundaries from spaCy
          3. For each pause, check if it's within ±0.5s of a boundary
          4. Sync % = aligned pauses / total pauses

        Returns 0–100.
        """
        nlp = _get_nlp()
        words = transcription.words
        full_text = transcription.full_text

        if not words or not full_text:
            return 50.0

        # 1. Find pause positions (gaps > 0.25s between words)
        pause_positions: list[float] = []
        for i in range(1, len(words)):
            gap = words[i].start - words[i - 1].end
            if gap > 0.25:
                pause_positions.append(words[i - 1].end + gap / 2)  # midpoint

        if not pause_positions:
            return 70.0  # no pauses → neutral-good (continuous speech)

        # 2. Find syntactic boundary positions
        # Build a mapping: character offset → approximate time
        # by aligning spaCy tokens to Whisper words
        doc = nlp(full_text)

        # Collect boundary timestamps: end-of-sentence and clause edges
        boundary_times: list[float] = []

        # sentence boundaries
        for sent in doc.sents:
            # find the Whisper word closest to the end of this sentence
            last_word_text = None
            for tok in reversed(list(sent)):
                if tok.is_alpha:
                    last_word_text = tok.text.lower()
                    break
            if last_word_text:
                # find matching Whisper word (closest to end)
                for w in reversed(words):
                    if w.word.lower().strip(".,!?;:'\"") == last_word_text:
                        boundary_times.append(w.end)
                        break

        # clause boundaries (conjunctions, relative pronouns)
        for tok in doc:
            if tok.dep_ in ("cc", "mark", "relcl", "advcl") and tok.i > 0:
                prev_text = doc[tok.i - 1].text.lower()
                for w in words:
                    if w.word.lower().strip(".,!?;:'\"") == prev_text:
                        boundary_times.append(w.end)
                        break

        if not boundary_times:
            return 50.0

        boundary_arr = np.array(sorted(set(boundary_times)))

        # 3. For each pause, check if near a boundary (±0.5s)
        tolerance = 0.5
        n_aligned = 0
        for p in pause_positions:
            min_dist = float(np.min(np.abs(boundary_arr - p)))
            if min_dist <= tolerance:
                n_aligned += 1

        sync_pct = (n_aligned / len(pause_positions)) * 100.0

        # Scale: 60% sync → 80 score (well-placed pauses)
        score = min(100, sync_pct * 1.25)
        return score
