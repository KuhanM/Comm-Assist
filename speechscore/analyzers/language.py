"""
SpeechScore 2.0 — Language Analyzer

Extracts **Language Quality** metrics on the *full transcript*:
  1. Grammar Score           — 1 − (errors / sentences)  via LanguageTool
  2. Vocabulary Richness     — lemmatised Type-Token Ratio (TTR)  via spaCy
  3. Sentence Complexity     — average clauses per sentence  via spaCy

These are computed once over the entire text rather than per-window
because grammar, vocabulary, and clause detection need broader context
to be meaningful.

LanguageTool is optional (requires Java).  If absent, grammar_score
is returned as ``None`` and the other two metrics still work.
"""

from __future__ import annotations

import logging
from typing import Optional

from speechscore.config.settings import SpeechScoreConfig
from speechscore.models.schemas import LanguageMetrics

logger = logging.getLogger(__name__)

# spaCy dependency labels that indicate subordinate/embedded clauses
_CLAUSE_DEPS = frozenset({
    "ccomp",       # clausal complement
    "xcomp",       # open clausal complement
    "advcl",       # adverbial clause modifier
    "acl",         # clausal modifier of noun (adjectival clause)
    "relcl",       # relative clause modifier
    "csubj",       # clausal subject
    "csubjpass",   # clausal passive subject
})


class LanguageAnalyzer:
    """Linguistic feature extractor (full-transcript scope)."""

    def __init__(self, config: SpeechScoreConfig) -> None:
        self.config = config
        self._nlp = None
        self._grammar_tool = None
        self._grammar_available = True

    # ── lazy loaders ─────────────────────────────────────────────

    def _load_spacy(self):
        if self._nlp is not None:
            return
        import spacy

        model = self.config.language.spacy_model
        try:
            self._nlp = spacy.load(model)
            logger.info("Loaded spaCy model: %s", model)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found — falling back to en_core_web_sm",
                model,
            )
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "No spaCy English model found. "
                    "Run: python -m spacy download en_core_web_lg"
                )

    def _load_grammar_tool(self):
        if self._grammar_tool is not None or not self._grammar_available:
            return
        try:
            import language_tool_python

            self._grammar_tool = language_tool_python.LanguageTool(
                self.config.language.language_tool_lang
            )
            logger.info("LanguageTool loaded")
        except Exception as exc:
            logger.warning(
                "LanguageTool unavailable (Java required): %s — "
                "grammar scoring will be skipped.",
                exc,
            )
            self._grammar_available = False

    # ── public API ───────────────────────────────────────────────

    def analyze(self, transcript: str) -> LanguageMetrics:
        """
        Compute language quality metrics on the full transcript.

        Parameters
        ----------
        transcript : the complete speech-to-text output.

        Returns
        -------
        LanguageMetrics with grammar, vocabulary, and complexity.
        """
        if not transcript or not transcript.strip():
            return LanguageMetrics()

        self._load_spacy()
        doc = self._nlp(transcript)

        grammar_score, grammar_errors = self._grammar_score(transcript, doc)
        ttr, unique, total = self._vocabulary_richness(doc)
        complexity, n_sent, n_clause = self._sentence_complexity(doc)

        return LanguageMetrics(
            grammar_score=grammar_score,
            grammar_error_count=grammar_errors,
            vocabulary_richness=ttr,
            unique_word_count=unique,
            total_word_count=total,
            sentence_complexity=complexity,
            sentence_count=n_sent,
            clause_count=n_clause,
        )

    # ── grammar ──────────────────────────────────────────────────

    def _grammar_score(self, text: str, doc) -> tuple[Optional[float], int]:
        """
        Score = 1 − (relevant_errors / sentences), clipped to [0, 1].

        Only ``grammar``, ``typographical``, and ``misspelling`` issues
        are counted — stylistic suggestions are ignored to avoid
        penalising informal speech transcripts.
        """
        self._load_grammar_tool()

        if not self._grammar_available or self._grammar_tool is None:
            return None, 0

        try:
            matches = self._grammar_tool.check(text)
            error_types = {"grammar", "typographical", "misspelling"}
            errors = [
                m for m in matches if m.rule_issue_type in error_types
            ]
            n_sent = max(1, len(list(doc.sents)))
            score = max(0.0, 1.0 - len(errors) / n_sent)
            return round(score, 4), len(errors)
        except Exception as exc:
            logger.warning("Grammar check failed: %s", exc)
            return None, 0

    # ── vocabulary richness ──────────────────────────────────────

    @staticmethod
    def _vocabulary_richness(doc) -> tuple[float, int, int]:
        """
        Lemmatised Type-Token Ratio.

        Only alphabetic tokens are included (digits, punctuation,
        and whitespace are excluded).
        """
        tokens = [
            tok.lemma_.lower()
            for tok in doc
            if tok.is_alpha
        ]
        if not tokens:
            return 0.0, 0, 0

        unique = set(tokens)
        ttr = len(unique) / len(tokens)
        return round(ttr, 4), len(unique), len(tokens)

    # ── sentence complexity ──────────────────────────────────────

    @staticmethod
    def _sentence_complexity(doc) -> tuple[float, int, int]:
        """
        Average clauses per sentence.

        Each sentence starts with 1 clause (the root / main clause).
        Extra clauses are counted via dependency labels in _CLAUSE_DEPS.
        """
        sentences = list(doc.sents)
        if not sentences:
            return 0.0, 0, 0

        total_clauses = 0
        for sent in sentences:
            clauses = 1  # main clause
            for tok in sent:
                if tok.dep_ in _CLAUSE_DEPS:
                    clauses += 1
            total_clauses += clauses

        avg = total_clauses / len(sentences)
        return round(avg, 4), len(sentences), total_clauses

    # ── cleanup ──────────────────────────────────────────────────

    def close(self):
        """Shut down the LanguageTool background server."""
        if self._grammar_tool is not None:
            try:
                self._grammar_tool.close()
            except Exception:
                pass
