"""Deterministic NLP features (MVP plan component 3a, milestone M3).

Turns raw essay text into measurable facts with ``source="nlp"``. Every feature
is computed without an LLM so ``rule_only`` mode has a real signal. Optional
dependencies degrade gracefully: if a spell checker or readability library is
unavailable the corresponding feature is 0 and a warning is logged.
"""

from __future__ import annotations

import re

from .logging_setup import get_logger
from .schemas import Fact

log = get_logger("ees.features")

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*")

CONNECTIVES = {
    "however", "therefore", "moreover", "furthermore", "additionally",
    "consequently", "nevertheless", "nonetheless", "thus", "hence",
    "although", "though", "whereas", "because", "since", "despite",
    "meanwhile", "similarly", "likewise", "conversely", "finally",
    "firstly", "secondly", "thirdly", "lastly", "overall", "in conclusion",
    "for example", "for instance", "on the other hand", "in addition",
    "as a result", "even though", "in contrast",
}

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?previous instructions",
    r"disregard (all |any )?(the )?(previous|above) (instructions|prompt)",
    r"ignore the (above|essay|prompt)",
    r"you are now",
    r"system\s*:",
    r"assistant\s*:",
    r"give (me|this essay) (a )?(full|max|maximum|top) (mark|score|grade)",
    r"award (me|this) (a )?(full|max|maximum)",
    r"do not (grade|mark|score) this",
]

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "as", "is", "are", "was", "were", "be", "been", "being", "it",
    "this", "that", "these", "those", "they", "their", "there", "than", "then",
    "so", "such", "not", "no", "do", "does", "did", "have", "has", "had",
    "will", "would", "can", "could", "should", "may", "might", "must", "at",
    "by", "from", "we", "you", "your", "our", "i", "he", "she", "his", "her",
    "its", "them", "us", "more", "most", "some", "any", "all", "each", "very",
    "which", "who", "whom", "what", "when", "where", "why", "how",
}


def tokenize(text: str) -> list[str]:
    """Return alphabetic word tokens."""

    return TOKEN_RE.findall(text)


def split_sentences(text: str) -> list[str]:
    """Split into sentences, dropping empties."""

    parts = [s.strip() for s in SENTENCE_RE.findall(text)]
    return [s for s in parts if s]


def split_paragraphs(text: str) -> list[str]:
    """Split into paragraphs on blank lines, falling back to single newlines."""

    blocks = [b.strip() for b in re.split(r"\n\s*\n", text.strip())]
    blocks = [b for b in blocks if b]
    if len(blocks) <= 1:
        blocks = [b.strip() for b in text.strip().split("\n") if b.strip()]
    return blocks


def count_connectives(text: str) -> int:
    """Count cohesive devices (words and multiword phrases, no double counting)."""

    lowered = " " + text.lower() + " "
    count = 0
    claimed: list[tuple[int, int]] = []
    for phrase in sorted(CONNECTIVES, key=len, reverse=True):
        start = 0
        while True:
            idx = lowered.find(phrase, start)
            if idx == -1:
                break
            end = idx + len(phrase)
            # word-boundary check
            left_ok = idx == 0 or not lowered[idx - 1].isalnum()
            right_ok = end >= len(lowered) or not lowered[end].isalnum()
            overlap = any(not (end <= s or idx >= e) for s, e in claimed)
            if left_ok and right_ok and not overlap:
                claimed.append((idx, end))
                count += 1
            start = idx + 1
    return count


def readability(text: str) -> float:
    """Flesch Reading Ease, via textstat if available else 0.0."""

    try:
        import textstat  # type: ignore

        return float(textstat.flesch_reading_ease(text))
    except Exception:  # pragma: no cover - optional dependency
        log.warning("textstat unavailable; readability set to 0.0")
        return 0.0


def _spell_checker():
    try:
        from spellchecker import SpellChecker  # type: ignore

        return SpellChecker(language="en")
    except Exception:  # pragma: no cover - optional dependency
        return None


def spelling_error_rate(text: str, words: list[str]) -> float:
    """Fraction of tokens not found in the dictionary (0.0 if unavailable)."""

    if not words:
        return 0.0
    checker = _spell_checker()
    if checker is None:  # pragma: no cover - optional dependency
        log.warning("spellchecker unavailable; spelling_error_rate set to 0.0")
        return 0.0
    # Ignore likely proper nouns and split hyphenated/apostrophe forms so that
    # "hand-me-downs" and "students'" are not counted as misspellings.
    candidates: list[str] = []
    for w in words:
        if not w or w[0].isupper():
            continue
        for part in re.split(r"[-']", w.lower()):
            if len(part) >= 3:
                candidates.append(part)
    if not candidates:
        return 0.0
    unknown = checker.unknown(candidates)
    return round(len(unknown) / len(words), 4)


def detect_injection(text: str) -> bool:
    """Detect text that looks like instructions aimed at the LLM."""

    lowered = text.lower()
    return any(re.search(p, lowered) for p in INJECTION_PATTERNS)


def prompt_keyword_overlap(text: str, prompt_text: str, key_point_texts: list[str]) -> float:
    """Proportion of prompt/key-point content words that appear in the essay."""

    essay_tokens = {w.lower() for w in tokenize(text)}
    ref_tokens: set[str] = set()
    for src in [prompt_text, *key_point_texts]:
        ref_tokens |= {
            w.lower() for w in tokenize(src) if w.lower() not in STOPWORDS and len(w) > 2
        }
    if not ref_tokens:
        return 0.0
    hit = sum(1 for t in ref_tokens if t in essay_tokens)
    return round(hit / len(ref_tokens), 4)


def extract_features(
    text: str,
    prompt_text: str = "",
    key_point_texts: list[str] | None = None,
) -> list[Fact]:
    """Compute all deterministic features and return them as :class:`Fact` objects."""

    key_point_texts = key_point_texts or []
    words = tokenize(text)
    sentences = split_sentences(text)
    paragraphs = split_paragraphs(text)
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    lower = [w.lower() for w in words]
    unique = len(set(lower))

    features: dict[str, object] = {
        "word_count": word_count,
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "avg_sentence_length": round(word_count / sentence_count, 2),
        "lexical_diversity": round(unique / word_count, 4) if word_count else 0.0,
        "readability": round(readability(text), 2),
        "connective_count": count_connectives(text),
        "spelling_error_rate": spelling_error_rate(text, words),
        "grammar_error_count": 0,
        "prompt_keyword_overlap": prompt_keyword_overlap(text, prompt_text, key_point_texts),
        "injection_detected": detect_injection(text),
    }
    return [Fact(name=k, value=v, source="nlp") for k, v in features.items()]
