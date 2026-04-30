from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_PATTERN = re.compile(r"[A-Za-z]+")

# A few corpus-specific discussion words tend to dominate several topics.
DEFAULT_TOPIC_STOP_WORDS = ENGLISH_STOP_WORDS.union(
    {
        "need",
        "question",
        "questions",
        "new",
        "thanks",
        "thank",
        "help",
        "looking",
        "look",
        "used",
    }
)

try:
    from nltk.stem import WordNetLemmatizer

    _WORDNET_LEMMATIZER = WordNetLemmatizer()
    _HAS_WORDNET_LEMMATIZER = True
except Exception:  # pragma: no cover - optional dependency
    _WORDNET_LEMMATIZER = None
    _HAS_WORDNET_LEMMATIZER = False


def _simple_rule_based_lemma(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


@lru_cache(maxsize=50000)
def lemmatize_token(token: str) -> str:
    token = token.lower()

    if _HAS_WORDNET_LEMMATIZER:
        try:
            lemma = _WORDNET_LEMMATIZER.lemmatize(token, pos="n")
            lemma = _WORDNET_LEMMATIZER.lemmatize(lemma, pos="v")
            return lemma
        except LookupError:
            # WordNet data may be missing locally; fall back gracefully.
            pass

    return _simple_rule_based_lemma(token)


def make_topic_analyzer(extra_stop_words: Iterable[str] | None = None):
    stop_words = set(DEFAULT_TOPIC_STOP_WORDS)
    if extra_stop_words:
        stop_words.update(word.lower() for word in extra_stop_words)

    def analyzer(text: str) -> list[str]:
        tokens: list[str] = []

        for raw_token in TOKEN_PATTERN.findall(text.lower()):
            if len(raw_token) < 2:
                continue

            lemma = lemmatize_token(raw_token)
            if len(lemma) < 2 or lemma in stop_words:
                continue

            tokens.append(lemma)

        return tokens

    return analyzer
