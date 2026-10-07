"""Text preprocessing module for Fake Product Review Detection.

This module provides clean, simple, and explainable NLP text cleaning functions.
The exact same cleaning pipeline is applied during both training and inference.
"""

import re
import string
import nltk

# Ensure stopwords are available; download safely if missing
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    # Safe fallback list of common English stopwords in case network is offline
    STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
        "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
        "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
        "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
        "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
        "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
        "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
        "yourself", "yourselves"
    }


def clean_text(text: str) -> str:
    """Preprocesses a single review string.

    Steps:
    1. Handle non-string / missing values
    2. Convert to lowercase
    3. Remove URLs
    4. Remove HTML tags / mentions
    5. Remove punctuation and special symbols
    6. Remove stopwords
    7. Normalize whitespace

    Args:
        text: Raw review string

    Returns:
        Cleaned, token-normalized string
    """
    if text is None or not isinstance(text, str):
        return ""

    # 1. Lowercase conversion
    text = text.lower()

    # 2. Remove URLs (http, https, www)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # 4. Remove punctuation and non-alphabetic characters
    # Keep words only (letters and whitespace)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # 5. Split tokens and remove stopwords
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 1]

    # 6. Reconstruct cleaned text
    return " ".join(filtered_tokens)


def extract_review_heuristics(raw_text: str) -> dict:
    """Extracts explainable linguistic heuristics from raw review text.

    These indicators serve as supporting information in the UI and Viva
    demonstration (not as absolute proof).

    Args:
        raw_text: Original raw review text entered by user

    Returns:
        Dictionary with character count, word count, exclamation count,
        all-caps count, and detected hype phrases.
    """
    if not raw_text or not isinstance(raw_text, str):
        return {
            "char_count": 0,
            "word_count": 0,
            "exclamation_count": 0,
            "uppercase_words": 0,
            "suspicious_patterns": [],
            "suspicion_heuristic_score": 0.0
        }

    char_count = len(raw_text)
    words = raw_text.split()
    word_count = len(words)
    exclamation_count = raw_text.count('!') + raw_text.count('?')

    # Words with 2+ characters that are fully capitalized (e.g. MUST, BEST, WOW)
    uppercase_words = sum(1 for w in words if len(w) >= 2 and w.isupper())

    # Common promotional / high-hype expressions frequently noted in NLP review studies
    hype_keywords = [
        "must buy", "best product ever", "100%", "miracle", "waste of money",
        "do not buy", "life changing", "guaranteed", "buy this now", "five stars"
    ]
    lower_raw = raw_text.lower()
    matched_patterns = [kw for kw in hype_keywords if kw in lower_raw]

    # Heuristic score for visual indicator (range 0 to 100)
    score = 0
    if exclamation_count >= 3:
        score += 25
    if uppercase_words >= 2:
        score += 25
    if matched_patterns:
        score += min(len(matched_patterns) * 20, 35)
    if word_count < 6 or word_count > 250:
        score += 15

    score = min(score, 100)

    return {
        "char_count": char_count,
        "word_count": word_count,
        "exclamation_count": exclamation_count,
        "uppercase_words": uppercase_words,
        "suspicious_patterns": matched_patterns,
        "suspicion_heuristic_score": float(score)
    }
