"""Text cleaning utilities for tweets."""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# Keep negations: they flip sentiment ("not good").
_KEEP = {"not", "no", "nor", "never", "n't"}
try:
    STOP_WORDS = set(stopwords.words("english")) - _KEEP
except LookupError:
    STOP_WORDS = set()
_lemmatizer = WordNetLemmatizer()

EMOJI_MAP = {
    ":)": " happy ", ":-)": " happy ", ":D": " happy ", "<3": " love ",
    ":(": " sad ", ":-(": " sad ", ":'(": " sad ",
    "😀": " happy ", "😊": " happy ", "😍": " love ", "😂": " laugh ",
    "😢": " sad ", "😭": " sad ", "😡": " angry ", "👍": " good ", "👎": " bad ",
}


def _lemmatize(word: str) -> str:
    try:
        return _lemmatizer.lemmatize(word)
    except LookupError:
        return word


def normalize_for_transformer(text: str) -> str:
    """Light cleaning for BERT/RoBERTa (they need natural text)."""
    text = re.sub(r"@\w+", "@user", str(text))
    text = re.sub(r"http\S+|www\.\S+", "http", text)
    return text.strip()


def clean_tweet(text: str) -> str:
    """Heavy cleaning for classical ML / LSTM models."""
    text = str(text)
    for k, v in EMOJI_MAP.items():
        text = text.replace(k, v)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"@\w+", " ", text)               # mentions
    text = re.sub(r"\brt\b", " ", text)             # retweet marker
    text = text.replace("#", " ")                   # keep hashtag word
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)      # sooooo -> soo
    text = re.sub(r"[^a-z\s']", " ", text)          # punctuation/numbers
    tokens = [_lemmatize(w) for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)
