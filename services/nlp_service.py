import re
import string
from collections import Counter


# ── Stopwords — built in, no NLTK needed ──────────────────────────────────────
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "this", "that", "these",
    "those", "it", "its", "as", "if", "then", "than", "so", "not", "no",
    "up", "out", "about", "into", "over", "after", "before", "between",
    "through", "during", "he", "she", "they", "we", "you", "i", "me",
    "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "what", "which", "who", "when", "where", "how", "why", "all", "any",
    "both", "each", "more", "most", "other", "some", "such", "only",
    "own", "same", "too", "very", "just", "now", "get", "got", "also",
    "says", "said", "new", "one", "two", "three", "four", "five", "six",
    "per", "amid", "amid", "while", "after", "says", "since", "still",
    "company", "stock", "share", "shares", "market", "year", "week",
    "month", "quarter", "today", "reuters", "bloomberg", "inc", "corp",
    "ltd", "co", "us", "vs", "first", "last", "could", "would", "will",
}


def _tokenize(text: str) -> list:
    """
    Simple pure-Python tokenizer.
    Lowercases, removes punctuation, splits on whitespace.
    No external dependencies needed.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


# ── Feature 1 — Keyword Frequency ─────────────────────────────────────────────
def extract_keywords(headlines: list, top_n: int = 15) -> list:
    """
    Extract top N keywords from headlines using pure Python tokenization.
    Removes stopwords and short words. Returns (word, count) tuples.
    """
    tokens = []
    for headline in headlines:
        words = _tokenize(headline)
        tokens.extend([
            w for w in words
            if len(w) > 3
            and w not in STOPWORDS
        ])

    freq = Counter(tokens)
    return freq.most_common(top_n)


# ── Feature 2 — NLP Pipeline Steps ────────────────────────────────────────────
def get_pipeline_steps(headline: str) -> dict:
    """
    Show step-by-step NLP preprocessing pipeline for one headline.
    Uses pure Python — no NLTK required.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    # Step 1 — raw
    raw = headline

    # Step 2 — lowercase
    lowered = headline.lower()

    # Step 3 — remove punctuation
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))

    # Step 4 — tokenize
    tokens = no_punct.split()

    # Step 5 — remove stopwords
    cleaned = [w for w in tokens if w.isalpha() and len(w) > 2 and w not in STOPWORDS]

    # Step 6 — simple POS approximation using word patterns
    pos_tagged = []
    for word in cleaned[:8]:
        if word.endswith("ing"):
            tag = "VBG"
        elif word.endswith("ed"):
            tag = "VBD"
        elif word.endswith("ly"):
            tag = "RB"
        elif word.endswith("tion") or word.endswith("ment") or word.endswith("ness"):
            tag = "NN"
        elif word.endswith("al") or word.endswith("ous") or word.endswith("ive"):
            tag = "JJ"
        elif word[0].isupper():
            tag = "NNP"
        else:
            tag = "NN"
        pos_tagged.append(f"{word}/{tag}")

    # Step 7 — VADER sentiment
    analyzer = SentimentIntensityAnalyzer()
    scores   = analyzer.polarity_scores(headline)
    compound = scores["compound"]

    if compound >= 0.05:
        sentiment_label = "Positive"
    elif compound <= -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    return {
        "raw":        raw,
        "lowercased": lowered,
        "tokens":     tokens[:12],
        "cleaned":    cleaned[:12],
        "pos_tags":   pos_tagged,
        "compound":   compound,
        "sentiment":  sentiment_label,
        "all_scores": scores,
    }


# ── Feature 3 — Confidence Scores ─────────────────────────────────────────────
def format_results_with_confidence(results: list) -> list:
    """
    Enrich sentiment results with formatted confidence percentage.
    Works with both FinBERT and VADER output formats.
    """
    enriched = []
    for r in results:
        label      = r.get("label", "neutral")
        all_s      = r.get("all",   {})
        confidence = all_s.get(label, r.get("score", 0.0))

        enriched.append({
            **r,
            "confidence":     confidence,
            "confidence_pct": f"{confidence:.0%}",
        })
    return enriched


# ── Feature 4 — VADER vs FinBERT Comparison ───────────────────────────────────
def compare_models(headlines: list, finbert_results: list) -> list:
    """
    Run VADER on the same headlines FinBERT already processed.
    Returns comparison list with agreement flags.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer   = SentimentIntensityAnalyzer()
    comparison = []

    for i, headline in enumerate(headlines[:10]):
        scores   = analyzer.polarity_scores(headline)
        compound = scores["compound"]

        if compound >= 0.05:
            vader_label = "positive"
        elif compound <= -0.05:
            vader_label = "negative"
        else:
            vader_label = "neutral"

        finbert_label = finbert_results[i]["label"] if i < len(finbert_results) else "neutral"

        comparison.append({
            "headline":    headline[:90] + "..." if len(headline) > 90 else headline,
            "vader":       vader_label,
            "finbert":     finbert_label,
            "vader_score": round(compound, 3),
            "agree":       vader_label == finbert_label,
        })

    return comparison

# ── Word Cloud ─────────────────────────────────────────────────────────────────
def generate_wordcloud(headlines: list) -> "matplotlib.figure.Figure":
    """
    Generate a word cloud from news headlines.
    Returns a matplotlib figure ready for st.pyplot().
    """
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    # Join all headlines into one text block
    text = " ".join(headlines)

    # Remove short words and stopwords using existing STOPWORDS set
    words   = _tokenize(text)
    cleaned = " ".join([
        w for w in words
        if len(w) > 3 and w not in STOPWORDS
    ])

    if not cleaned.strip():
        return None

    wc = WordCloud(
        width=800,
        height=400,
        background_color="black",
        colormap="Blues",
        max_words=60,
        collocations=False,
        prefer_horizontal=0.85,
    )
    wc.generate(cleaned)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor("black")
    plt.tight_layout(pad=0)

    return fig