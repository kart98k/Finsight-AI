import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_finbert_pipeline = None
_finbert_available = None


def _load_finbert():
    global _finbert_pipeline, _finbert_available

    if _finbert_available is False:
        return None

    if _finbert_pipeline is not None:
        return _finbert_pipeline

    try:
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
            truncation=True,
            max_length=512,
        )
        _finbert_available = True
        print("[FinBERT] Loaded successfully.")
        return _finbert_pipeline
    except Exception as e:
        _finbert_available = False
        print(f"[FinBERT] Failed to load: {e}. Falling back to VADER.")
        return None


def analyze_with_vader(texts: list) -> list:
    """
    Rule-based sentiment using VADER.
    Fast, no model download needed.
    Returns list of dicts with: text, compound, label, scores
    """
    analyzer = SentimentIntensityAnalyzer()
    results  = []

    for text in texts:
        scores   = analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        results.append({
            "text":     text,
            "compound": compound,
            "label":    label,
            "scores":   scores,
            "all": {
                "positive": max(0.0, compound),
                "negative": max(0.0, -compound),
                "neutral":  1.0 - abs(compound),
            },
            "score": abs(compound),
        })

    return results


def analyze_with_finbert(texts: list) -> list:
    """
    Domain-specific financial sentiment using FinBERT.
    Falls back to VADER if FinBERT is unavailable.
    """
    pipe = _load_finbert()

    # Fallback to VADER if FinBERT not available
    if pipe is None:
        print("[Sentiment] Using VADER fallback.")
        return analyze_with_vader(texts)

    results = []
    for text in texts:
        try:
            output = pipe(text[:512])
            scores = output[0] if isinstance(output[0], list) else output
            best   = max(scores, key=lambda x: x["score"])

            # Normalize label
            label = best["label"].lower()
            if label not in ("positive", "negative", "neutral"):
                label = "neutral"

            all_scores = {
                item["label"].lower(): item["score"]
                for item in scores
            }

            results.append({
                "text":  text,
                "label": label,
                "score": best["score"],
                "all":   all_scores,
            })

        except Exception as e:
            print(f"[FinBERT] Error on text: {e}. Using VADER for this item.")
            vader_result = analyze_with_vader([text])
            results.append(vader_result[0])

    return results


def aggregate_sentiment(results: list) -> dict:
    """
    Aggregate sentiment results into summary stats.
    Works with both FinBERT and VADER output.
    """
    if not results:
        return {
            "positive":           0,
            "negative":           0,
            "neutral":            0,
            "positive_pct":       0.0,
            "negative_pct":       0.0,
            "neutral_pct":        0.0,
            "overall_label":      "neutral",
            "avg_positive_score": 0.0,
            "avg_negative_score": 0.0,
            "total":              0,
        }

    total  = len(results)
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    pos_scores, neg_scores = [], []

    for r in results:
        label = r.get("label", "neutral")
        counts[label] = counts.get(label, 0) + 1

        all_s = r.get("all", {})
        pos_scores.append(all_s.get("positive", 0.0))
        neg_scores.append(all_s.get("negative", 0.0))

    overall_label = max(counts, key=counts.get)

    return {
        "positive":           counts["positive"],
        "negative":           counts["negative"],
        "neutral":            counts["neutral"],
        "positive_pct":       round(counts["positive"] / total * 100, 1),
        "negative_pct":       round(counts["negative"] / total * 100, 1),
        "neutral_pct":        round(counts["neutral"]  / total * 100, 1),
        "overall_label":      overall_label,
        "avg_positive_score": round(sum(pos_scores) / total, 3),
        "avg_negative_score": round(sum(neg_scores) / total, 3),
        "total":              total,
    }