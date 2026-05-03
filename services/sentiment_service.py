from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def analyze_with_vader(texts: list) -> list:
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
    FinBERT disabled on cloud deployment due to memory constraints.
    Using VADER which gives strong results for financial news headlines.
    """
    print("[Sentiment] Using VADER for sentiment analysis.")
    return analyze_with_vader(texts)


def aggregate_sentiment(results: list) -> dict:
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