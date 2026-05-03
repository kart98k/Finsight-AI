import os
import anthropic
from datetime import datetime


def _get_anthropic_key() -> str:
    try:
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY", "")


def summarize_transcript(ticker: str, transcript: dict) -> dict:
    """
    Use Claude Haiku to extract structured NLP insights from
    an earnings call transcript.
    Returns a dict with: summary, themes, tone, guidance, risks, date, quarter, year
    """
    if not transcript:
        return {
            "summary":  "No transcript available for this ticker.",
            "themes":   [],
            "tone":     "N/A",
            "guidance": "N/A",
            "risks":    [],
            "date":     "N/A",
            "quarter":  "N/A",
            "year":     "N/A",
        }

    content = transcript.get("content", "")
    date    = transcript.get("date",    "N/A")
    quarter = transcript.get("quarter", None)
    year    = transcript.get("year",    None)

    # Derive quarter and year from date if not provided by API
    if date and date != "N/A" and len(date) >= 7:
        try:
            dt = datetime.strptime(date[:10], "%Y-%m-%d")
            if not year:
                year = str(dt.year)
            if not quarter:
                quarter = str((dt.month - 1) // 3 + 1)
        except Exception:
            pass

    quarter = quarter or "N/A"
    year    = year    or "N/A"

    content_excerpt = content[:4000] if content else ""

    if not content_excerpt:
        return {
            "summary":  "Transcript content is empty.",
            "themes":   [],
            "tone":     "N/A",
            "guidance": "N/A",
            "risks":    [],
            "date":     date,
            "quarter":  quarter,
            "year":     year,
        }

    api_key = _get_anthropic_key()
    client  = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are an NLP analyst specializing in financial earnings calls.
Analyze this earnings call transcript for {ticker} (Q{quarter} {year}, {date}).

TRANSCRIPT EXCERPT:
{content_excerpt}

Extract the following and respond in this exact format:

SUMMARY:
[2-3 sentence overview of the call]

TONE:
[One word: Optimistic / Cautious / Neutral / Concerned / Confident]

KEY THEMES:
- [theme 1]
- [theme 2]
- [theme 3]

FORWARD GUIDANCE:
[What management said about future outlook, revenue, or growth]

KEY RISKS MENTIONED:
- [risk 1]
- [risk 2]

Be concise and data-driven. Only use information from the transcript."""

    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text

    # ── Parse structured output ────────────────────────────────────────────────
    def extract_section(text: str, header: str, next_header: str = None) -> str:
        try:
            start = text.index(header) + len(header)
            if next_header and next_header in text:
                end = text.index(next_header)
                return text[start:end].strip()
            return text[start:].strip()
        except ValueError:
            return ""

    def extract_bullets(text: str, header: str, next_header: str = None) -> list:
        section = extract_section(text, header, next_header)
        lines   = [
            l.strip().lstrip("- ").strip()
            for l in section.split("\n")
            if l.strip().startswith("-")
        ]
        return lines if lines else [section.strip()] if section.strip() else []

    summary  = extract_section(raw, "SUMMARY:",              "TONE:")
    tone     = extract_section(raw, "TONE:",                 "KEY THEMES:")
    guidance = extract_section(raw, "FORWARD GUIDANCE:",     "KEY RISKS MENTIONED:")
    themes   = extract_bullets(raw, "KEY THEMES:",           "FORWARD GUIDANCE:")
    risks    = extract_bullets(raw, "KEY RISKS MENTIONED:")

    return {
        "summary":  summary  or raw,
        "tone":     tone     or "N/A",
        "themes":   themes   or [],
        "guidance": guidance or "N/A",
        "risks":    risks    or [],
        "date":     date,
        "quarter":  quarter,
        "year":     year,
    }