import os
import datetime
import requests
import streamlit as st

FMP_BASE = "https://financialmodelingprep.com/stable"


def _get(endpoint: str, params: dict = {}) -> dict | list:
    api_key = os.environ.get("FMP_API_KEY", "")
    params  = {**params, "apikey": api_key}
    response = requests.get(f"{FMP_BASE}/{endpoint}", params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and "Error Message" in data:
        raise ValueError(data["Error Message"])
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def get_income_statement(ticker: str, limit: int = 5) -> list:
    data = _get("income-statement", {"symbol": ticker, "limit": limit, "period": "annual"})
    if not data:
        raise ValueError(f"No income statement data found for {ticker}")
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def get_balance_sheet(ticker: str, limit: int = 5) -> list:
    data = _get("balance-sheet-statement", {"symbol": ticker, "limit": limit, "period": "annual"})
    if not data:
        raise ValueError(f"No balance sheet data found for {ticker}")
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def get_cash_flow(ticker: str, limit: int = 5) -> list:
    data = _get("cash-flow-statement", {"symbol": ticker, "limit": limit, "period": "annual"})
    if not data:
        raise ValueError(f"No cash flow data found for {ticker}")
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def get_key_metrics(ticker: str, limit: int = 5) -> list:
    data = _get("key-metrics", {"symbol": ticker, "limit": limit, "period": "annual"})
    if not data:
        raise ValueError(f"No key metrics data found for {ticker}")
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def get_company_profile(ticker: str) -> dict:
    data = _get("profile", {"symbol": ticker})
    if not data:
        raise ValueError(f"No company profile found for {ticker}")
    return data[0] if isinstance(data, list) else data


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_news(ticker: str, limit: int = 30) -> list:
    """
    Fetch recent news headlines using NewsAPI.org.
    Uses precise query with company name + ticker symbol.
    Filters out unrelated articles using keyword matching.
    Free tier: 100 requests/day at newsapi.org.
    """
    try:
        news_api_key = os.environ.get("NEWS_API_KEY", "")
        if not news_api_key:
            return []

        # Get full company name for better search precision
        try:
            profile      = get_company_profile(ticker)
            company_name = profile.get("companyName", ticker)
        except Exception:
            company_name = ticker

        # Build precise query — exact phrase match forces relevance
        first_word    = company_name.split()[0]
        precise_query = f'"{first_word}" "{ticker}" stock'

        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q":        precise_query,
                "language": "en",
                "sortBy":   "publishedAt",
                "pageSize": limit * 2,   # fetch extra buffer for filtering
                "apiKey":   news_api_key,
            },
            timeout=10,
        )
        response.raise_for_status()
        data     = response.json()
        articles = data.get("articles", [])

        if not articles:
            return []

        # Keywords to match against title + description
        keywords = [
            ticker.lower(),
            first_word.lower(),
            company_name.lower()[:15],
        ]

        filtered = []
        for item in articles:
            title       = item.get("title",       "") or ""
            description = item.get("description", "") or ""
            combined    = (title + " " + description).lower()

            # Skip removed or empty articles
            if title == "[Removed]" or not title:
                continue

            # Keep only articles mentioning the company or ticker
            if any(kw in combined for kw in keywords):
                filtered.append({
                    "title":         title,
                    "text":          title,
                    "publishedDate": (item.get("publishedAt", "") or "")[:10],
                    "url":           item.get("url",  ""),
                    "site":          (item.get("source") or {}).get("name", ""),
                })

        # Fallback — if filtering removed everything use unfiltered results
        if not filtered:
            filtered = [
                {
                    "title":         item.get("title", ""),
                    "text":          item.get("title", ""),
                    "publishedDate": (item.get("publishedAt", "") or "")[:10],
                    "url":           item.get("url",  ""),
                    "site":          (item.get("source") or {}).get("name", ""),
                }
                for item in articles
                if item.get("title") and item.get("title") != "[Removed]"
            ]

        return filtered[:limit]

    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_earnings_transcript(ticker: str) -> dict:
    """
    Fetch the most recent earnings call transcript.
    Stable API requires explicit year + quarter params.
    Walks back up to 6 quarters to find the latest available transcript.
    """
    try:
        current_year  = datetime.datetime.now().year
        current_month = datetime.datetime.now().month

        if current_month <= 3:
            year, quarter = current_year - 1, 4
        elif current_month <= 6:
            year, quarter = current_year, 1
        elif current_month <= 9:
            year, quarter = current_year, 2
        else:
            year, quarter = current_year, 3

        for _ in range(6):
            try:
                data = _get(
                    "earning-call-transcript",
                    {"symbol": ticker, "year": year, "quarter": quarter}
                )
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                if isinstance(data, dict) and data:
                    return data
            except Exception:
                pass

            quarter -= 1
            if quarter == 0:
                quarter = 4
                year   -= 1

        return {}

    except Exception:
        return {}

@st.cache_data(ttl=300, show_spinner=False)
def get_stock_quote(ticker: str) -> dict:
    """Fetch current stock price and change for the header."""
    try:
        data = _get("quote", {"symbol": ticker})
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return {}
    except Exception:
        return {}