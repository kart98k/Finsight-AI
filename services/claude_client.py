import os
import anthropic
from typing import Generator


def _get_anthropic_key() -> str:
    # Check os.environ first — set by dashboard when user enters key in sidebar
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if env_key:
        return env_key
    # Fall back to st.secrets
    try:
        import streamlit as st
        val = st.secrets.get("ANTHROPIC_API_KEY", "")
        if val:
            return val
    except Exception:
        pass
    return ""


def _build_prompt(ticker: str, kpis: dict) -> str:
    return f"""You are a senior financial analyst. Analyze the following KPIs for {ticker} and provide concise insights.

KPI Data:
- Revenue (latest): ${kpis.get('latest_revenue', 0):,.0f}
- Revenue Growth YoY: {kpis.get('revenue_growth', 0):.1f}%
- Net Income (latest): ${kpis.get('latest_net_income', 0):,.0f}
- Net Profit Margin: {kpis.get('net_margin', 0):.1f}%
- Gross Profit Margin: {kpis.get('gross_margin', 0):.1f}%
- Operating Cash Flow: ${kpis.get('operating_cash_flow', 0):,.0f}
- Free Cash Flow: ${kpis.get('free_cash_flow', 0):,.0f}
- Debt to Equity Ratio: {kpis.get('debt_to_equity', 0):.2f}
- Return on Equity: {kpis.get('roe', 0):.1f}%
- Current Ratio: {kpis.get('current_ratio', 0):.2f}

Provide exactly 3 insights in this format:
1. [Strength or positive trend observed]
2. [Risk or concern to watch]
3. [Overall outlook or recommendation]

Keep each point under 30 words. Be direct and data-driven."""


def get_financial_insights(ticker: str, kpis: dict) -> str:
    """Non-streaming version — returns full response as a string."""
    api_key = _get_anthropic_key()
    client  = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": _build_prompt(ticker, kpis)}]
    )
    return message.content[0].text


def stream_financial_insights(ticker: str, kpis: dict) -> Generator[str, None, None]:
    """Streaming version — yields text chunks word by word for st.write_stream."""
    api_key = _get_anthropic_key()
    client  = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": _build_prompt(ticker, kpis)}]
    ) as stream:
        for text in stream.text_stream:
            yield text


def get_investment_recommendation(
    ticker_1: str, kpis_1: dict,
    ticker_2: str, kpis_2: dict,
) -> str:
    """Generate a concise AI investment recommendation comparing two companies."""
    api_key = _get_anthropic_key()
    client  = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are a senior equity analyst. Compare these two companies and give a clear investment recommendation.

{ticker_1} Financials:
- Revenue: ${kpis_1.get('latest_revenue', 0) / 1e9:.2f}B | Growth: {kpis_1.get('revenue_growth', 0):.1f}%
- Net Margin: {kpis_1.get('net_margin', 0):.1f}% | Gross Margin: {kpis_1.get('gross_margin', 0):.1f}%
- Free Cash Flow: ${kpis_1.get('free_cash_flow', 0) / 1e9:.2f}B
- ROE: {kpis_1.get('roe', 0):.1f}% | Debt/Equity: {kpis_1.get('debt_to_equity', 0):.2f}
- Current Ratio: {kpis_1.get('current_ratio', 0):.2f}

{ticker_2} Financials:
- Revenue: ${kpis_2.get('latest_revenue', 0) / 1e9:.2f}B | Growth: {kpis_2.get('revenue_growth', 0):.1f}%
- Net Margin: {kpis_2.get('net_margin', 0):.1f}% | Gross Margin: {kpis_2.get('gross_margin', 0):.1f}%
- Free Cash Flow: ${kpis_2.get('free_cash_flow', 0) / 1e9:.2f}B
- ROE: {kpis_2.get('roe', 0):.1f}% | Debt/Equity: {kpis_2.get('debt_to_equity', 0):.2f}
- Current Ratio: {kpis_2.get('current_ratio', 0):.2f}

Respond in exactly this format:

RECOMMENDATION: [ticker_1 or ticker_2]

REASON: [2-3 sentences explaining why, citing specific metrics. Be direct and data-driven.]

RISK: [One sentence on the key risk of your recommended pick.]"""

    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text