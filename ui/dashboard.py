import os
import time
import streamlit as st

st.set_page_config(
    page_title="FinSight AI",
    page_icon="📈",
    layout="wide",
)

TICKER_OPTIONS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
    "Finance":    ["JPM",  "BAC",  "GS",    "WFC",  "C"],
    "Healthcare": ["JNJ",  "PFE",  "ABBV", "UNH"],
    "Consumer":   ["AMZN", "TSLA", "WMT",   "COST",  "NKE"],
    "Energy":     ["XOM",  "CVX"],
}

ALL_TICKERS = [ticker for group in TICKER_OPTIONS.values() for ticker in group]


# ── Helper: run LangGraph pipeline with progress bar ──────────────────────────
def run_analysis(ticker: str, progress_bar=None, status_text=None) -> dict:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.graph import graph

    steps = [
        "📡 Fetching financial data...",
        "🔢 Extracting KPIs...",
        "📰 Analysing news sentiment...",
        "🎙️ Summarising earnings call...",
        "🤖 Generating AI insights...",
        "📊 Building charts...",
        "✅ Analysis complete!",
    ]

    def update(step_idx):
        if progress_bar and status_text:
            progress_bar.progress(step_idx / (len(steps) - 1))
            status_text.markdown(f"**{steps[step_idx]}**")

    update(0)

    result = graph.invoke({
        "ticker":              ticker,
        "company_name":        "",
        "raw_income":          [],
        "raw_balance":         [],
        "raw_cashflow":        [],
        "raw_metrics":         [],
        "kpis":                {},
        "chart_data":          {},
        "insights":            "",
        "news_sentiment":      {},
        "transcript_analysis": {},
        "error":               None,
    })

    for i in range(1, len(steps)):
        update(i)
        time.sleep(0.3)

    return result


# ── Helper: company logo + stock price header ──────────────────────────────────
def render_company_header(company_name: str, ticker: str, profile: dict, quote: dict):
    logo_url   = profile.get("image",             "")
    price      = quote.get("price",               0)
    change     = quote.get("change",              0)
    change_pct = quote.get("changesPercentage",   0)
    exchange   = profile.get("exchangeShortName", "")
    sector     = profile.get("sector",            "")

    arrow = "▲" if change >= 0 else "▼"
    color = "green" if change >= 0 else "red"

    col_logo, col_info = st.columns([1, 5])
    with col_logo:
        if logo_url:
            st.image(logo_url, width=72)
    with col_info:
        st.markdown(f"## {company_name} &nbsp; `{ticker}`")
        st.markdown(
            f"**${price:,.2f}** &nbsp; "
            f"<span style='color:{color}'>{arrow} {change:+.2f} ({change_pct:+.2f}%)</span>"
            f" &nbsp; · &nbsp; {exchange} &nbsp; · &nbsp; {sector}",
            unsafe_allow_html=True,
        )


# ── Helper: KPI cards ──────────────────────────────────────────────────────────
def render_kpis(kpis: dict):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Revenue",
            value=f"${kpis['latest_revenue'] / 1e9:.2f}B",
            delta=f"{kpis['revenue_growth']:.1f}% YoY",
        )
    with col2:
        st.metric(label="Net Income",     value=f"${kpis['latest_net_income'] / 1e9:.2f}B")
    with col3:
        st.metric(label="Net Margin",     value=f"{kpis['net_margin']:.1f}%")
    with col4:
        st.metric(label="Free Cash Flow", value=f"${kpis['free_cash_flow'] / 1e9:.2f}B")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric(label="Gross Margin",     value=f"{kpis['gross_margin']:.1f}%")
    with col6:
        st.metric(label="Operating Margin", value=f"{kpis['operating_margin']:.1f}%")
    with col7:
        st.metric(label="Return on Equity", value=f"{kpis['roe']:.1f}%")
    with col8:
        st.metric(label="Current Ratio",    value=f"{kpis['current_ratio']:.2f}")


# ── Helper: financial charts ───────────────────────────────────────────────────
def render_charts(chart_data: dict, kpis: dict):
    from ui.charts import (
        revenue_chart,
        margins_chart,
        cashflow_chart,
        debt_equity_chart,
        de_ratio_gauge,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(revenue_chart(chart_data),     use_container_width=True)
    with col_b:
        st.plotly_chart(margins_chart(chart_data),     use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(cashflow_chart(chart_data),    use_container_width=True)
    with col_d:
        st.plotly_chart(debt_equity_chart(chart_data), use_container_width=True)

    col_e, _ = st.columns([1, 1])
    with col_e:
        st.plotly_chart(de_ratio_gauge(kpis["debt_to_equity"]), use_container_width=True)


# ── Helper: News Sentiment ─────────────────────────────────────────────────────
def render_sentiment(news_sentiment: dict, ticker: str):
    from ui.charts import (
        sentiment_donut_chart,
        sentiment_timeline_chart,
        keyword_frequency_chart,
    )

    st.markdown("#### 📰 News Sentiment Analysis")

    aggregate  = news_sentiment.get("aggregate",  {})
    results    = news_sentiment.get("results",    [])
    keywords   = news_sentiment.get("keywords",   [])
    pipeline   = news_sentiment.get("pipeline",   {})
    comparison = news_sentiment.get("comparison", [])

    if not aggregate:
        st.warning(f"No news data available for {ticker}.")
        return

    overall    = aggregate.get("overall_label", "neutral").capitalize()
    color_icon = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}.get(overall, "⚪")

    st.markdown(
        f"**Overall Market Sentiment: {color_icon} {overall}** "
        f"— based on {aggregate.get('total', 0)} headlines analysed by VADER"
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("🟢 Positive", aggregate.get('positive', 0), f"{aggregate.get('positive_pct', 0)}%")
    with m2:
        st.metric("🔴 Negative", aggregate.get('negative', 0), f"{aggregate.get('negative_pct', 0)}%")
    with m3:
        st.metric("🟡 Neutral",  aggregate.get('neutral',  0), f"{aggregate.get('neutral_pct',  0)}%")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(sentiment_donut_chart(aggregate), use_container_width=True)
    with c2:
        if results:
            st.plotly_chart(sentiment_timeline_chart(results), use_container_width=True)

    # ── Most Positive and Most Negative Headline ──────────────────────────────
    if results:
        positive_results = [r for r in results if r["label"] == "positive"]
        negative_results = [r for r in results if r["label"] == "negative"]

        best  = max(positive_results, key=lambda x: x.get("confidence", 0)) if positive_results else None
        worst = max(negative_results, key=lambda x: x.get("confidence", 0)) if negative_results else None

        hi_col, lo_col = st.columns(2)
        with hi_col:
            with st.container(border=True):
                st.markdown("**🟢 Most Positive Headline**")
                if best:
                    st.markdown(f"*\"{best['text']}\"*")
                    st.caption(f"Confidence: **{best.get('confidence_pct', '')}** · {best.get('date', '')}")
                    if best.get("url"):
                        st.markdown(f"[Read article →]({best['url']})")
                else:
                    st.markdown("No positive headlines found.")

        with lo_col:
            with st.container(border=True):
                st.markdown("**🔴 Most Negative Headline**")
                if worst:
                    st.markdown(f"*\"{worst['text']}\"*")
                    st.caption(f"Confidence: **{worst.get('confidence_pct', '')}** · {worst.get('date', '')}")
                    if worst.get("url"):
                        st.markdown(f"[Read article →]({worst['url']})")
                else:
                    st.markdown("No negative headlines found.")

        st.divider()

    # ── Headlines with confidence scores ──────────────────────────────────────
    if results:
        st.markdown("**Recent Headlines with Confidence Scores**")
        for r in results[:8]:
            icon       = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(r["label"], "⚪")
            label      = r["label"].capitalize()
            confidence = r.get("confidence_pct", "")
            text       = r["text"][:110] + "..." if len(r["text"]) > 110 else r["text"]
            url        = r.get("url",  "")
            date       = r.get("date", "")
            conf_str   = f" `{confidence}`" if confidence else ""
            if url:
                st.markdown(f"{icon} **{label}**{conf_str} [{text}]({url}) — *{date}*")
            else:
                st.markdown(f"{icon} **{label}**{conf_str} {text} — *{date}*")

    st.divider()

    # ── Keyword Frequency + Word Cloud ────────────────────────────────────────
    if keywords:
        st.markdown("#### 🔑 Keyword Analysis")
        st.caption("Top keywords extracted from recent news headlines after tokenization and stopword removal.")

        wc_fig = news_sentiment.get("wordcloud", None)
        if wc_fig is not None:
            col_kw, col_wc = st.columns(2)
            with col_kw:
                st.markdown("**Keyword Frequency**")
                st.plotly_chart(keyword_frequency_chart(keywords), use_container_width=True)
            with col_wc:
                st.markdown("**Word Cloud**")
                st.pyplot(wc_fig, use_container_width=True)
        else:
            st.plotly_chart(keyword_frequency_chart(keywords), use_container_width=True)

    st.divider()

    # ── NLP Pipeline Expander ─────────────────────────────────────────────────
    if pipeline:
        with st.expander("🔬 NLP Pipeline - Step by Step Breakdown"):
            st.markdown("This expander shows how a raw news headline is processed through the NLP pipeline.")
            st.divider()

            st.markdown("**📌 Step 1 - Raw Headline**")
            st.info(pipeline.get("raw", ""))

            st.markdown("**🔡 Step 2 - Lowercased**")
            st.code(pipeline.get("lowercased", ""), language=None)

            st.markdown("**✂️ Step 3 - Tokenized**")
            st.code(str(pipeline.get("tokens", [])), language=None)

            st.markdown("**🧹 Step 4 - Stopwords Removed**")
            st.code(str(pipeline.get("cleaned", [])), language=None)

            st.markdown("**🏷️ Step 5 - POS Tags**")
            st.code(str(pipeline.get("pos_tags", [])), language=None)
            st.caption("NN=Noun, VB=Verb, JJ=Adjective, RB=Adverb, NNP=Proper Noun")

            st.markdown("**💬 Step 6 - Sentiment Score (VADER)**")
            scores    = pipeline.get("all_scores", {})
            compound  = pipeline.get("compound",   0)
            sentiment = pipeline.get("sentiment",  "Neutral")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1: st.metric("Positive", f"{scores.get('pos', 0):.3f}")
            with col_s2: st.metric("Negative", f"{scores.get('neg', 0):.3f}")
            with col_s3: st.metric("Neutral",  f"{scores.get('neu', 0):.3f}")
            with col_s4: st.metric("Compound", f"{compound:.3f}")
            icon = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}.get(sentiment, "⚪")
            st.success(f"**Final Sentiment: {icon} {sentiment}**")

    st.divider()

    # ── VADER Sentiment Analysis Expander ─────────────────────────────────────
    if comparison:
        with st.expander("📊 Sentiment Model VADER Analysis"):
            st.markdown("""
**VADER (Valence Aware Dictionary and sEntiment Reasoner)** is a rule-based
sentiment model specifically designed for social media and financial news text.
It uses a lexicon of words rated for sentiment polarity and intensity.
Financial language nuances — words like *correction*, *bearish*, *headwinds*,
*miss*, *beat* — are captured through VADER's financial domain lexicon.
            """)

            st.markdown("| # | Headline | Sentiment | VADER Score |")
            st.markdown("|---|---|---|---|")

            icons = {"positive": "🟢 Positive", "negative": "🔴 Negative", "neutral": "🟡 Neutral"}

            for i, row in enumerate(comparison, 1):
                v     = icons.get(row["vader"], row["vader"])
                score = row.get("vader_score", 0)
                st.markdown(f"| {i} | {row['headline']} | {v} | `{score}` |")

            st.divider()
            total_pos = sum(1 for r in comparison if r["vader"] == "positive")
            total_neg = sum(1 for r in comparison if r["vader"] == "negative")
            total_neu = sum(1 for r in comparison if r["vader"] == "neutral")

            a1, a2, a3 = st.columns(3)
            with a1: st.metric("🟢 Positive", total_pos)
            with a2: st.metric("🔴 Negative", total_neg)
            with a3: st.metric("🟡 Neutral",  total_neu)


# ── Helper: Transcript Analysis ───────────────────────────────────────────────
def render_transcript(transcript_analysis: dict, ticker: str):
    st.markdown("#### 🎙️ Earnings Call Transcript Analysis")

    if not transcript_analysis or transcript_analysis.get("summary") in ("N/A", ""):
        st.warning(f"No earnings call transcript available for {ticker}.")
        return

    quarter = transcript_analysis.get("quarter", "N/A")
    year    = transcript_analysis.get("year",    "N/A")
    date    = transcript_analysis.get("date",    "N/A")
    tone    = transcript_analysis.get("tone",    "N/A")

    tone_color = {
        "Optimistic": "🟢", "Confident": "🟢",
        "Cautious":   "🟡", "Neutral":   "🟡",
        "Concerned":  "🔴",
    }.get(tone, "⚪")

    st.caption(f"Q{quarter} {year} — {date}")

    with st.container(border=True):
        tone_col, summary_col = st.columns([1, 3])
        with tone_col:
            st.markdown("**Management Tone**")
            st.markdown(f"#### {tone_color} {tone}")
        with summary_col:
            st.markdown("**Call Summary**")
            st.markdown(transcript_analysis.get("summary", "N/A"))

    col_themes, col_risks = st.columns(2)
    with col_themes:
        with st.container(border=True):
            st.markdown("**🔑 Key Themes**")
            themes = transcript_analysis.get("themes", [])
            if themes:
                for theme in themes:
                    st.markdown(f"- {theme}")
            else:
                st.markdown("No themes extracted.")

    with col_risks:
        with st.container(border=True):
            st.markdown("**⚠️ Risks Mentioned**")
            risks = transcript_analysis.get("risks", [])
            if risks:
                for risk in risks:
                    st.markdown(f"- {risk}")
            else:
                st.markdown("No risks extracted.")

    with st.container(border=True):
        st.markdown("**📈 Forward Guidance**")
        st.markdown(transcript_analysis.get("guidance", "N/A"))


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("📈 FinSight AI")

st.markdown("""
Analyse any public company using **real-time financials** and **NLP-powered insights** 
from VADER news sentiment and earnings call summarization to KPI dashboards and
competitor comparison. Built on LangGraph + Claude Haiku.
""")

st.caption("😄 Built by **Srikonda Karthik**")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🔑 API Key")

    anthropic_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your key at console.anthropic.com",
    )

    key_provided = bool(anthropic_key)

    if not key_provided:
        st.warning("Enter your Anthropic API key to start.")

    st.divider()

    st.header("⚙️ Mode")
    mode = st.radio(
        "Analysis mode",
        options=["Single Ticker", "Compare Two Tickers"],
        index=0,
    )

    st.divider()
    st.header("🔍 Company Lookup")

    sector = st.selectbox(
        "Filter by sector",
        options=["All"] + list(TICKER_OPTIONS.keys()),
    )

    available_tickers = ALL_TICKERS if sector == "All" else TICKER_OPTIONS[sector]

    ticker_1 = st.selectbox(
        "Select ticker" if mode == "Single Ticker" else "Select ticker 1",
        options=available_tickers,
        index=0,
        key="ticker_1",
    )

    custom_1 = st.text_input(
        "Or type a custom ticker" if mode == "Single Ticker" else "Or type custom ticker 1",
        placeholder="e.g. TSM, SAP, BABA",
        max_chars=10,
        key="custom_1",
    ).upper().strip()

    if custom_1:
        ticker_1 = custom_1

    ticker_2 = None
    if mode == "Compare Two Tickers":
        ticker_2 = st.selectbox(
            "Select ticker 2",
            options=available_tickers,
            index=1,
            key="ticker_2",
        )

        custom_2 = st.text_input(
            "Or type custom ticker 2",
            placeholder="e.g. TSM, SAP, BABA",
            max_chars=10,
            key="custom_2",
        ).upper().strip()

        if custom_2:
            ticker_2 = custom_2

        if ticker_1 == ticker_2:
            st.warning("Please select two different tickers.")

    analyze = st.button(
        "Analyze",
        use_container_width=True,
        type="primary",
        disabled=not key_provided,
    )

    st.divider()
    st.caption("🔒 Your API key is never stored or logged.")
    st.caption("It is used only for the duration of your session.")


# ══════════════════════════════════════════════════════════════════════════════
# LANDING STATE
# ══════════════════════════════════════════════════════════════════════════════
if not key_provided:
    with st.container(border=True):
        st.markdown("##### 🚀 Get started in 2 steps")
        st.markdown("""
1. Grab a free Anthropic key from [console.anthropic.com](https://console.anthropic.com)
2. Paste it in the sidebar, pick a ticker, and hit **Analyze**
        """)

elif analyze:
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key

    from services.claude_client  import stream_financial_insights
    from services.fmp_client     import get_company_profile, get_stock_quote

    # ══════════════════════════════════════════════════════════════════════════
    # SINGLE TICKER MODE
    # ══════════════════════════════════════════════════════════════════════════
    if mode == "Single Ticker":

        progress_bar = st.progress(0)
        status_text  = st.empty()
        result       = run_analysis(ticker_1, progress_bar, status_text)
        progress_bar.empty()
        status_text.empty()

        if result.get("error"):
            st.error(f"Error: {result['error']}")
            st.stop()

        kpis                = result["kpis"]
        chart_data          = result["chart_data"]
        company_name        = result["company_name"]
        news_sentiment      = result.get("news_sentiment",      {})
        transcript_analysis = result.get("transcript_analysis", {})

        profile = get_company_profile(ticker_1)
        quote   = get_stock_quote(ticker_1)
        render_company_header(company_name, ticker_1, profile, quote)
        st.divider()

        st.markdown("#### Key Performance Indicators")
        render_kpis(kpis)

        st.divider()

        st.markdown("#### 🤖 AI Insights")
        with st.container(border=True):
            st.write_stream(stream_financial_insights(ticker_1, kpis))

        st.divider()
        render_sentiment(news_sentiment, ticker_1)

        st.divider()
        render_transcript(transcript_analysis, ticker_1)

        st.divider()
        st.markdown("#### 📊 Financial Charts")
        render_charts(chart_data, kpis)

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARE TWO TICKERS MODE
    # ══════════════════════════════════════════════════════════════════════════
    elif mode == "Compare Two Tickers" and ticker_2 and ticker_1 != ticker_2:

        st.markdown(f"**Analysing {ticker_1}...**")
        progress_bar_1 = st.progress(0)
        status_text_1  = st.empty()
        result_1       = run_analysis(ticker_1, progress_bar_1, status_text_1)
        progress_bar_1.empty()
        status_text_1.empty()

        st.markdown(f"**Analysing {ticker_2}...**")
        progress_bar_2 = st.progress(0)
        status_text_2  = st.empty()
        result_2       = run_analysis(ticker_2, progress_bar_2, status_text_2)
        progress_bar_2.empty()
        status_text_2.empty()

        if result_1.get("error"):
            st.error(f"Error fetching {ticker_1}: {result_1['error']}")
        if result_2.get("error"):
            st.error(f"Error fetching {ticker_2}: {result_2['error']}")
        if result_1.get("error") or result_2.get("error"):
            st.stop()

        kpis_1                = result_1["kpis"]
        kpis_2                = result_2["kpis"]
        chart_data_1          = result_1["chart_data"]
        chart_data_2          = result_2["chart_data"]
        name_1                = result_1["company_name"]
        name_2                = result_2["company_name"]
        news_sentiment_1      = result_1.get("news_sentiment",      {})
        news_sentiment_2      = result_2.get("news_sentiment",      {})
        transcript_analysis_1 = result_1.get("transcript_analysis", {})
        transcript_analysis_2 = result_2.get("transcript_analysis", {})

        profile_1 = get_company_profile(ticker_1)
        profile_2 = get_company_profile(ticker_2)
        quote_1   = get_stock_quote(ticker_1)
        quote_2   = get_stock_quote(ticker_2)

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            render_company_header(name_1, ticker_1, profile_1, quote_1)
        with col_h2:
            render_company_header(name_2, ticker_2, profile_2, quote_2)

        st.divider()

        # ── KPI comparison table ──────────────────────────────────────────────
        st.markdown("#### 📋 KPI Comparison")

        kpi_labels = {
            "Revenue":          (kpis_1["latest_revenue"],    kpis_2["latest_revenue"],    "B", 1e9),
            "Net Income":       (kpis_1["latest_net_income"], kpis_2["latest_net_income"], "B", 1e9),
            "Free Cash Flow":   (kpis_1["free_cash_flow"],    kpis_2["free_cash_flow"],    "B", 1e9),
            "Revenue Growth":   (kpis_1["revenue_growth"],    kpis_2["revenue_growth"],    "%", 1),
            "Gross Margin":     (kpis_1["gross_margin"],      kpis_2["gross_margin"],      "%", 1),
            "Net Margin":       (kpis_1["net_margin"],        kpis_2["net_margin"],        "%", 1),
            "Operating Margin": (kpis_1["operating_margin"],  kpis_2["operating_margin"],  "%", 1),
            "Return on Equity": (kpis_1["roe"],               kpis_2["roe"],               "%", 1),
            "Current Ratio":    (kpis_1["current_ratio"],     kpis_2["current_ratio"],     "x", 1),
            "Debt / Equity":    (kpis_1["debt_to_equity"],    kpis_2["debt_to_equity"],    "x", 1),
        }

        lower_is_better = {"Debt / Equity"}

        h_col, v_col_1, v_col_2, w_col = st.columns([2, 1.5, 1.5, 1])
        with h_col:   st.markdown("**Metric**")
        with v_col_1: st.markdown(f"**{ticker_1}**")
        with v_col_2: st.markdown(f"**{ticker_2}**")
        with w_col:   st.markdown("**Better**")
        st.markdown("---")

        for label, (v1, v2, suffix, divisor) in kpi_labels.items():
            h_col, v_col_1, v_col_2, w_col = st.columns([2, 1.5, 1.5, 1])
            if suffix == "B":
                fmt1, fmt2 = f"${v1 / divisor:.2f}B", f"${v2 / divisor:.2f}B"
            elif suffix == "%":
                fmt1, fmt2 = f"{v1:.1f}%", f"{v2:.1f}%"
            else:
                fmt1, fmt2 = f"{v1:.2f}x", f"{v2:.2f}x"
            winner = ticker_1 if (v1 < v2 if label in lower_is_better else v1 > v2) else ticker_2
            with h_col:   st.markdown(label)
            with v_col_1: st.markdown(fmt1)
            with v_col_2: st.markdown(fmt2)
            with w_col:   st.markdown(f"✅ {winner}")

        st.divider()

        # ── AI Investment Recommendation ──────────────────────────────────────
        st.markdown("#### 🤖 AI Investment Recommendation")
        with st.spinner("Generating investment recommendation..."):
            from services.claude_client import get_investment_recommendation
            recommendation = get_investment_recommendation(
                ticker_1, kpis_1,
                ticker_2, kpis_2,
            )

        rec_lines  = recommendation.strip().split("\n")
        rec_ticker = ""
        rec_reason = ""
        rec_risk   = ""

        for line in rec_lines:
            if line.startswith("RECOMMENDATION:"):
                rec_ticker = line.replace("RECOMMENDATION:", "").strip()
            elif line.startswith("REASON:"):
                rec_reason = line.replace("REASON:", "").strip()
            elif line.startswith("RISK:"):
                rec_risk = line.replace("RISK:", "").strip()

        with st.container(border=True):
            if rec_ticker:
                st.markdown(f"### 🟢 Recommended: **{rec_ticker}**")
            if rec_reason:
                st.markdown(f"**Why:** {rec_reason}")
            if rec_risk:
                st.markdown(f"**Key Risk:** ⚠️ {rec_risk}")
            if not rec_ticker:
                st.markdown(recommendation)

        st.divider()

        # ── AI Insights side by side ──────────────────────────────────────────
        st.markdown("#### 🤖 AI Insights")
        ins_col_1, ins_col_2 = st.columns(2)
        with ins_col_1:
            st.markdown(f"**{name_1} ({ticker_1})**")
            with st.container(border=True):
                st.write_stream(stream_financial_insights(ticker_1, kpis_1))
        with ins_col_2:
            st.markdown(f"**{name_2} ({ticker_2})**")
            with st.container(border=True):
                st.write_stream(stream_financial_insights(ticker_2, kpis_2))

        st.divider()

        # ── Sentiment side by side ────────────────────────────────────────────
        st.markdown("### 📰 News Sentiment Comparison")
        sent_col1, sent_col2 = st.columns(2)
        with sent_col1:
            st.markdown(f"**{name_1} ({ticker_1})**")
            render_sentiment(news_sentiment_1, ticker_1)
        with sent_col2:
            st.markdown(f"**{name_2} ({ticker_2})**")
            render_sentiment(news_sentiment_2, ticker_2)

        st.divider()

        # ── Transcript side by side ───────────────────────────────────────────
        st.markdown("### 🎙️ Earnings Call Comparison")
        trans_col1, trans_col2 = st.columns(2)
        with trans_col1:
            st.markdown(f"**{name_1} ({ticker_1})**")
            render_transcript(transcript_analysis_1, ticker_1)
        with trans_col2:
            st.markdown(f"**{name_2} ({ticker_2})**")
            render_transcript(transcript_analysis_2, ticker_2)

        st.divider()

        # ── Charts side by side ───────────────────────────────────────────────
        st.markdown("#### 📊 Financial Charts")

        from ui.charts import (
            revenue_chart, margins_chart, cashflow_chart,
            debt_equity_chart, de_ratio_gauge,
        )

        st.markdown("##### Revenue vs Net Income")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{ticker_1}**")
            st.plotly_chart(revenue_chart(chart_data_1), use_container_width=True)
        with c2:
            st.markdown(f"**{ticker_2}**")
            st.plotly_chart(revenue_chart(chart_data_2), use_container_width=True)

        st.markdown("##### Profit Margins")
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(margins_chart(chart_data_1), use_container_width=True)
        with c4:
            st.plotly_chart(margins_chart(chart_data_2), use_container_width=True)

        st.markdown("##### Cash Flow")
        c5, c6 = st.columns(2)
        with c5:
            st.plotly_chart(cashflow_chart(chart_data_1), use_container_width=True)
        with c6:
            st.plotly_chart(cashflow_chart(chart_data_2), use_container_width=True)

        st.markdown("##### Debt vs Equity")
        c7, c8 = st.columns(2)
        with c7:
            st.plotly_chart(debt_equity_chart(chart_data_1), use_container_width=True)
        with c8:
            st.plotly_chart(debt_equity_chart(chart_data_2), use_container_width=True)

        st.markdown("##### Debt / Equity Gauge")
        c9, c10 = st.columns(2)
        with c9:
            st.plotly_chart(de_ratio_gauge(kpis_1["debt_to_equity"]), use_container_width=True)
        with c10:
            st.plotly_chart(de_ratio_gauge(kpis_2["debt_to_equity"]), use_container_width=True)

else:
    st.info("Select a ticker from the sidebar and click **Analyze** to get started.")