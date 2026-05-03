from agent.state import GraphState
from services.fmp_client import (
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_key_metrics,
    get_company_profile,
    get_stock_news,
    get_earnings_transcript,
)


def _extract_year(item: dict) -> str:
    year = item.get("calendarYear")
    if year and str(year) != "None" and str(year) != "":
        return str(year)
    date = item.get("date", "")
    if date and len(date) >= 4:
        return date[:4]
    return "N/A"


def fetch_data_node(state: GraphState) -> GraphState:
    ticker = state["ticker"].upper().strip()
    try:
        profile = get_company_profile(ticker)
        return {
            **state,
            "ticker":       ticker,
            "company_name": profile.get("companyName", ticker),
            "raw_income":   get_income_statement(ticker),
            "raw_balance":  get_balance_sheet(ticker),
            "raw_cashflow": get_cash_flow(ticker),
            "raw_metrics":  get_key_metrics(ticker),
            "error":        None,
        }
    except Exception as e:
        return {**state, "error": str(e)}


def analyst_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state
    try:
        income   = state["raw_income"]
        balance  = state["raw_balance"]
        cashflow = state["raw_cashflow"]
        metrics  = state["raw_metrics"]

        latest_income   = income[0]   if income          else {}
        prev_income     = income[1]   if len(income) > 1 else {}
        latest_balance  = balance[0]  if balance         else {}
        latest_cashflow = cashflow[0] if cashflow        else {}
        latest_metrics  = metrics[0]  if metrics         else {}

        latest_revenue   = latest_income.get("revenue", 0)          or 0
        prev_revenue     = prev_income.get("revenue", 0)             or 0
        revenue_growth   = ((latest_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue else 0
        gross_profit     = latest_income.get("grossProfit", 0)       or 0
        net_income       = latest_income.get("netIncome", 0)         or 0
        operating_income = latest_income.get("operatingIncome", 0)   or 0
        gross_margin     = (gross_profit     / latest_revenue * 100) if latest_revenue else 0
        net_margin       = (net_income       / latest_revenue * 100) if latest_revenue else 0
        operating_margin = (operating_income / latest_revenue * 100) if latest_revenue else 0
        total_debt       = latest_balance.get("totalDebt", 0)                or 0
        total_equity     = latest_balance.get("totalStockholdersEquity", 0)  or 0
        debt_to_equity   = (total_debt / total_equity) if total_equity else 0
        operating_cf     = latest_cashflow.get("operatingCashFlow", 0)       or 0
        capex            = latest_cashflow.get("capitalExpenditure", 0)       or 0
        free_cash_flow   = operating_cf - abs(capex)
        roe              = latest_metrics.get("roe", 0)          or 0
        current_ratio    = latest_metrics.get("currentRatio", 0) or 0
        pe_ratio         = latest_metrics.get("peRatio", 0)      or 0

        kpis = {
            "latest_revenue":      latest_revenue,
            "prev_revenue":        prev_revenue,
            "revenue_growth":      revenue_growth,
            "gross_margin":        gross_margin,
            "net_margin":          net_margin,
            "operating_margin":    operating_margin,
            "latest_net_income":   net_income,
            "operating_cash_flow": operating_cf,
            "free_cash_flow":      free_cash_flow,
            "debt_to_equity":      debt_to_equity,
            "roe":                 roe * 100 if roe and roe < 1 else roe,
            "current_ratio":       current_ratio,
            "pe_ratio":            pe_ratio,
        }
        return {**state, "kpis": kpis, "error": None}

    except Exception as e:
        return {**state, "error": str(e)}


def news_sentiment_node(state: GraphState) -> GraphState:
    """
    Feature 1 — Fetch news + run FinBERT/VADER sentiment.
    Also prepares keyword, pipeline, confidence, model comparison and wordcloud data.
    """
    if state.get("error"):
        return state
    try:
        from services.sentiment_service import (
            analyze_with_finbert,
            aggregate_sentiment,
        )
        from services.nlp_service import (
            extract_keywords,
            get_pipeline_steps,
            format_results_with_confidence,
            compare_models,
            generate_wordcloud,
        )

        ticker = state["ticker"]
        news   = get_stock_news(ticker, limit=15)

        print(f"[Sentiment] Fetched {len(news)} articles for {ticker}")

        if not news:
            return {
                **state,
                "news_sentiment": {
                    "headlines":  [],
                    "results":    [],
                    "aggregate":  {},
                    "keywords":   [],
                    "pipeline":   {},
                    "comparison": [],
                    "wordcloud":  None,
                },
            }

        headlines = [
            item.get("title") or item.get("text", "")
            for item in news
            if item.get("title") or item.get("text")
        ][:15]

        headlines = [h for h in headlines if h and h != "[Removed]"]

        print(f"[Sentiment] Analysing {len(headlines)} headlines")

        # Run FinBERT / VADER sentiment
        finbert_results  = analyze_with_finbert(headlines)
        aggregate        = aggregate_sentiment(finbert_results)

        print(f"[Sentiment] {aggregate.get('overall_label')} | pos={aggregate.get('positive')} neg={aggregate.get('negative')} neu={aggregate.get('neutral')}")

        # Feature 3 — enrich with confidence scores
        enriched_results = format_results_with_confidence(finbert_results)

        # Attach date / url to each result
        dated_results = []
        for i, item in enumerate(news[:len(enriched_results)]):
            dated_results.append({
                **enriched_results[i],
                "date":   item.get("publishedDate", "")[:10],
                "source": item.get("site", ""),
                "url":    item.get("url",  ""),
            })

        # Feature 1 — keyword extraction
        keywords = extract_keywords(headlines, top_n=15)

        # Feature 2 — NLP pipeline for first headline
        pipeline = get_pipeline_steps(headlines[0]) if headlines else {}

        # Feature 4 — VADER vs FinBERT comparison
        comparison = compare_models(headlines, finbert_results)

        # Word cloud
        wordcloud_fig = generate_wordcloud(headlines)

        return {
            **state,
            "news_sentiment": {
                "headlines":  headlines,
                "results":    dated_results,
                "aggregate":  aggregate,
                "keywords":   keywords,
                "pipeline":   pipeline,
                "comparison": comparison,
                "wordcloud":  wordcloud_fig,
            },
        }

    except Exception as e:
        print(f"[Sentiment] Node error: {e}")
        return {
            **state,
            "news_sentiment": {
                "headlines":  [],
                "results":    [],
                "aggregate":  {},
                "keywords":   [],
                "pipeline":   {},
                "comparison": [],
                "wordcloud":  None,
                "error":      str(e),
            },
        }


def transcript_node(state: GraphState) -> GraphState:
    """
    Feature 2 — Fetch earnings call transcript and run NLP summarization.
    """
    if state.get("error"):
        return state
    try:
        from services.transcript_service import summarize_transcript

        ticker     = state["ticker"]
        transcript = get_earnings_transcript(ticker)
        analysis   = summarize_transcript(ticker, transcript)

        return {**state, "transcript_analysis": analysis}

    except Exception as e:
        return {
            **state,
            "transcript_analysis": {
                "summary":  f"Could not load transcript: {str(e)}",
                "themes":   [],
                "tone":     "N/A",
                "guidance": "N/A",
                "risks":    [],
            },
        }


def insights_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state
    # Insights are streamed directly in the UI layer
    return {**state, "insights": "__stream__", "error": None}


def chart_builder_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state
    try:
        income   = state["raw_income"]
        cashflow = state["raw_cashflow"]
        balance  = state["raw_balance"]

        years             = [_extract_year(i) for i in income]
        revenues          = [i.get("revenue", 0)        or 0 for i in income]
        net_incomes       = [i.get("netIncome", 0)       or 0 for i in income]
        gross_profits     = [i.get("grossProfit", 0)     or 0 for i in income]
        operating_incomes = [i.get("operatingIncome", 0) or 0 for i in income]

        gross_margins     = [(g / r * 100) if r else 0 for g, r in zip(gross_profits,     revenues)]
        net_margins       = [(n / r * 100) if r else 0 for n, r in zip(net_incomes,       revenues)]
        operating_margins = [(o / r * 100) if r else 0 for o, r in zip(operating_incomes, revenues)]

        cf_years       = [_extract_year(i) for i in cashflow]
        op_cashflows   = [i.get("operatingCashFlow", 0)    or 0 for i in cashflow]
        capexes        = [abs(i.get("capitalExpenditure", 0) or 0) for i in cashflow]
        free_cashflows = [o - c for o, c in zip(op_cashflows, capexes)]

        bal_years      = [_extract_year(i) for i in balance]
        total_debts    = [i.get("totalDebt", 0)               or 0 for i in balance]
        total_equities = [i.get("totalStockholdersEquity", 0) or 0 for i in balance]

        chart_data = {
            "revenue":    {"years": years, "values": revenues},
            "net_income": {"years": years, "values": net_incomes},
            "margins": {
                "years":     years,
                "gross":     gross_margins,
                "operating": operating_margins,
                "net":       net_margins,
            },
            "cashflow": {
                "years":     cf_years,
                "operating": op_cashflows,
                "capex":     capexes,
                "free":      free_cashflows,
            },
            "balance": {
                "years":  bal_years,
                "debt":   total_debts,
                "equity": total_equities,
            },
        }
        return {**state, "chart_data": chart_data, "error": None}

    except Exception as e:
        return {**state, "error": str(e)}


def aggregator_node(state: GraphState) -> GraphState:
    return state