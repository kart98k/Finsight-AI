from typing import TypedDict, Optional


class GraphState(TypedDict):
    ticker:              str
    company_name:        str
    raw_income:          list
    raw_balance:         list
    raw_cashflow:        list
    raw_metrics:         list
    kpis:                dict
    chart_data:          dict
    insights:            str
    news_sentiment:      dict
    transcript_analysis: dict
    error:               Optional[str]