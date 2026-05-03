import plotly.graph_objects as go

COLORS = {
    "blue": "#378ADD",
    "teal": "#1D9E75",
    "coral": "#D85A30",
    "amber": "#BA7517",
    "purple": "#7F77DD",
    "gray": "#888780",
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", size=12, color="#444441"),
    margin=dict(l=40, r=20, t=70, b=40),
    title=dict(
        font=dict(size=14, color="#444441"),
        x=0,
        xanchor="left",
        y=1,
        yanchor="top",
        pad=dict(b=10),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.08,
        xanchor="right",
        x=1,
        font=dict(size=11),
    ),
)


def revenue_chart(chart_data: dict) -> go.Figure:
    years = chart_data["revenue"]["years"][::-1]
    revenues = [v / 1e9 for v in chart_data["revenue"]["values"][::-1]]
    net_incomes = [v / 1e9 for v in chart_data["net_income"]["values"][::-1]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years, y=revenues,
        name="Revenue",
        marker_color=COLORS["blue"],
    ))
    fig.add_trace(go.Bar(
        x=years, y=net_incomes,
        name="Net Income",
        marker_color=COLORS["teal"],
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title_text="Revenue vs Net Income (USD Billions)",
        barmode="group",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#E8E8E8", ticksuffix="B"),
    )
    return fig


def margins_chart(chart_data: dict) -> go.Figure:
    years = chart_data["margins"]["years"][::-1]
    gross = chart_data["margins"]["gross"][::-1]
    operating = chart_data["margins"]["operating"][::-1]
    net = chart_data["margins"]["net"][::-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=gross, name="Gross Margin",
        mode="lines+markers", line=dict(color=COLORS["blue"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=years, y=operating, name="Operating Margin",
        mode="lines+markers", line=dict(color=COLORS["teal"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=years, y=net, name="Net Margin",
        mode="lines+markers", line=dict(color=COLORS["coral"], width=2),
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title_text="Profit Margins Over Time (%)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#E8E8E8", ticksuffix="%"),
    )
    return fig


def cashflow_chart(chart_data: dict) -> go.Figure:
    years = chart_data["cashflow"]["years"][::-1]
    operating = [v / 1e9 for v in chart_data["cashflow"]["operating"][::-1]]
    capex = [v / 1e9 for v in chart_data["cashflow"]["capex"][::-1]]
    free = [v / 1e9 for v in chart_data["cashflow"]["free"][::-1]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years, y=operating, name="Operating CF",
        marker_color=COLORS["teal"],
    ))
    fig.add_trace(go.Bar(
        x=years, y=capex, name="CapEx",
        marker_color=COLORS["coral"],
    ))
    fig.add_trace(go.Scatter(
        x=years, y=free, name="Free Cash Flow",
        mode="lines+markers", line=dict(color=COLORS["purple"], width=2),
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title_text="Cash Flow Analysis (USD Billions)",
        barmode="group",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#E8E8E8", ticksuffix="B"),
    )
    return fig


def debt_equity_chart(chart_data: dict) -> go.Figure:
    years = chart_data["balance"]["years"][::-1]
    debt = [v / 1e9 for v in chart_data["balance"]["debt"][::-1]]
    equity = [v / 1e9 for v in chart_data["balance"]["equity"][::-1]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years, y=debt, name="Total Debt",
        marker_color=COLORS["coral"],
    ))
    fig.add_trace(go.Bar(
        x=years, y=equity, name="Total Equity",
        marker_color=COLORS["blue"],
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title_text="Debt vs Equity (USD Billions)",
        barmode="group",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#E8E8E8", ticksuffix="B"),
    )
    return fig


def de_ratio_gauge(de_ratio: float) -> go.Figure:
    capped = min(de_ratio, 5.0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(de_ratio, 2),
        title={"text": "Debt / Equity Ratio", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 5]},
            "bar": {"color": COLORS["blue"]},
            "steps": [
                {"range": [0, 1], "color": "#E1F5EE"},
                {"range": [1, 2.5], "color": "#FAEEDA"},
                {"range": [2.5, 5], "color": "#FAECE7"},
            ],
            "threshold": {
                "line": {"color": COLORS["coral"], "width": 2},
                "thickness": 0.75,
                "value": capped,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=12, color="#444441"),
        margin=dict(l=20, r=20, t=60, b=20),
        height=220,
    )
    return fig
    
def sentiment_donut_chart(aggregate: dict) -> go.Figure:
    labels = ["Positive", "Negative", "Neutral"]
    values = [
        aggregate.get("positive", 0),
        aggregate.get("negative", 0),
        aggregate.get("neutral",  0),
    ]
    colors = [COLORS["teal"], COLORS["coral"], COLORS["gray"]]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} headlines<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title_text="News Sentiment Distribution",
        showlegend=True,
        height=320,
    )
    return fig


def sentiment_timeline_chart(results: list) -> go.Figure:
    from collections import defaultdict

    daily = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})
    for r in results:
        date  = r.get("date", "")
        label = r.get("label", "neutral")
        if date:
            daily[date][label] += 1

    dates     = sorted(daily.keys())
    positives = [daily[d]["positive"] for d in dates]
    negatives = [daily[d]["negative"] for d in dates]
    neutrals  = [daily[d]["neutral"]  for d in dates]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=positives, name="Positive", marker_color=COLORS["teal"]))
    fig.add_trace(go.Bar(x=dates, y=negatives, name="Negative", marker_color=COLORS["coral"]))
    fig.add_trace(go.Bar(x=dates, y=neutrals,  name="Neutral",  marker_color=COLORS["gray"]))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title_text="Sentiment Over Time",
        barmode="stack",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#E8E8E8"),
        height=320,
    )
    return fig


def keyword_frequency_chart(keywords: list) -> go.Figure:
    """Horizontal bar chart of top keywords from news headlines."""
    if not keywords:
        return go.Figure()

    words  = [k[0] for k in reversed(keywords)]
    counts = [k[1] for k in reversed(keywords)]

    fig = go.Figure(go.Bar(
        x=counts,
        y=words,
        orientation="h",
        marker=dict(
            color=counts,
            colorscale=[[0, COLORS["blue"]], [1, COLORS["teal"]]],
            showscale=False,
        ),
        text=counts,
        textposition="outside",
    ))

    # Build layout without margin first then update margin separately
    layout = {k: v for k, v in LAYOUT_DEFAULTS.items() if k != "margin"}

    fig.update_layout(
        **layout,
        title_text="Top Keywords from Recent News Headlines",
        xaxis=dict(showgrid=True, gridcolor="#E8E8E8", title="Frequency"),
        yaxis=dict(showgrid=False),
        height=420,
    )

    # Update margin separately to avoid conflict with LAYOUT_DEFAULTS
    fig.update_layout(margin=dict(l=120, r=60, t=70, b=40))

    return fig