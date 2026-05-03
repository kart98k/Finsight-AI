from langgraph.graph import StateGraph, END
from agent.state import GraphState
from agent.nodes import (
    fetch_data_node,
    analyst_node,
    news_sentiment_node,
    transcript_node,
    insights_node,
    chart_builder_node,
    aggregator_node,
)


def should_continue(state: GraphState) -> str:
    return "end" if state.get("error") else "continue"


def build_graph() -> StateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node("fetch_data",        fetch_data_node)
    workflow.add_node("analyst",           analyst_node)
    workflow.add_node("sentiment_runner",  news_sentiment_node)   # renamed
    workflow.add_node("transcript_runner", transcript_node)        # renamed
    workflow.add_node("insights_runner",   insights_node)          # renamed
    workflow.add_node("chart_builder",     chart_builder_node)
    workflow.add_node("aggregator",        aggregator_node)

    workflow.set_entry_point("fetch_data")

    workflow.add_conditional_edges(
        "fetch_data",
        should_continue,
        {"continue": "analyst", "end": END}
    )
    workflow.add_conditional_edges(
        "analyst",
        should_continue,
        {"continue": "sentiment_runner", "end": END}
    )
    workflow.add_conditional_edges(
        "sentiment_runner",
        should_continue,
        {"continue": "transcript_runner", "end": END}
    )
    workflow.add_conditional_edges(
        "transcript_runner",
        should_continue,
        {"continue": "insights_runner", "end": END}
    )
    workflow.add_conditional_edges(
        "insights_runner",
        should_continue,
        {"continue": "chart_builder", "end": END}
    )
    workflow.add_edge("chart_builder", "aggregator")
    workflow.add_edge("aggregator",    END)

    return workflow.compile()


graph = build_graph()