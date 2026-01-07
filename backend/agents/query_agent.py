"""Query processing agents"""
from langgraph.func import task
from langchain_core.messages import HumanMessage, SystemMessage
from models import NewQueries, Route
from config.prompts import QUERY_REWRITE_PROMPT


@task
def rewrite_query(llm, query: str) -> list[str]:
    """
    Rewrite query to be more specific and split multi-parameter queries.

    Args:
        llm: Language model
        query: Original user query

    Returns:
        List of rewritten queries
    """
    result = llm.with_structured_output(NewQueries).invoke([
        SystemMessage(content=QUERY_REWRITE_PROMPT),
        HumanMessage(content=query),
    ])
    return result.query


@task
def route_query(llm, query: str) -> str:
    """
    Route query to appropriate handler.

    Args:
        llm: Language model
        query: User query

    Returns:
        Route decision (parameter_recommendation, document_extraction, online_search, unknown)
    """
    from models.query_models import QUESTION_TYPES

    decision = llm.with_structured_output(Route).invoke([
        SystemMessage(
            content=f"Route the input to one of these types: {QUESTION_TYPES.__args__} based on the user's request."
        ),
        HumanMessage(content=query),
    ])
    return decision.step
