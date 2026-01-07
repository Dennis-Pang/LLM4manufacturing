"""Relevance rating for retrieved references"""
from langgraph.func import task
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from models import Feedback
from config.prompts import RELEVANCE_RATING_PROMPT


@task
def rate_relevance(llm, reference: str, query: str) -> bool:
    """
    Rate if a reference is relevant to the query.
    Uses dual-evaluator approach for better accuracy.

    Args:
        llm: Primary LLM evaluator
        reference: Retrieved reference text
        query: User query

    Returns:
        True if relevant, False otherwise
    """
    evaluator1 = llm.with_structured_output(Feedback)
    evaluator2 = ChatAnthropic(model="claude-3-5-haiku-20241022").with_structured_output(Feedback)

    def evaluate(evaluator):
        return evaluator.invoke([
            SystemMessage(content=RELEVANCE_RATING_PROMPT),
            HumanMessage(content=f"Query: {query}\nReference: {reference}"),
        ])

    # First evaluation
    decision1 = evaluate(evaluator1)

    if decision1.judge == "relevant":
        return True

    # Second evaluation (only if first says not relevant)
    decision2 = evaluate(evaluator2)

    return decision2.judge == "relevant"
