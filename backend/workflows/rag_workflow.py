"""Main RAG workflow"""
import uuid
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import MemorySaver
from agents.query_agent import rewrite_query, route_query
from agents.recommendation_agent import recommend_parameters
from utils import ResultLogger


@entrypoint(checkpointer=MemorySaver())
def process_single_query(llm, query: str):
    """
    Process a single query through the RAG workflow.

    Args:
        llm: Language model
        query: User query

    Returns:
        Tuple of (response, is_successful)
    """
    next_step = route_query(llm, query).result()
    print(f"\n🎯 Router leads to: {next_step}\n")

    if next_step == "parameter_recommendation":
        response = recommend_parameters(llm, query).result()
        return response, True

    elif next_step == "document_extraction":
        return "Picture reference feature not implemented yet.", False

    elif next_step == "online_search":
        return "Online search feature moved to legacy.", False

    elif next_step == "unknown":
        return f"Unknown question type: {query}", False


@entrypoint(checkpointer=MemorySaver())
def rag_pipeline(llm, query: str):
    """
    Complete RAG pipeline with query rewriting and logging.

    Args:
        llm: Language model
        query: Original user query
    """
    # Initialize logger
    logger = ResultLogger("rag_logs", llm)
    logger.add_result("Original Query", query)

    # Rewrite query
    queries = rewrite_query(llm, query).result()
    logger.add_result("Rewritten Queries", queries)

    # Process each rewritten query
    for each_query in queries:
        try:
            # Create config for workflow
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            # Execute workflow
            workflow_result, is_successful = process_single_query.invoke(
                (llm, each_query),
                config=config
            )

            # Log result
            logger.add_result(each_query, {
                "result": workflow_result,
                "is_successful": is_successful
            })

            if not is_successful:
                print(f"\n⚠️ Warning: Unable to get valid response for: {each_query}")

        except Exception as e:
            print(f"\n⚠️ Error processing query '{each_query}': {str(e)}")
            logger.add_result(each_query, {
                "result": str(e),
                "is_successful": False
            })

    # Save all results
    logger.save_results()
