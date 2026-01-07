"""Tool reference retrieval"""
from retrieval.vector_store import similarity_search
from retrieval.relevance_rater import rate_relevance


def search_tool_references(llm, query: str) -> list[str] | None:
    """
    Search for tool-related references and filter by relevance.

    Args:
        llm: LLM for relevance rating
        query: User query

    Returns:
        List of relevant references, or None if none found
    """
    filtered_references = []

    # Search vector database
    results = similarity_search(
        query=query,
        file_path="washed_documents/Summurized_Diametal_Turning.md",
        mapping_file="mappings/table_mappings.json",
        top_k=5
    )

    # Filter by relevance
    for i, result in enumerate(results):
        is_relevant = rate_relevance(llm, result, query).result()
        if is_relevant:
            print(f"Chunk {i+1}: ✅")
            filtered_references.append(result)
        else:
            print(f"Chunk {i+1}: ❌")

    return filtered_references if filtered_references else None
