"""Main entry point for RAG system"""
import uuid
from config.llm_config import DEFAULT_LLM, llm_openai, llm_anthropic, llm_deepseek
from workflows import rag_pipeline


def get_valid_query() -> str:
    """Get non-empty query from user"""
    while True:
        query = input("\n💡 Please enter your query: ")
        if query.strip():
            return query
        print("⚠️  Query cannot be empty")


if __name__ == "__main__":
    # Configuration
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Get user query
    query = get_valid_query()

    # Select LLM (default: OpenAI GPT-4o)
    llm = llm_openai

    # Run RAG pipeline
    print("\n🚀 Starting RAG pipeline...\n")
    rag_pipeline.invoke((llm, query), config=config)

    print("\n✅ RAG pipeline completed!")
