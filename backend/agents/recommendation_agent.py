"""Parameter recommendation agent"""
from langgraph.func import task
from langchain_core.messages import HumanMessage, SystemMessage
from models import Check, Answer
from config.prompts import FACTOR_CHECK_PROMPT, PARAMETER_RECOMMEND_PROMPT
from retrieval import fuzzy_match_metal, search_tool_references


@task
def check_factors(llm, query: str) -> Check:
    """
    Check if query contains all necessary factors.

    Args:
        llm: Language model
        query: User query

    Returns:
        Check result with extracted factors
    """
    result = llm.with_structured_output(Check).invoke([
        SystemMessage(content=FACTOR_CHECK_PROMPT),
        HumanMessage(content=f"Query: {query}")
    ])
    return result


@task
def recommend_parameters(llm, query: str) -> Answer:
    """
    Main parameter recommendation function.

    Args:
        llm: Language model
        query: User query

    Returns:
        Parameter recommendation answer
    """
    # Check factors
    check = check_factors(llm, query).result()
    if check.judge == "no":
        print("⚠️ Please provide a complete query with operation, metal and tool information.")
        return None

    print("-- Start parameter recommendation:\n")

    # Fuzzy match metal
    metal_name = check.metal
    _, doc_path, _ = fuzzy_match_metal(metal_name).result()

    # Read metal document
    metal_doc = None
    if doc_path:
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                metal_doc = f.read()
        except FileNotFoundError:
            print(f"⚠️ No metal references found for {metal_name}")

    # Search tool references
    tool_refs = search_tool_references(llm, query)
    if tool_refs is None:
        print("⚠️ No valid tool references found.")

    # Merge references
    references = [metal_doc] if metal_doc else []
    references.extend(tool_refs or [])

    # Generate recommendation
    messages = [
        SystemMessage(content=PARAMETER_RECOMMEND_PROMPT),
        HumanMessage(content=f"Query: {query}\nReferences: {references}")
    ]

    response = llm.with_structured_output(Answer).invoke(messages)

    # Print formatted output
    output = f"""
    🔍 Questioned parameter: {response.questioned_parameter}
    🔧 Metal's source: {response.metal_range}
    🛠️  Tool's source: {response.tool_range}
    🎯 Combined range: {response.combined_range}
    💭 RagBot's thoughts: {response.thoughts}
    """
    print("\n🤖 RagBot's Answer:\n", output)

    return response
