from tavily import TavilyClient
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

client = TavilyClient("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")


def search_online(query: str):
    response = client.search(
        query=query
    )
    
    search_results = response.get("results", [])
    
    summary_text = f"Original question: {query}\n\nHere are the search results summary:\n\n"
    
    for i, result in enumerate(search_results, 1):
        title = result.get("title", "No title")
        content = result.get("content", "No content") 
        url = result.get("url", "No link")
        
        summary_text += f"{i}. {title}\n"
        summary_text += f"   Link: {url}\n"
        summary_text += f"   Content summary: {content[:200]}...\n\n"
    
    return summary_text, query
    
@tool
def online_search(llm, query):
    """
    use online search to get the related information of the query.
    
    Args:
        llm: the language model used to process the search results
        query: the query string to search
    
    Returns:
        str: the summary of the search results
    """
    summary_text, query = search_online(query)
    prompt = f"Question: {query}\n\n{summary_text}\n\nBased on the search results above, please answer the original question."
        
    response = llm.invoke(
        [   SystemMessage(content="""
            You are an expert in manufacturing. You are helpful assistant that can answer questions based on the provided search results.
            Summarize the search results and provide a brief and concise answer to the original question.
            """),
            HumanMessage(content=prompt)
        ]
    )
    return response.content

