from typing_extensions import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
import os
import uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langgraph.types import interrupt, Command
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from parameter_recommendator import parameter_recommendation
from online_search import online_search
import json
from datetime import datetime
from result_logger import ResultLogger

################################################################################################################################
load_dotenv()
openai_api_key = os.environ["OPENAI_API_KEY"]

llm_openai = ChatOpenAI(
        model="gpt-4o-2024-08-06",
        temperature=0.7,
        streaming=True
    )

llm_anthropic = ChatAnthropic(model="claude-3-7-sonnet-20250219",
        temperature=0.3,)

llm_deepseek = ChatDeepSeek(model="deepseek-chat",
        temperature=1.0,)

################################################################################################################################

def get_valid_query() -> str:
    while True:
        query = input("\n💡 Please enter your query: ")
        if query.strip():
            return query
        print("⚠️  Query cannot be empty")

class new_queries(BaseModel):
    query: list[str] = Field(
        None, description="The rewritten queries in a list"
    )

@task
def rewrite_query(query: str) -> str:
    new_q = llm.with_structured_output(new_queries).invoke(
        [
            SystemMessage(content="""
            Please rewrite the query to be more specific and clear to improve retrieval effectiveness.
            If multiple parameters are requested in the original query, please split them into separate queries.

            INSTRUCTIONS:
            1. Identify the machining operation, material, tool, and requested parameters
            2. Create one specific query for EACH requested parameter
            3. Return ONLY a properly formatted Python list of strings in a list

            Examples:

            Original query: "I wanna turn 1.4125 steel with D10, cutting speed?"
            Rewritten: "What's the cutting speed for turning 1.4125 steel with D10 tool?"

            Original query: "I wanna turn 1.4125 steel with D10, cutting speed and feed rate?"
            Rewritten: "What's the cutting speed for turning 1.4125 steel with D10 tool?",
            "What's the feed rate for turning 1.4125 steel with D10 tool?"
                                  
            """),
            HumanMessage(content=query),
        ]
    )
    return new_q.query
################################################################################################################################

# Define question typesI
QUESTION_TYPES = Literal[
    "parameter_recommendation",  # For questions about cutting parameters
    "document_extraction",        # For questions about diagrams/images
    "online_search",
    "unknown"             # For general product/company info
]

class Route(BaseModel):
    step: QUESTION_TYPES = Field(
        None, description="Router for different type of questions"
    )


@task
def llm_call_router(query:str):
    decision = llm.with_structured_output(Route).invoke(
        [
            SystemMessage(
                content=f"Route the input to one of these types: {QUESTION_TYPES.__args__} based on the user's request."
            ),
            HumanMessage(content=query),
        ]
    )
    return decision.step
                
# Create workflow
@entrypoint(checkpointer=MemorySaver())
def router_workflow(query: str):
        
    next_step = llm_call_router(query).result()
    print(f"\n🎯 Router leads to: {next_step}\n")
    
    if next_step == "parameter_recommendation":
        response = parameter_recommendation(llm,query).result()
        return response, True

    elif next_step == "document_extraction":
        return "Picture reference feature not implemented yet."

    elif next_step == "online_search":
        try:
            return online_search(llm,query),True
        except Exception as e:
            return "Error occurred in online search: " + str(e),False

    elif next_step == "unknown":
        return "Unknown question type: " + query
################################################################################################################################
@entrypoint(checkpointer=MemorySaver())
def RAG(query):
    # 初始化结果记录器
    logger = ResultLogger("rag_logs", llm_openai)

    logger.add_result("Original Query", query)
    # 获取重写后的查询列表
    queries = rewrite_query(query).result()
    logger.add_result("Rewritten Queries", queries)  # 记录重写后的查询
    
    # 处理每个查询
    for each_query in queries:
        try:
            # 执行处理流程
            workflow_result, is_successful = router_workflow.invoke(each_query, config=config)
            
            # 存储结果
            logger.add_result(each_query, {
                "result": workflow_result,
                "is_successful": is_successful
            })
            
            if not is_successful:
                print(f"\n⚠️ warning: unable to get a valid response for the query: {each_query}")
                continue
                
        except Exception as e:
            print(f"\n⚠️ warning: error occurred when processing the query: {each_query}: {str(e)}")
            logger.add_result(each_query, {
                "result": str(e),
                "is_successful": False
            })
            continue

    # 保存所有结果
    logger.save_results()

    return

################################################################################################################################

# def log_rag_results(query: str, workflow_results: tuple, llm) -> str:
#     """记录RAG查询和结果到JSON文件
#     Args:
#         query: 原始查询
#         workflow_results: RAG处理的结果元组 (results_dict, is_successful)
#         llm: 使用的语言模型
#     Returns:
#         str: 日志文件路径
#     """
#     log_dir = "rag_logs"
#     os.makedirs(log_dir, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M")
#     log_file = os.path.join(log_dir, f"rag_{timestamp}.json")
    
#     results_dict, _ = workflow_results
    
#     log_data = {
#         "original_query": query,
#         "rewritten_queries": list(results_dict.keys()),  # 使用结果字典的键作为重写的查询列表
#         "timestamp": timestamp,
#         "model": {
#             "name": llm.__class__.__name__,
#             "model": getattr(llm, 'model', "unknown"),
#             "temperature": getattr(llm, 'temperature', "unknown")
#         },
#         "results": {}
#     }
    
#     for query_text, query_result in results_dict.items():
#         if isinstance(query_result, dict) and 'result' in query_result:
#             result = query_result['result']
#             if hasattr(result, 'questioned_parameter'):
#                 log_data["results"][query_text] = {
#                     "parameter": result.questioned_parameter,
#                     "tool_range": result.tool_range,
#                     "metal_range": result.metal_range if hasattr(result, 'metal_range') else "None",
#                     "combined_range": result.combined_range if hasattr(result, 'combined_range') else result.tool_range,
#                     "thoughts": result.thoughts
#                 }
#             else:
#                 log_data["results"][query_text] = {
#                     "parameter": "General Information",
#                     "tool_range": "N/A",
#                     "metal_range": "N/A",
#                     "combined_range": "N/A",
#                     "thoughts": str(result)
#                 }
    
#     with open(log_file, "w", encoding="utf-8") as f:
#         json.dump(log_data, f, ensure_ascii=False, indent=2)
    
#     print(f"\n💾 Results saved to: {log_file}")
#     return log_file

################################################################################################################################
if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    query = get_valid_query()
    llm = llm_openai
    
    # 获取完整结果
    results = RAG.invoke(query, config)
    
    # 显示流式输出
    for step in RAG.stream(query, config, stream_mode="updates"):
        for _, event in step.items():
            print(event)


