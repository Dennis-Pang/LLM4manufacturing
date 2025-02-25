from typing_extensions import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
import os
import uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.types import interrupt,Command
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from parameter_recommendator import parameter_recommendation
from online_search import online_search

load_dotenv()
openai_api_key = os.environ["OPENAI_API_KEY"]

llm_openai = ChatOpenAI(
        model="gpt-4o-2024-08-06",
        temperature=0,
        streaming=True
    )

llm_anthropic = ChatAnthropic(model="claude-3-7-sonnet-20250219",
        temperature=0,)

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

# class human_feedback(BaseModel):
#     is_satisfied: bool = Field(
#         None, description="Is the answer satisfactory?"
#     )
#     new_info: str = Field(
#         None, description="If not, please provide additional information."
#     )


@entrypoint(checkpointer=MemorySaver())
def main(query):
    query = rewrite_query(query).result()
    workflow_result, is_successful = router_workflow.invoke(query, config=config)
    if not is_successful:
        print("\n⚠️ Error: Unable to get a valid response")
        return None, False
    return workflow_result, True

def get_valid_query() -> str:
    while True:
        query = input("\n💡 Please enter your query: ")
        if query.strip():
            return query
        print("⚠️  Query cannot be empty")

@task
def rewrite_query(query: str) -> str:
    new_query = llm.invoke(
        [
            SystemMessage(content="""
            Please rewrite the query to be more specific and clear to improve retrieval effectiveness.
            For example:
            Original query: "I wanna turn 1.4125 steel with D10, cutting speed?"
            Rewritten: "What's the cutting speed for turning 1.4125 with D10 tool?"
            
            Ensure the rewritten query includes all necessary details to accurately match relevant information.
            """),
            HumanMessage(content=query),
        ]
    ).content
    return new_query


if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    query = get_valid_query()
    current_input = query
    llm = llm_openai
    for step in main.stream(current_input, config):
        for task_name, event in step.items():
            print(event)


