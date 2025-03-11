from pydantic import BaseModel, Field
from typing_extensions import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.func import task
from langchain_anthropic import ChatAnthropic

class Feedback(BaseModel):
    thought: str = Field(
        description="Explain your reasoning for the decision.",
    )
    judge: Literal["relevant","not relevant"] = Field(
        None, description="Decide if the reference is relevant or not."
    )


@task
def rating(llm, reference: str, query: str):
    evaluator1 = llm.with_structured_output(Feedback)
    evaluator2 = ChatAnthropic(model="claude-3-5-haiku-20241022").with_structured_output(Feedback)

    content = """
        As a manufacturing expert, your task is to evaluate if a reference matches a user's query about machining tools.

        Evaluation Rules:
        1. Focus ONLY on three elements in the query:
           - Tool name (from: HM Carbide, D10, D20, D60, Cermet, PKD/PCD)
           - Machining operation (e.g., turning, milling, drilling, etc.)
           - Questioned parameters (e.g., cutting speed, feed rate)
        
        2. IGNORE all material/metal specifications in the query

        Example:
        Query: "What's the cutting speed for turning ABC-1234 steel with D10?"
        Analysis:
        - Tool: D10 ✓
        - Operation: turning ✓
        - Questioned parameters: cutting speed ✓
        - Material (ABC-1234): ignore this

        Your task: Determine if the provided reference contains relevant information to answer the query, focusing only on the tool and operation.

        Please provide a very brief explanation for your decision.
    """
    
    def evaluate(evaluator, **kwargs):
        return evaluator.invoke(
            [
                SystemMessage(content=content),
                HumanMessage(content=f"Query: {query}\nReference: {reference}"),
            ],
            **kwargs
        )
    
    decision1 = evaluate(evaluator1)
    
    if decision1.judge == "relevant":
        return True ##  if the first evaluation result is "relevant", return True
        
    # only run the second evaluation when the first evaluation result is "not relevant"
    decision2 = evaluate(evaluator2)
    
    return decision2.judge == "relevant"
