"""Parameter recommendation related Pydantic models"""
from pydantic import BaseModel, Field
from typing_extensions import Literal


class Check(BaseModel):
    """Factor check result model"""
    judge: Literal["yes", "no"] = Field(
        description="Decide if all necessary factors are fully included in the query.",
    )
    tool: str = Field(
        description="The tool name if it exists in the query, None if not.",
    )
    metal: str = Field(
        description="The metal name if it exists in the query, None if not.",
    )
    operation: str = Field(
        description="The operation name if it exists in the query, None if not.",
    )
    questioned_parameters: str = Field(
        description="The questioned parameters if it exists in the query, None if not.",
    )


class Answer(BaseModel):
    """Parameter recommendation answer model"""
    questioned_parameter: str = Field(
        description="The questioned parameter",
    )
    tool_range: str = Field(
        description="The range of tool's parameter recommendation, None if not.",
    )
    metal_range: str = Field(
        description="The range of metal's parameter recommendation, None if not.",
    )
    combined_range: str = Field(
        description="The combined source of tool and metal's parameter recommendation, 'conflicted' if conflicted parameters between tool and metal.",
    )
    thoughts: str = Field(
        description="""
        Some brief internal thought process of parameter recommendation in 2-3 sentences,
        if there is no conflicted parameters between tool and metal;
        If there is conflicted parameters between tool and metal, give a brief explanation and make a suggestion based both sources and your own knowledge.
        """,
    )


class Feedback(BaseModel):
    """Relevance rating feedback model"""
    thought: str = Field(
        description="Explain your reasoning for the decision.",
    )
    judge: Literal["relevant", "not relevant"] = Field(
        None, description="Decide if the reference is relevant or not."
    )
