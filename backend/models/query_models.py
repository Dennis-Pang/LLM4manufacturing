"""Query related Pydantic models"""
from pydantic import BaseModel, Field
from typing_extensions import Literal


class NewQueries(BaseModel):
    """Rewritten queries model"""
    query: list[str] = Field(
        None, description="The rewritten queries in a list"
    )


# Define question types
QUESTION_TYPES = Literal[
    "parameter_recommendation",  # For questions about cutting parameters
    "document_extraction",        # For questions about diagrams/images
    "online_search",
    "unknown"                     # For general product/company info
]


class Route(BaseModel):
    """Router decision model"""
    step: QUESTION_TYPES = Field(
        None, description="Router for different type of questions"
    )
