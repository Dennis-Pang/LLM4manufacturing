"""System prompts for different agents"""
from config.prompts.query_rewrite import QUERY_REWRITE_PROMPT
from config.prompts.factor_check import FACTOR_CHECK_PROMPT
from config.prompts.parameter_recommend import PARAMETER_RECOMMEND_PROMPT
from config.prompts.relevance_rating import RELEVANCE_RATING_PROMPT

__all__ = [
    "QUERY_REWRITE_PROMPT",
    "FACTOR_CHECK_PROMPT",
    "PARAMETER_RECOMMEND_PROMPT",
    "RELEVANCE_RATING_PROMPT",
]
