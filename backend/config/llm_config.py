"""LLM configuration"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek

load_dotenv()

# Initialize LLMs
llm_openai = ChatOpenAI(
    model="gpt-4o-2024-08-06",
    temperature=0.7,
    streaming=True
)

llm_anthropic = ChatAnthropic(
    model="claude-3-7-sonnet-20250219",
    temperature=0.3,
)

llm_deepseek = ChatDeepSeek(
    model="deepseek-chat",
    temperature=1.0,
)

# Default LLM
DEFAULT_LLM = llm_openai
