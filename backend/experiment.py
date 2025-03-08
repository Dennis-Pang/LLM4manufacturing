from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.func import entrypoint, task
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from datetime import datetime
import json
from result_logger import ResultLogger

class MetalAnalysis(BaseModel):
    Carbon_analysis: str = Field(
        None,
        description="Analysis of the carbon content and its impact on cutting speed, cutting depth, and feed rate. For example, high carbon content typically increases hardness, which may require lower parameter values to reduce tool wear."
    )
    Alloying_analysis: str = Field(
        None,
        description="Analysis of the alloying elements and their overall impact on cutting speed, cutting depth, and feed rate, including their role in improving machinability and chip breakage behavior."
    )
    Heat_treatment_analysis: str = Field(
        None,
        description="Analysis of the heat treatment state and base structure, and how these factors influence cutting speed, cutting depth, and feed rate. Emphasize how variations in hardness and toughness from different treatment conditions affect these parameters."
    )
    Thermal_analysis: str = Field(
        None,
        description="Analysis of thermal properties (thermal conductivity, specific heat, and thermal expansion) and their impact on cutting speed, cutting depth, and feed rate, with a focus on heat dissipation and local temperature rise during cutting."
    )
    Chip_formation_analysis: str = Field(
        None,
        description="Analysis of chip formation and breakage characteristics and their effect on cutting speed, cutting depth, and feed rate, discussing how effective chip evacuation can support higher parameter settings, and how poor chip breakage might restrict them."
    )
    Additional_parameters: str = Field(
        None,
        description="Discussion on if any other factors in the documents might also influence the settings for cutting speed, cutting depth and feed rate."
    )

# 加载环境变量
load_dotenv()

def read_file(file_path: str):
    """读取文件内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

model = ChatAnthropic(model="claude-3-7-sonnet-20250219",temperature=0)
model1 = ChatOpenAI(model="gpt-4.5-preview-2025-02-27", temperature=0) #o3-mini-2025-01-31
model2 = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0) #o3-mini-2025-01-31
model0 = ChatOpenAI(model="o3-mini-2025-01-31") #temperature not supported

@task
def metal_analysis(reference_path=r"backend/markdowns/Klein_Metals/1.4125.md", new_metal_path=r"backend/markdowns/Klein_Metals/1.4598.md"):
    """Analyze metal materials."""
    reference = read_file(reference_path)
    new_metal = read_file(new_metal_path)
    messages = [
        SystemMessage(
            content="You are a professional metal materials engineer. Please carefully read the two provided metal material documents, then compare and analyze the key influencing factors to consider when setting machining parameters. Do not provide specific numerical recommendations; this result will serve as a reference for a subsequent LLM."
        ),
        HumanMessage(
            content=f"""
            Reference Document Content:
            {reference}

            Target Document Content:
            {new_metal}

            Based on the above documents, please provide a detailed analysis and comparison from the following aspects:
            1. Metal Composition: Analyze the carbon content and other alloying elements, and discuss how they affect material hardness, machinability, and tool wear.
            2. Heat Treatment State and Base Structure: Analyze how the microstructure under different heat treatment conditions influences cutting forces and surface quality.
            3. Thermal Properties: Discuss the impact of thermal conductivity, specific heat, and thermal expansion on heat dissipation and local temperature rise during the cutting process.
            4. Chip Formation and Evacuation Characteristics: Analyze how chip breakage behavior affects cutting efficiency and tool life.

            Additionally, please discuss whether, besides cutting speed, cutting depth and feed rate are also affected by these factors, and explain the importance of these factors in determining those two parameters.

            Note: The reference metal has established recommended parameters. Based on the similarity and differences between the two metals, please analyze how the parameters for the target metal should be adjusted relative to the reference standard.

            Provide a detailed explanation of the influence of each factor, but do not give specific numerical values.
            """
        )
    ]
    result = model.with_structured_output(MetalAnalysis).invoke(messages)
    return result

class Answer(BaseModel):
    Cutting_speed: float = Field(
        None,
        description="The recommended cutting speed for the new metal."
    )
    Cutting_depth: float = Field(
        None,
        description="The recommended cutting depth for the new metal."
    )
    Feed_rate: float = Field(
        None,
        description="The recommended feed rate for the new metal."
    )
    Thoughts: str = Field(
        None,
        description="The thoughts and reasoning process for the recommended parameters."
    )

# 定义结果存储字典


@task
def estimate_cutting_parameters(compare_result):
    """Estimate cutting parameters based on comparison results."""
    result = model.with_structured_output(Answer).invoke(
        [
            SystemMessage(content="You are a professional metal materials engineer. Please analyze the comparison results of two metal materials and answer the user's question."),
            HumanMessage(content=f"""
            
            The reference metal has an optimal cutting speed of 97m/min (optimal cutting depth and feed rate unknown). The target metal has a recommended cutting speed range of 80-180 (the tool documentation provides ranges for cutting depth and feed rate, which are 0.5-2 and 0.01-0.12 respectively). Based on the comparison of the two metals' compositions, please narrow down this range.
            
            Consider the metal comparison information: {compare_result}

            Balance the following factors:
            1. Tool life
            2. Productivity
            3. Surface roughness

            Provide your recommendations for target metal's all three parameters with explanations.
            """)
        ])
    return result

@entrypoint(checkpointer=MemorySaver())
def main(Q):
    # 初始化结果记录器
    logger = ResultLogger("experiment_results", model)
    
    # 获取比较结果
    compare_result = metal_analysis().result()
    logger.add_result("Metal_analysis", compare_result)
    
    # 获取参数推荐结果
    result = estimate_cutting_parameters(compare_result).result()
    logger.add_result("Answer", result)
    
    # 保存所有结果
    logger.save_results()

    return

config = {
    "configurable": {
        "thread_id": 1
    }
}
for each in main.stream("hah", config):
    print(each)