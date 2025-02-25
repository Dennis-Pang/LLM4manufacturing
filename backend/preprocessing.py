import re
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json

load_dotenv()
openai_api_key = os.environ["OPENAI_API_KEY"]

def LLM_summary_tables(prompt: str, table_id: int) -> str:
    sys_prompt = (
        """
        You are an expert in manufacturing. 

        Your task is to **summarize the content** of the given table in a **concise** and **easy-to-understand** manner.

        - **Do not** include exact data values or detailed descriptions from the table.
        - **Do** mention all important topics, categories, or items the table contains.
        - Keep the summary **brief** but **comprehensive** enough so that an LLM can grasp the key points.

        For example, if the table is about cutting speeds for different materials, a suitable summary would be:
        "The table lists cutting speeds for various materials, including A, B, and C."

        """
    )
    
    combined_prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("user", "{prompt}")
    ])
    
    chat_model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )
    
    pipeline = combined_prompt | chat_model
    
    result = pipeline.invoke(prompt)

    return {
        'table_id': table_id,
        'summary':result.content,
        'original_table':prompt
        }

def replace_tables(md_text, json_path="processed_info\table_mappings.json"):
    table_pattern = re.compile(r"(<table.*?</table>)", re.DOTALL)
    tables = table_pattern.findall(md_text)
    # 用占位符替换表格，防止误删
    table_mappings = []
    modified_text = md_text
    for i, table in enumerate(tables):
        table_info = LLM_summary_tables(table, i)
        placeholder = f"__TABLE{i}__:{table_info['summary']}"
        table_mappings.append(table_info)
        modified_text = modified_text.replace(table, placeholder)
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(table_mappings, json_file, ensure_ascii=False, indent=4)
    print("Json is done")
    return modified_text

def remove_images(md_text):
    """去除 Markdown 图片（![alt](url) 和 <img>）"""
    md_text = re.sub(r'!\[.*?\]\(.*?\)', '', md_text)  # 删除 ![描述](url)
    md_text = re.sub(r'<img.*?>', '', md_text)  # 删除 <img> 标签
    return md_text


input_md_file = "Diametal_Turning.md"  
output_md_file = "step0.md"
with open(input_md_file, 'r', encoding='utf-8') as f:
    md_text = f.read()

md_text = remove_images(md_text)
md_text = replace_tables(md_text)

with open(output_md_file, 'w', encoding='utf-8') as f:
    f.write(md_text)