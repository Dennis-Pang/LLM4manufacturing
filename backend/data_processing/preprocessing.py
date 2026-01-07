"""Document preprocessing functions"""
import re
import json
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def llm_summary_tables(table_content: str, table_id: int) -> dict:
    """Summarize table content using LLM"""
    sys_prompt = """
    You are an expert in manufacturing.

    Your task is to **summarize the content** of the given table in a **concise** and **easy-to-understand** manner.

    - **Do not** include exact data values or detailed descriptions from the table.
    - **Do** mention all important topics, categories, or items the table contains.
    - Keep the summary **brief** but **comprehensive** enough so that an LLM can grasp the key points.

    For example, if the table is about cutting speeds for different materials, a suitable summary would be:
    "The table lists cutting speeds for various materials, including A, B, and C."
    """

    combined_prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("user", "{prompt}")
    ])

    chat_model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    pipeline = combined_prompt | chat_model
    result = pipeline.invoke(table_content)

    return {
        'table_id': table_id,
        'summary': result.content,
        'original_table': table_content
    }


def replace_tables(md_text: str, json_path: str = "mappings/table_mappings.json") -> str:
    """Replace tables with placeholders and save mappings"""
    table_pattern = re.compile(r"(<table.*?</table>)", re.DOTALL)
    tables = table_pattern.findall(md_text)

    table_mappings = []
    modified_text = md_text

    for i, table in enumerate(tables):
        table_info = llm_summary_tables(table, i)
        placeholder = f"__TABLE{i}__:{table_info['summary']}"
        table_mappings.append(table_info)
        modified_text = modified_text.replace(table, placeholder)

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(table_mappings, json_file, ensure_ascii=False, indent=4)

    print(f"✅ Table mappings saved to {json_path}")
    return modified_text


def remove_images(md_text: str) -> str:
    """Remove images from markdown text"""
    md_text = re.sub(r'!\[.*?\]\(.*?\)', '', md_text)  # Remove ![alt](url)
    md_text = re.sub(r'<img.*?>', '', md_text)  # Remove <img> tags
    return md_text


def preprocess_document(input_path: str, output_path: str, mappings_path: str = None):
    """Complete preprocessing pipeline"""
    with open(input_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # Remove images
    md_text = remove_images(md_text)

    # Replace tables
    if mappings_path is None:
        mappings_path = "mappings/table_mappings.json"
    md_text = replace_tables(md_text, mappings_path)

    # Save preprocessed document
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_text)

    print(f"✅ Preprocessed document saved to {output_path}")
    return md_text
