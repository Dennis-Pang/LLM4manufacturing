import os
import re
import tiktoken
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# multi-language suported by text-embedding-ada-002 
tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")

def split_by_tables_combined(text):
    """
    将文本按表格标记拆分，确保以 __TABLE\d+__ 开头的部分及其后续内容（直到下一个标记或结尾）
    保持在同一个 chunk 中。
    例如：
      "Some text... __TABLE9__:The table presents cutting speeds... More text..."
    返回：
      ["Some text...", "__TABLE9__:The table presents cutting speeds...", " More text..."]
    """
    pattern = r"(__TABLE\d+__[:\s]*)"
    parts = re.split(pattern, text)
    chunks = []
    i = 0
    if parts and not re.match(pattern, parts[0]):
        chunks.append(parts[0])
        i = 1
    while i < len(parts):
        if i + 1 < len(parts):
            combined = parts[i] + parts[i+1]
            chunks.append(combined)
            i += 2
        else:
            chunks.append(parts[i])
            i += 1
    return chunks

def smart_chunking(text, max_tokens=1000):
    """
    智能 Chunking：
      - 先使用 split_by_tables_combined() 保证表格块完整；
      - 对非表格部分，如果 token 数超过 max_tokens，则进一步拆分成多个子块。
    """
    initial_chunks = split_by_tables_combined(text)
    final_chunks = []
    for chunk in initial_chunks:
        # 如果 chunk 是以 __TABLE\d+__ 开头（表格块），直接保留
        if re.match(r"^__TABLE\d+__", chunk):
            final_chunks.append(chunk)
        else:
            tokens = tokenizer.encode(chunk)
            if len(tokens) <= max_tokens:
                final_chunks.append(chunk)
            else:
                # 按 max_tokens 拆分为多个子块
                for i in range(0, len(tokens), max_tokens):
                    sub_tokens = tokens[i:i+max_tokens]
                    sub_chunk = tokenizer.decode(sub_tokens)
                    final_chunks.append(sub_chunk)
    return final_chunks

def create_vector_DB(file_path):
    # 从文件路径中提取文件名
    filename = os.path.splitext(os.path.basename(file_path))[0]
    
    # 构建持久化目录和集合名称
    persist_directory = os.path.join("backend\\VectorDBs", filename)
    collection_name = f"rag-{filename}"

    # 检查向量数据库是否已存在
    if os.path.exists(persist_directory):
        print(f"🔹 Vector database already exists at {persist_directory}, skipping creation.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        text_content = f.read()
        
    chunks = smart_chunking(text_content)
    print(f"🔹 Got {len(chunks)} chunks without breaking a table.")

    # 将字符串 chunk 转换成 Document 对象
    docs = [Document(page_content=chunk) for chunk in chunks]

    # 创建向量数据库，自动计算 embeddings 并持久化到指定目录
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(openai_api_key=openai_api_key),
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    vectorstore.persist()
    print(f"🔹 Vector database has been successfully persisted, saved to {persist_directory}.")

washed_doc_path = ["backend\\washed_documents\\Summurized_Diametal_Turning.md",]

for file_path in washed_doc_path:
    create_vector_DB(file_path)


