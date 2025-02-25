import os
import re
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def load_and_get_table(mapping_file, table_id):
    # 使用 os.path.join 来构建路径
    mapping_file_path = os.path.join('backend', 'mappings', 'table_mappings.json')
    
    with open(mapping_file_path, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
        
    if 0 <= table_id < len(mappings):
        return mappings[table_id].get("original_table")
    else:
        raise ValueError(f"Table with id {table_id} not found")
        
def detect_table_markers(text):
    """
    检测文本中所有表格标记，例如 "__TABLE3__" 返回 [3]
    """
    # 使用正则表达式查找形如 __TABLE\d+__ 的标记
    markers = re.findall(r"__TABLE(\d+)__", text)
    # 将字符串数字转换为 int
    return [int(m) for m in markers]

def similarity_search(query: str, 
                     file_path: str,# same as the file_path in markdown2embedding.py
                     mapping_file: str,
                     top_k: int = 6):
    
    filename = os.path.splitext(os.path.basename(file_path))[0]
    
    # 构建持久化目录和集合名称
    persist_directory = os.path.join("backend\VectorDBs", filename)
    collection_name = f"rag-{filename}"

    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key),
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print(f"🔹 VectorDB for {filename} loaded")
    
    results = vectorstore.similarity_search(query, k=top_k)
    
    references = []
    
    print(f"🔹 Top {len(results)} most related chunks：")
    
    for i, doc in enumerate(results):
        chunk = doc.page_content
        print(f"\n🔹 Processing the {i+1}th result:")  # 调试信息
        print(f"Text content: {chunk[:100]}...")  # 显示前100个字符
        
        markers = detect_table_markers(chunk)
        info = chunk
        
        if markers:
            print("🔹 This chunk contains table markers:", markers)
            table_contents = []
            for table_id in markers:
                original_table = load_and_get_table(mapping_file, table_id)
                if original_table:
                    table_contents.append(original_table)
            
            if table_contents:
                info = chunk + " " + " ".join(table_contents)
        
        references.append(info)
    
    return references

# result = similarity_search(query="What's the cutting speed for D10?", 
#                      file_path="washed_documents\Summurized_Diametal_Turning.md",
#                      mapping_file=r"mappings\table_mappings.json",
#                      top_k=3)
