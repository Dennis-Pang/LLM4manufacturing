import os
import re
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def load_and_get_table(mapping_file, table_id):
    # use os.path.join to build the path
    mapping_file_path = os.path.join('backend', 'mappings', 'table_mappings.json')
    
    with open(mapping_file_path, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
        
    if 0 <= table_id < len(mappings):
        return mappings[table_id].get("original_table")
    else:
        raise ValueError(f"Table with id {table_id} not found")
        
def detect_table_markers(text):
    """
    detect all table markers in the text, for example "__TABLE3__" returns [3]
    """
    # use the regular expression to find the markers like "__TABLE\d+__"
    markers = re.findall(r"__TABLE(\d+)__", text)
    # convert the string numbers to int
    return [int(m) for m in markers]

def similarity_search(query: str, 
                     file_path: str,# same as the file_path in markdown2embedding.py
                     mapping_file: str,
                     top_k: int = 6):
    
    filename = os.path.splitext(os.path.basename(file_path))[0]
    
    # build the persistent directory and collection name
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
        print(f"\n🔹 Processing the {i+1}th result:")  # debug information
        print(f"Text content: {chunk[:100]}...")  # display the first 100 characters
        
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
