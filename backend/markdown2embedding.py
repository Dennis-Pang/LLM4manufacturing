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
    split the text by table markers, ensuring that the part starting with __TABLE\d+__ and its subsequent content (until the next marker or end) are kept in the same chunk.
    for example:
      "Some text... __TABLE9__:The table presents cutting speeds... More text..."
    return:
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
    smart Chunking:
      - first use split_by_tables_combined() to ensure the table blocks are complete;
      - for non-table parts, if the token number exceeds max_tokens, further split into multiple sub-blocks.
    """
    initial_chunks = split_by_tables_combined(text)
    final_chunks = []
    for chunk in initial_chunks:
        # if the chunk starts with __TABLE\d+__, keep it
        if re.match(r"^__TABLE\d+__", chunk):
            final_chunks.append(chunk)
        else:
            tokens = tokenizer.encode(chunk)
            if len(tokens) <= max_tokens:
                final_chunks.append(chunk)
            else:
                # split into multiple sub-blocks by max_tokens
                for i in range(0, len(tokens), max_tokens):
                    sub_tokens = tokens[i:i+max_tokens]
                    sub_chunk = tokenizer.decode(sub_tokens)
                    final_chunks.append(sub_chunk)
    return final_chunks

def create_vector_DB(file_path):
    # extract the filename from the file path
    filename = os.path.splitext(os.path.basename(file_path))[0]
    
    # build the persistent directory and collection name
    persist_directory = os.path.join("backend\\VectorDBs", filename)
    collection_name = f"rag-{filename}"

    # check if the vector database already exists
    if os.path.exists(persist_directory):
        print(f"🔹 Vector database already exists at {persist_directory}, skipping creation.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        text_content = f.read()
        
    chunks = smart_chunking(text_content)
    print(f"🔹 Got {len(chunks)} chunks without breaking a table.")

    # convert the string chunks into Document objects
    docs = [Document(page_content=chunk) for chunk in chunks]

    # create the vector database, automatically calculate embeddings and persist to the specified directory
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


