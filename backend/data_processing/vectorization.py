"""Document vectorization and chunking"""
import os
import re
import tiktoken
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

load_dotenv()

# Tokenizer for text-embedding-ada-002
tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")


def split_by_tables(text: str) -> list[str]:
    """
    Split text by table markers, ensuring table content stays intact.

    Example:
      "Some text... __TABLE9__:summary... More text..."
    Returns:
      ["Some text...", "__TABLE9__:summary...", " More text..."]
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


def smart_chunking(text: str, max_tokens: int = 1000) -> list[str]:
    """
    Smart chunking that preserves table integrity.

    - First split by table markers
    - For non-table parts, split by max_tokens if needed
    """
    initial_chunks = split_by_tables(text)
    final_chunks = []

    for chunk in initial_chunks:
        # Keep table chunks intact
        if re.match(r"^__TABLE\d+__", chunk):
            final_chunks.append(chunk)
        else:
            tokens = tokenizer.encode(chunk)
            if len(tokens) <= max_tokens:
                final_chunks.append(chunk)
            else:
                # Split into sub-chunks by max_tokens
                for i in range(0, len(tokens), max_tokens):
                    sub_tokens = tokens[i:i+max_tokens]
                    sub_chunk = tokenizer.decode(sub_tokens)
                    final_chunks.append(sub_chunk)

    return final_chunks


def create_vector_db(file_path: str, persist_directory: str = None, collection_name: str = None):
    """Create vector database from document"""
    # Extract filename
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # Set default paths
    if persist_directory is None:
        persist_directory = os.path.join("VectorDBs", filename)
    if collection_name is None:
        collection_name = f"rag-{filename}"

    # Check if already exists
    if os.path.exists(persist_directory):
        print(f"🔹 Vector database already exists at {persist_directory}, skipping creation.")
        return

    # Read document
    with open(file_path, "r", encoding="utf-8") as f:
        text_content = f.read()

    # Chunk document
    chunks = smart_chunking(text_content)
    print(f"🔹 Created {len(chunks)} chunks (tables preserved)")

    # Convert to Document objects
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Create vector database
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    vectorstore.persist()
    print(f"✅ Vector database saved to {persist_directory}")
