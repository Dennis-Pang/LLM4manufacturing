"""Vector store operations"""
import os
import re
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def load_table_from_mapping(mapping_file: str, table_id: int) -> str:
    """Load original table content from mapping file"""
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mappings = json.load(f)

    if 0 <= table_id < len(mappings):
        return mappings[table_id].get("original_table")
    else:
        raise ValueError(f"Table with id {table_id} not found")


def detect_table_markers(text: str) -> list[int]:
    """
    Detect all table markers in text.
    Example: "__TABLE3__" returns [3]
    """
    markers = re.findall(r"__TABLE(\d+)__", text)
    return [int(m) for m in markers]


def similarity_search(
    query: str,
    file_path: str,
    mapping_file: str,
    top_k: int = 6
) -> list[str]:
    """
    Perform similarity search and restore table content.

    Args:
        query: Search query
        file_path: Path to the original document (used to derive DB name)
        mapping_file: Path to table mappings JSON
        top_k: Number of results to return

    Returns:
        List of reference texts with restored table content
    """
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # Build persist directory and collection name
    persist_directory = os.path.join("VectorDBs", filename)
    collection_name = f"rag-{filename}"

    # Load vector store
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print(f"🔹 VectorDB for {filename} loaded")

    # Search
    results = vectorstore.similarity_search(query, k=top_k)
    references = []

    print(f"🔹 Top {len(results)} most related chunks:")

    for i, doc in enumerate(results):
        chunk = doc.page_content
        print(f"\n🔹 Processing chunk {i+1}:")
        print(f"Text: {chunk[:100]}...")

        # Detect and restore tables
        markers = detect_table_markers(chunk)
        info = chunk

        if markers:
            print(f"🔹 Found table markers: {markers}")
            table_contents = []
            for table_id in markers:
                original_table = load_table_from_mapping(mapping_file, table_id)
                if original_table:
                    table_contents.append(original_table)

            if table_contents:
                info = chunk + " " + " ".join(table_contents)

        references.append(info)

    return references
