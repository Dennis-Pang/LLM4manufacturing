from retriever import similarity_search
from rater import rating

def tool_search(llm,query):
    filtered_reference = []
    result = similarity_search(query=query, 
                    file_path="backend\washed_documents\Summurized_Diametal_Turning.md",
                    mapping_file="backend\mappings\table_mappings.json",
                    top_k=5)
    for i, each in enumerate(result):
        feedback = rating(llm,each,query).result()
        if feedback:
            print(f"Chunk {i+1}: ✅ ")
            filtered_reference.append(each)
        else:
            print(f"Chunk {i+1}: ❌ ")
    if len(filtered_reference) != 0:
        return filtered_reference
    else:
        return None

