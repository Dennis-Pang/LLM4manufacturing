from langgraph.func import task

import json
from rapidfuzz import fuzz

# class MetalName(BaseModel):
#     metal_name: str = Field(
#         description="The identified metal name from the query"
#     )

# @task
# def extract_metal_name(llm, query: str):
#     """
#     Extract metal name from the query using LLM.
    
#     For specific metal codes (e.g., CCR-1150, TI-64), extract them exactly; 
#     for generic metals (e.g., titanium alloy, stainless steel), use your domain knowledge.
#     If both are present, prioritize the specific code.
#     If no metal is found, return "UNKNOWN" as metal_name.
    
#     Returns:
#         MetalName: The extracted metal name
#     """
#     system_instructions = """
#     You are a metallurgy expert. Your task is to extract the metal information from a given query.
    
#     Rules:
#     1. If a specific metal code is mentioned in the query (e.g., CCR-1150, TI-64), extract it exactly.
#     2. For generic metals (e.g., titanium alloy, stainless steel), identify them using your domain knowledge.
#     3. If both a specific code and a generic metal are present, prioritize the specific code.
#     4. If no metal is found, return "UNKNOWN" as metal_name.
#     5. ***Ignore any tool names include: HM Carbide, D10, D20, D60, Cermet, PKD/PCD.***
    
#     Examples:
#       - "using BT-30 for turning" -> metal_name: "BT-30"
#       - "machining stainless steel" -> metal_name: "STAINLESS STEEL"
#     """
    
#     metal_classifier = llm.with_structured_output(MetalName)
    
#     result = metal_classifier.invoke(
#         [
#             SystemMessage(content=system_instructions),
#             HumanMessage(content=f"Query: {query}")
#         ]
#     )

#     return result.metal_name

@task
def fuzzy_match_metal(query: str, metal_mapping_path=r"backend\mappings\metal_mappings.json", threshold: int = 80):
    """
    fuzzy search for the metal name and its aliases in metal_data.
    
    :param query: the metal string input by the user, for example "1.4125", "CCR1150" etc.
    :param metal_data: the metal data loaded from the JSON file, the format is like:
        {
          "CHRONIFER M-17C": {
            "aliases": ["1.4125", "AISI 440C", "X105CrMo17", "SUS440C"],
            "doc_path": "markdowns/Klein_Metals/CCR-1150.md"
          },
          "CCR-1150": {
            "aliases": ["CCR1150", "CCR 1150"],
            "doc_path": "markdowns/Klein_Metals/1.4125.md"
          }
        }
    :param threshold: the threshold of fuzzy matching score, default is 80.
    :return: if the match is successful, return (main name, doc_path, score); otherwise return (None, None, 0).
    """
    best_score = 0
    best_main_name = None
    best_doc_path = None
    matched_key = None

    with open(metal_mapping_path, "r", encoding="utf-8") as f:
        metal_data = json.load(f)

    query_norm = query.strip().lower()

    for main_name, info in metal_data.items():
        # 1. fuzzy match with the main name
        main_name_norm = main_name.strip().lower()
        score = fuzz.ratio(query_norm, main_name_norm)
        if score > best_score:
            best_score = score
            best_main_name = main_name
            matched_key = main_name

        # 2. fuzzy match with each alias
        for alias in info.get("aliases", []):
            alias_norm = alias.strip().lower()
            score = fuzz.ratio(query_norm, alias_norm)
            if score > best_score:
                best_score = score
                best_main_name = main_name
                matched_key = main_name

    if best_score >= threshold and matched_key:
        best_doc_path = metal_data[matched_key].get("doc_path")
        return best_main_name, best_doc_path, best_score
    else:
        return None, None, 0

# if __name__ == "__main__":
#     # specify the JSON file path of metal_data (recommended to use raw string or forward slash)
#     json_path = r"mappings\metal_mappings.json"
#     # or: json_path = "mappings/metal_mappings.json"

#     metal_data = load_metal_data(json_path)

#     # example user queries
#     queries = ["1.4125", "CCR1150", "AIS440C", "M-17C", "foo-bar"]
#     for q in queries:
#         main_name, doc_path, score = fuzzy_match_metal(q, metal_data)
#         if main_name:
#             print(f"Query='{q}' -> Matched: {main_name} (score={score}), doc_path={doc_path}")
#         else:
#             print(f"Query='{q}' -> No match (score={score})")

# Query='1.4125' -> Matched: CHRONIFER M-17C (score=100.0), doc_path=markdowns/Klein_Metals/CCR-1150.md
# Query='CCR1150' -> Matched: CCR-1150 (score=100.0), doc_path=markdowns/Klein_Metals/1.4125.md
# Query='AIS440C' -> Matched: CHRONIFER M-17C (score=93.33333333333333), doc_path=markdowns/Klein_Metals/CCR-1150.md
# Query='M-17C' -> No match (score=0)
# Query='foo-bar' -> No match (score=0)


