"""Metal fuzzy matching"""
import json
from rapidfuzz import fuzz
from langgraph.func import task


@task
def fuzzy_match_metal(
    query: str,
    metal_mapping_path: str = "mappings/metal_mappings.json",
    threshold: int = 80
) -> tuple[str, str, float]:
    """
    Fuzzy match metal name and aliases.

    Args:
        query: Metal string input (e.g., "1.4125", "CCR1150")
        metal_mapping_path: Path to metal mappings JSON
        threshold: Fuzzy matching score threshold (default: 80)

    Returns:
        Tuple of (main_name, doc_path, score)
        Returns (None, None, 0) if no match found

    Metal mapping format:
        {
          "CHRONIFER M-17C": {
            "aliases": ["1.4125", "AISI 440C", "X105CrMo17", "SUS440C"],
            "doc_path": "markdowns/Klein_Metals/CCR-1150.md"
          }
        }
    """
    best_score = 0
    best_main_name = None
    best_doc_path = None
    matched_key = None

    with open(metal_mapping_path, "r", encoding="utf-8") as f:
        metal_data = json.load(f)

    query_norm = query.strip().lower()

    for main_name, info in metal_data.items():
        # Match main name
        main_name_norm = main_name.strip().lower()
        score = fuzz.ratio(query_norm, main_name_norm)
        if score > best_score:
            best_score = score
            best_main_name = main_name
            matched_key = main_name

        # Match aliases
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
