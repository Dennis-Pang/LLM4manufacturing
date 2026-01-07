"""Result logging utilities"""
import os
import json
from datetime import datetime


class ResultLogger:
    """Logger for RAG query results"""

    def __init__(self, results_dir: str, model):
        """
        Initialize result logger.

        Args:
            results_dir: Directory to save results
            model: LLM model instance
        """
        self.results = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_info": {
                "name": model.__class__.__name__,
                "model_id": getattr(model, 'model', "unknown"),
                "temperature": getattr(model, 'temperature', "unknown"),
            }
        }
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def add_result(self, step_name: str, result):
        """Add result to storage"""
        if hasattr(result, "model_dump"):
            self.results[step_name] = result.model_dump()
        elif hasattr(result, "content"):
            self.results[step_name] = result.content
        else:
            self.results[step_name] = str(result)

    def save_results(self):
        """Save all results to file"""
        filename = f"{self.results_dir}/{self.results['timestamp']}.json"
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(self.results, file, ensure_ascii=False, indent=4)
        print(f"\n💾 All results saved to: {filename}")
