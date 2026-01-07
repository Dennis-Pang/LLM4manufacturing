"""Relevance rating prompt"""

RELEVANCE_RATING_PROMPT = """
As a manufacturing expert, your task is to evaluate if a reference matches a user's query about machining tools.

Evaluation Rules:
1. Focus ONLY on three elements in the query:
   - Tool name (from: HM Carbide, D10, D20, D60, Cermet, PKD/PCD)
   - Machining operation (e.g., turning, milling, drilling, etc.)
   - Questioned parameters (e.g., cutting speed, feed rate)

2. IGNORE all material/metal specifications in the query

Example:
Query: "What's the cutting speed for turning ABC-1234 steel with D10?"
Analysis:
- Tool: D10 ✓
- Operation: turning ✓
- Questioned parameters: cutting speed ✓
- Material (ABC-1234): ignore this

Your task: Determine if the provided reference contains relevant information to answer the query, focusing only on the tool and operation.

Please provide a very brief explanation for your decision.
"""
