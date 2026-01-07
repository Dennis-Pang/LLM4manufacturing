"""Query rewrite prompt"""

QUERY_REWRITE_PROMPT = """
Please rewrite the query to be more specific and clear to improve retrieval effectiveness.
If multiple parameters are requested in the original query, please split them into separate queries.

INSTRUCTIONS:
1. Identify the machining operation, material, tool, and requested parameters
2. Create one specific query for EACH requested parameter
3. Return ONLY a properly formatted Python list of strings in a list

Examples:

Original query: "I wanna turn 1.4125 steel with D10, cutting speed?"
Rewritten: "What's the cutting speed for turning 1.4125 steel with D10 tool?"

Original query: "I wanna turn 1.4125 steel with D10, cutting speed and feed rate?"
Rewritten: "What's the cutting speed for turning 1.4125 steel with D10 tool?",
"What's the feed rate for turning 1.4125 steel with D10 tool?"
"""
