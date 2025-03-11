from tool_extrator import tool_search
from metal_extractor import fuzzy_match_metal
from langgraph.func import task
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import Literal
from pydantic import BaseModel, Field
from langgraph.types import interrupt
from online_search import online_search

class Check(BaseModel):
    judge: Literal["yes","no"] = Field(
        description="Decide if all necessary factors are fully included in the query.",
    )
    tool: str = Field(
        description="The tool name if it exists in the query, None if not.",
    )
    metal: str = Field(
        description="The metal name if it exists in the query, None if not.",
    )
    operation: str = Field(
        description="The operation name if it exists in the query, None if not.",
    )
    questioned_parameters: str = Field(
        description="The questioned parameters if it exists in the query, None if not.",
    )

@task
def factors_check(llm, query):
    """check if the query contains all necessary factors, and extract the metal name"""
    result = llm.with_structured_output(Check).invoke(
        [
            SystemMessage(content="""
            You're an expert in manufacturing and metallurgy. Your task is to check if the query contains all necessary elements and accurately extract the metal information.

            Required elements to check:
            1. Operation (e.g., milling, turning, drilling, machining)
            2. Metal/Material (e.g., 1.4125 steel, 1.4425, CCR-1150, TI-64, BT-30)
            3. Tool (e.g., HM Carbide, D10, D20, D60, Cermet, PKD/PCD)
            4. Questioned parameters (e.g., cutting speed, feed rate)

            Metal extraction rules:
            1. For specific metal codes (e.g., CCR-1150, TI-64), extract them exactly
            2. For generic metals (e.g., titanium alloy, stainless steel), use domain knowledge
            3. If both specific code and generic metal present, prioritize the specific code
            4. Ignore tool names like HM Carbide, D10, D20, D60, Cermet, PKD/PCD
            5. If no metal found, return "UNKNOWN" as metal name

            Analysis steps:
            1. Identify operation mentioned in query
            2. Extract precise metal/material name following metal rules
            3. Verify tool specification
            4. Confirm presence of questioned parameters

            For judge, return "yes" only if ALL elements above are present in the query, "no" if any element is missing.
            
            Example Query: "What cutting speed should I use for milling 1.4125 steel with a D10 carbide tool?"
            Example Output:
            {
                "judge": "yes",
                "tool": "D10",
                "metal": "1.4125",
                "operation": "milling",
                "questioned_parameters": "cutting speed"
            }
            """),
            HumanMessage(content=f"Query: {query}")
        ]
    )
    return result

class Answer(BaseModel):
    questioned_parameter: str = Field(
        description="The questioned parameter",
    )
    tool_range: str = Field(
        description="The range of tool's parameter recommendation, None if not.",
    )
    metal_range: str = Field(
        description="The range of metal's parameter recommendation, None if not.",
    )
    combined_range: str = Field(
        description="The combined source of tool and metal's parameter recommendation, 'conflicted' if conflicted parameters between tool and metal.",
    )
    thoughts: str = Field(
        description="""
        Some brief internal thought process of parameter recommendation in 2-3 sentences, 
        if there is no conflicted parameters between tool and metal;
        If there is conflicted parameters between tool and metal, give a brief explanation and make a suggestion based both sources and your own knowledge.
        """,
    )
    
@task
def parameter_recommendation(llm, query: str):
    """main function of parameter recommendation"""
    llm = llm.bind_tools([online_search])
    check = factors_check(llm, query).result()
    if check.judge == "no":
        print("Please provide a complete query with operation, metal and tool information.")
        return 
    
    print("-- Start to generate parameter recommendation:\n")
    metal_name = check.metal

    _, doc_path, _ = fuzzy_match_metal(metal_name).result()
    
    # read the metal document
    metal_doc = None
    if doc_path:
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                metal_doc = f.read()
        except FileNotFoundError:
            return f"No metal references found for {metal_name}", False

    # search the tool references
    tool_refs = tool_search(llm, query)
    if tool_refs is None:
        print("No valid tool references found.")

    # merge the reference information
    references = [metal_doc] if metal_doc else []
    references.extend(tool_refs or [])

    llm = llm.bind_tools([online_search])
    
    # create the conversation history list
    messages = [
        SystemMessage(content="""
        You are a manufacturing expert. Your task is to recommend cutting parameters based on metal and tool references.

        Analysis Process (internal thought process, not for final answer):

        1. Metal Analysis:
           - Extract Tensile Strength (Rm/UTS)
           - Identify Metal Category (e.g., steel, aluminum alloy)
           - Identify Specific Composition (e.g., Si content in aluminum)
           - Note Additional Properties (hardness, cutting speed limits)
           - Record Recommended Parameters from Metal Documentation
           - If composition details are unclear, list parameters for ALL possible variations

        2. Tool Requirements Analysis:
           - Material Strength Limits
           - Material Composition Specifications
           - Special Conditions or Restrictions
           - Record Recommended Parameters from Tool Documentation

        3. Parameter Integration:
           - Compare parameters from both metal and tool sources
           - For materials with multiple possible compositions:
             * List ALL applicable parameter ranges
             * Clearly state conditions for each range
           - If ranges conflict, use the more conservative values
           - Note any special considerations from either source

        4. Unit Standardization:
           - Convert all strength values to MPa (1 MPa = 1 N/mm²)
           - Ensure consistent units for all parameters (e.g., speeds in m/min, feeds in mm/rev)

        5. Output Format:
           Recommendations:
           For questioned parameters mentioned in the query:
           - If ranges from metal and tool sources align: Present as single merged range
           - If ranges conflict: List both separately as
             * Metal Source: [range] [unit]
             * Tool Source: [range] [unit]
            ** DO NOT OMIT ANY PARAMETER from any source in this case!**
           
           For materials with known composition:
           - Parameter Name: [range] [unit] (considering both metal and tool limits)
           
           For materials with uncertain composition:
           - If [condition A]: Parameter Name: [range A] [unit]
           - If [condition B]: Parameter Name: [range B] [unit]
           
           Reasoning: Explain how recommendations were derived, noting any assumptions or conditions. In total, provide 2-3 sentences. 
           ** DO NOT PRESENT ALL YOUR CHAIN OF THOUGHTS IN THE REASONING SECTION!**

        Important:
        - Do NOT make assumptions about material composition unless explicitly stated
        - List ALL applicable parameter ranges when composition is uncertain
        - Consider and combine recommendations from BOTH metal and tool sources
        - Use the more conservative values when recommendations differ
        - Maintain numerical accuracy - no rounding or approximating
        - Explicitly state if any parameters conflict between sources
        - Keep final response concise and focused on parameters
        """),
    ]
    
    # add the initial query
    messages.append(HumanMessage(content=f"Query: {query}\nReferences: {references}"))
    
    # get the initial answer
    response = llm.with_structured_output(Answer).invoke(messages)

    llm_response =   f"""
    🔍 Questioned parameter: {response.questioned_parameter}
    🔧 Metal's source: {response.metal_range}
    🛠️  Tool's source: {response.tool_range}
    🎯 Combined range: {response.combined_range}
    💭 RagBot's thoughts: {response.thoughts}
    """ 
    print("\n🤖 RagBot's Answer:\n\n", llm_response)
    return response
    
    # while True:
    #     is_approved = input("\nPlease say 'yes' if the answer is satisfactory, otherwise just provide additional information:").lower().strip()
    #     if is_approved == "yes":
    #         return llm_response, True
        
    #     # 将用户反馈添加到对话历史
    #     messages.append(HumanMessage(content=is_approved))
    #     # 添加助手回答到对话历史
    #     messages.append(SystemMessage(content="""
    #     Based on your feedback, I will:
    #     1. If you provided additional specifications: Narrow down the parameter ranges accordingly
    #     2. If you asked about other aspects: Provide additional insights based on manufacturing expertise
    #     3. If you pointed out issues: Correct and clarify the previous recommendations
    #     No need to provide the previous answer.
    #     No format for the answer, just provide the answer in a concise and clear manner.
    #     """))
        
    #     # 使用完整对话历史获取新回答
    #     llm_response = llm.invoke(messages).content
    #     print("\n🤖 Updated Answer:\n", llm_response)
    

