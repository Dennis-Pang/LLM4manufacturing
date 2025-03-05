from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

checkpointer = InMemorySaver()
store = InMemoryStore()

class AgentOutput(BaseModel):
    """单次agent输出的数据结构"""
    v_c: float = Field(description="Cutting speed (m/min)")
    f_c: float = Field(description="Feed rate (mm/rev)")
    a: float = Field(description="Cutting depth (mm)")
    cost: float = Field(description="Calculated cost")
    thoughts: str = Field(description="Agent's reasoning")
    round: int = Field(description="Optimization round number")

load_dotenv()
model = ChatOpenAI(model="gpt-4o").with_structured_output(AgentOutput)


    
def calculate_production_time_cost(v_c: float, f_c: float) -> float:
    """
    计算生产时间成本
    假设生产时间与切削速度和进给率成反比
    """
    return 1/(v_c*f_c)

def calculate_tool_life(v_c: float, C_T: float, k_T: float) -> float:
    """
    计算刀具寿命 (T_c) 使用 Taylor 公式：
      T_c = C_T * v_c^(k_T)
    其中:
      C_T : 在 v_c = 1 m/min 时的刀具寿命（单位与 T_c 相同）
      k_T : 经验指数（通常为负，表示速度升高时寿命降低）
    """
    T_c = C_T * (v_c ** k_T)
    return T_c

def calculate_tool_cost(v_c: float, C_T: float, k_T: float, C_I: float) -> float:
    """
    计算刀具成本贡献
    假设每个零件的刀具成本 = C_I / T_c, 其中 T_c 为刀具寿命
    """
    T_c = calculate_tool_life(v_c, C_T, k_T)
    tool_cost = C_I / T_c
    return tool_cost

def surface_roughness_cost(v_c: float, f_c: float, a: float) -> float:
    """
    表面粗糙度成本（示例函数）：
    假设粗糙度与进给率正相关，与切削速度负相关，同时切削深度也有一定影响。
    这里采用一个简单模型：
         cost = k * (f_c / v_c) + a
    k 为常数（示例中取10）
    """
    k = 10.0
    return k * (f_c / v_c) + a

wear_agent = create_react_agent(
    model=model,
    tools=[calculate_tool_cost, calculate_tool_life],
    name="wear_expert",
    prompt="""
    You are an expert in machining tool wear analysis and optimization. 
    Your task is to evaluate and recommend optimal cutting parameters based on tool wear considerations.
    A input range is provided, you need to evaluate the tool wear cost based on the input range then give a recommendation.

    ANALYSIS PROCESS:
    1. Calculate tool life using your tool
    2. Evaluate tool cost considering:
       - Tool life expectancy
       - Tool replacement cost
       - Production volume impact
    3. Consider the trade-off between:
       - Higher cutting speeds (increased productivity)
       - Lower tool life (increased tool cost)

    OUTPUT REQUIREMENTS:
    1. Tool life analysis for the given speed range
    2. Cost impact assessment
    3. Specific recommendation for optimal cutting speed
    4. Brief explanation of the recommendation

    IMPORTANT:
    - Use the provided tools to perform calculations
    - Consider both technical and economic factors
    - Provide specific numerical recommendations
    - Explain the reasoning behind your recommendation
    """
)

time_agent = create_react_agent(
    model=model,
    tools=[calculate_production_time_cost],
    name="time_expert",
    prompt="""
    You are an expert in machining process optimization and production time analysis. 
    Your task is to evaluate and optimize production time based on cutting parameters.

    ANALYSIS PROCESS:
    1. Calculate production time cost for the given parameters
    2. Consider the relationship between:
       - Cutting speed and production time
       - Feed rate and production time
    3. Evaluate the impact on overall productivity

    OUTPUT REQUIREMENTS:
    1. Production time analysis
    2. Cost impact assessment
    3. Specific recommendation for optimal parameters
    4. Brief explanation of the recommendation

    IMPORTANT:
    - Use the provided tools for calculations
    - Consider both speed and feed rate impacts
    - Provide specific numerical recommendations
    - Explain the reasoning behind your recommendation
    """
)

roughness_agent = create_react_agent(
    model=model,
    tools=[surface_roughness_cost],
    name="roughness_expert",
    prompt="""
    You are an expert in surface quality and machining finish analysis. 
    Your task is to evaluate and optimize surface roughness based on cutting parameters.

    ANALYSIS PROCESS:
    1. Calculate surface roughness cost for the given parameters
    2. Consider the relationship between:
       - Cutting speed and surface finish
       - Feed rate and surface finish
       - Cutting depth and surface finish
    3. Evaluate the impact on part quality

    OUTPUT REQUIREMENTS:
    1. Surface roughness analysis
    2. Quality impact assessment
    3. Specific recommendation for optimal parameters
    4. Brief explanation of the recommendation

    IMPORTANT:
    - Use the provided tools for calculations
    - Consider all parameter impacts on surface finish
    - Provide specific numerical recommendations
    - Explain the reasoning behind your recommendation
    """
)
# Create supervisor workflow
workflow = create_supervisor(
    [wear_agent, time_agent, roughness_agent],
    model=model, 
    output_mode="last_message",
    prompt="""
    You are the supervisor managing three machining experts:
    1. wear_expert: Tool wear analysis
    2. time_expert: Production time optimization
    3. roughness_expert: Surface quality analysis

    TASK:
    Optimize machining parameters (v_c, f_c, a) to minimize total cost.

    PROCESS:
    1. Parse user input ranges and requirements
    2. Get expert evaluations
    3. Propose and evaluate parameters
    4. Check convergence (stop if cost change ≤ 10%)

    OUTPUT:
    1. Recommended parameters (v_c, f_c, a)
    2. Cost breakdown (wear, time, roughness)
    3. Brief justification

    IMPORTANT:
    - Prioritize user requirements
    - Stay within safe machining limits
    - Consider practical constraints
    """
)

# Compile and run
app = workflow.compile(
    checkpointer=checkpointer,
    store=store
)

# config = {"configurable": {"thread_id": "1"}}

# message = "The cut speed is 60-120 from tool and 20-30 from metal, cut depth is 0.5-2.0 mm, feed rate is 0.01-0.12 mm/U, make sure the tool life preserved."

# result = app.invoke({
#     "messages": [{"role": "user","content": message}]},
#     config=config)

# print(result)

