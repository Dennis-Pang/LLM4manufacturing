# 重构指南

## 新架构概览

```
backend/
├── config/              # 配置管理
│   ├── llm_config.py   # LLM配置
│   └── prompts/        # 系统提示词库
│
├── data_processing/     # 数据处理
│   ├── preprocessing.py    # 文档预处理
│   └── vectorization.py    # 向量化
│
├── retrieval/          # 检索模块
│   ├── vector_store.py     # 向量存储
│   ├── metal_matcher.py    # 金属匹配
│   ├── tool_retriever.py   # 刀具检索
│   └── relevance_rater.py  # 相关性评估
│
├── agents/             # Agent核心
│   ├── query_agent.py          # 查询处理
│   └── recommendation_agent.py # 参数推荐
│
├── workflows/          # 工作流
│   └── rag_workflow.py # 主RAG工作流
│
├── models/             # Pydantic模型
│   ├── query_models.py         # 查询模型
│   └── recommendation_models.py # 推荐模型
│
├── utils/              # 工具函数
│   └── logger.py       # 日志记录
│
├── legacy/             # 废弃代码
│   ├── online_search.py
│   ├── experiment.py
│   └── battlefield.py
│
└── main.py             # 主入口
```

## 使用方法

### 运行RAG系统

```bash
python main.py
```

### 数据预处理

```python
from data_processing import preprocess_document, create_vector_db

# 预处理文档
preprocess_document(
    input_path="pdfs/document.md",
    output_path="washed_documents/document.md"
)

# 创建向量数据库
create_vector_db("washed_documents/document.md")
```

### 使用不同的LLM

```python
from config import llm_openai, llm_anthropic, llm_deepseek
from workflows import rag_pipeline

# 使用Claude
rag_pipeline.invoke((llm_anthropic, query), config=config)
```

## 核心改进

### 1. 模块化设计
- **配置分离**: 提示词、LLM配置独立管理
- **数据处理独立**: 预处理和向量化分离
- **检索逻辑清晰**: 向量搜索、金属匹配、刀具检索独立
- **Agent简洁**: 查询处理和参数推荐分离

### 2. 提示词管理
所有系统提示词集中在 `config/prompts/`，便于：
- 版本管理
- A/B测试
- 多语言支持

### 3. 结构化输出
所有Pydantic模型集中在 `models/`，类型安全且易于维护

### 4. 简洁的Agent设计
避免过度封装，每个agent函数都是独立的task，直接调用即可

## 主要变化

### 旧架构 vs 新架构

| 文件 | 新位置 | 说明 |
|------|--------|------|
| RAG.py | workflows/rag_workflow.py + main.py | 拆分为工作流和入口 |
| parameter_recommendator.py | agents/recommendation_agent.py | 重构为agent |
| retriever.py | retrieval/vector_store.py | 移至检索模块 |
| metal_extractor.py | retrieval/metal_matcher.py | 移至检索模块 |
| tool_extrator.py | retrieval/tool_retriever.py | 移至检索模块 |
| rater.py | retrieval/relevance_rater.py | 移至检索模块 |
| preprocessing.py | data_processing/preprocessing.py | 移至数据处理 |
| markdown2embedding.py | data_processing/vectorization.py | 移至数据处理 |
| result_logger.py | utils/logger.py | 移至工具模块 |
| online_search.py | legacy/online_search.py | 移至废弃代码 |
| experiment.py | legacy/experiment.py | 移至废弃代码 |
| battlefield.py | legacy/battlefield.py | 移至废弃代码 |

## 迁移完成后的清理

重构完成并测试通过后，可以删除以下旧文件：
- RAG.py
- parameter_recommendator.py
- retriever.py
- metal_extractor.py
- tool_extrator.py
- rater.py
- preprocessing.py
- markdown2embedding.py
- result_logger.py
