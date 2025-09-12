from llama_index.llms.dashscope import DashScope
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 创建索引
documents = SimpleDirectoryReader("../chatbot/docs").load_data()
index = VectorStoreIndex.from_documents(
  documents,
  embed_model = DashScopeEmbedding(
    model_name = DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2
  )
)

# 创建查询引擎
query_engine = index.as_query_engine(
  streaming = True,
  llm = DashScope(
    model_name = "qwen-plus",
    api_key = os.getenv("DASHSCOPE_API_KEY")
  )
)

# print(query_engine.get_prompts())

# 打印prompt模板
print(f"LlamaIndex default prompt template: \n{query_engine.get_prompts()['response_synthesizer:text_qa_template'].default_template.template}")

# a=100
# b=200
# print(f"a的值是{a}  , b的值是{b}")

# LlamaIndex default prompt template:
# Context information is below.
# ---------------------
# {context_str}
# ---------------------
# Given the context information and not prior knowledge, answer the query.
# Query: {query_str}
# Answer:
