from typing import Optional
from pathlib import Path
from llama_index.llms.dashscope import DashScope
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels
)

import logging
logging.basicConfig(level=logging.ERROR)

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def indexing(
    document_path: str = "./docs",
    persist_path: str = "knowledge_base/test"
) -> None:
    """建立索引并持久化存储

    Args:
        document_path: 文档路径
        persist_path: 索引存储路径

    Raises:
        FileNotFoundError: 当文档路径不存在时
        PermissionError: 当没有写入权限时
    """
    try:
        index = create_index(document_path)
        index.storage_context.persist(persist_path)
    except Exception as e:
        logging.error(f"索引创建失败: {str(e)}")
        raise

def create_index(document_path: str = "./docs") -> VectorStoreIndex:
    """建立文档索引

    Args:
        document_path: 文档路径

    Returns:
        VectorStoreIndex: 创建的索引对象

    Raises:
        FileNotFoundError: 当文档路径不存在时
    """
    if not Path(document_path).exists():
        raise FileNotFoundError(f"文档路径不存在: {document_path}")

    documents = SimpleDirectoryReader(document_path).load_data()
    # 创建一个自定义的节点解析器
    # from llama_index.core import VectorStoreIndex
    # from llama_index.core.node_parser import SentenceSplitter
    # node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)

    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=DashScopeEmbedding(
            model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2
        )
        # ),
        # node_parser=node_parser  # 使用自定义的节点解析器
    )
    return index

def load_index(persist_path="knowledge_base/test"):
    """
    加载索引
    参数
      persist_path(str): 索引文件路径
    返回
      VectorStoreIndex: 索引对象
    """
    storage_context = StorageContext.from_defaults(persist_dir=persist_path)
    return load_index_from_storage(storage_context,embed_model=DashScopeEmbedding(
      model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2
    ))

def create_query_engine(index):
    """
    创建查询引擎
    参数
      index(VectorStoreIndex): 索引对象
    返回
      QueryEngine: 查询引擎对象
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

    query_engine = index.as_query_engine(
        streaming=True,
        llm=DashScope(
            model_name="qwen-plus",
            api_key=api_key
        ),
        # 设置检索召回的文档切片数量，默认是2
        similarity_top_k=5,
    #   node_postprocessors=[
    #         # 根据相似度分数过滤 score
    #         SimilarityPostprocessor(similarity_cutoff=0.7),
    #         # 使用关键词重排序
    #         KeywordNodePostprocessor(
    #             required_keywords=["RAG", "检索"], # 必须包含的关键词列表
    #             exclude_keywords=["测试", "样例"] # 必须排除的关键词列表
    #         )
    #     ]
    )
    return query_engine

# 使用自定义排序函数
# from llama_index.core.postprocessor import CustomPostprocessor

# def custom_rank(nodes):
#     # 自定义排序逻辑
#     return sorted(nodes, key=lambda x: len(x.text))

# query_engine = index.as_query_engine(
#     node_postprocessors=[
#         CustomPostprocessor(custom_rank)
#     ]
# )

def debug_output(func):
    """调试装饰器，用于打印检索召回的文档切片"""
    def wrapper(*args, **kwargs):
        response = func(*args, **kwargs)

        # 打印检索召回的文档切片
        print('\n\n检索召回的文档切片如下：')
        for i, source_node in enumerate(response.source_nodes, 1):
            print(f"\n[{i}] {source_node}")

        return response
    return wrapper

def ask(
    question: str,
    query_engine,
    stream: bool = True,
    debug: bool = False
) -> Optional[str]:
    """向答疑机器人提问

    Args:
        question: 问题内容
        query_engine: 查询引擎对象
        stream: 是否使用流式输出
        debug: 是否打印检索召回的文档切片

    Returns:
        Optional[str]: 如果stream=False，返回完整回答；否则返回None

    Raises:
        ValueError: 当问题为空时
    """
    if not question.strip():
        raise ValueError("问题不能为空")

    try:
        # 根据debug参数决定是否使用装饰器
        query_func = debug_output(query_engine.query) if debug else query_engine.query
        response = query_func(question)

        if stream:
            response.print_response_stream()
            return None
        return str(response)
    except Exception as e:
        logging.error(f"查询失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 建立索引并持久化存储
    indexing()
    # 加载索引
    index = load_index()
    # 创建查询引擎
    query_engine = create_query_engine(index)
    # 提问
    ask("100字以内，简要回答RAG的4个典型应用场景", query_engine, stream=True, debug=False)

