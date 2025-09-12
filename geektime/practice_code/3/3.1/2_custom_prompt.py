import sys
sys.path.append('..')
from chatbot import rag

# 加载索引
index = rag.load_index()
# 设置query engine
query_engine = rag.create_query_engine(index=index)
# 自定义prompt模板
from llama_index.core import PromptTemplate

prompt_template_string = (
  "你是贾维斯，你回答问题时，需要在回答前加上贾维斯说:"
  "注意事项：\n"
  "1. 根据上下文信息而非先验知识来回答问题。\n"
  "2. 如果是工具咨询类问题，请务必给出下载地址链接。\n"
  "以下是参考信息。"
  "---------------------\n"
  "{context_str}\n"
  "---------------------\n"
  "问题：{query_str}\n。"
  "回答：贾维斯说:<answer>"
)

prompt_template = PromptTemplate(prompt_template_string)
query_engine.update_prompts({"response_synthesizer:text_qa_template": prompt_template})

def qa_v2(question, query_engine):
  """执行问答"""
  streaming_response = query_engine.query(question)
  streaming_response.print_response_stream()



if __name__ == "__main__":
  qa_v2('我们公司项目管理应该用什么工具', query_engine = query_engine)
