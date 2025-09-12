import os
from llama_index.llms.dashscope import DashScope
from llama_index.core.base.llms.types import MessageRole, ChatMessage

# pip install llama-index-llms-dashscope

# 从环境变量获取 API key
api_key = os.getenv("DASHSCOPE_API_KEY")

# 定义 llm 用于后续和大模型交互
llm = DashScope(
  api_key = api_key,
  model_name = "qwen-plus",
  stream = True,
  enable_search = True
)

# 定义输入到大模型中的messages，你可以在此定义system message与user message。system message用于指定大模型的人设、回复方式等，user message用于接收用户的输入内容。
messages = [
  ChatMessage(role=MessageRole.SYSTEM, content="你是一个新闻助手。你需要为我搜索新闻并提供新闻的来源网址。"),
  ChatMessage(role=MessageRole.USER, content="为我提供，截止到今天的3个阅兵相关新闻")
]

responses = llm.stream_chat(messages)
for response in responses:
  print(response.delta, end="")
