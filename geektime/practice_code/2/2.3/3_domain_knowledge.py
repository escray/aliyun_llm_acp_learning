import os
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai_like import OpenAILike

# 你需要安装：llama-index 包
# 否则会报错：ModuleNotFoundError: No module named 'llama_index'
# 安装方法：
# pip install llama-index
# pip install llama-index-llms-openai-like

# 所调用的模型
llm = OpenAILike(
  model="qwen-plus",
  api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
  api_key=os.getenv("DASHSCOPE_API_KEY"),
  is_chat_model=True
)

# 用户问题
user_question = "我是软件一组的，请问项目管理应该用什么工具"

# 公司项目管理工具相关的知识
knowledge = '''公司项目管理工具有两种选择:
1. **Jira**: 对于软件开发团队来说,Jira是一个非常强大的工具,
   支持敏捷开发方法,如Scrum和Kanban。它提供了丰富的功能,
   包括问题跟踪、时间跟踪等。

2. **Microsoft Project**: 对于大型企业或复杂项目,
   Microsoft Project提供了详细的计划制定、资源分配和成本控制等功能。
   它更适合那些需要严格控制项目时间和成本的场景。

在一般情况下请使用Microsoft Project,公司购买了完整的许可证。
软件研发一组、三组和四组正在使用Jira,
计划于2026年之前逐步切换至Microsoft Project。'''

# 构建消息内容
messages = [
  ChatMessage(
    role="system",
    content="""你负责教育内容开发公司的答疑,
        你的名字叫贾维斯,你要回答学员的问题。"""
        + knowledge
  ),
  ChatMessage(role="user", content=user_question)
]

# 使用 OpenAILike 调用大模型，并流式返回结果
response = llm.stream_chat(messages)
for r in response:
  print(r.delta, end="")
