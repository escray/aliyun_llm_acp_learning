import sys
sys.path.append('..')
from chatbot import llm

prompt = '''
【角色背景】
你是一个问题分类路由器，需要识别问题的类型。
---
【任务要求】
问题的类型目前有：公司内部文档查询、内容翻译、文档审查
你只能选择一类意图作为你的判断，并不要说任何无关内容。
---
【用户输入】
以下是用户输入，请判断：
'''

def get_question_type(question):
  """
    获取问题的类型

    参数:
    - question (str): 需要判断的问题文本

    返回值:
    - str: 问题类型，可能是 "文档审查" 、 "内容翻译"或“公司内部文档查询”
  """
  return llm.invoke(prompt + question)

print(get_question_type('https://www.promptingguide.ai/zh/techniques/fewshot'))
print('\n')
print(get_question_type('That is a big one I dont know why'))
print('\n')
print(get_question_type('作为技术内容工程师有什么需要注意的吗?'))
