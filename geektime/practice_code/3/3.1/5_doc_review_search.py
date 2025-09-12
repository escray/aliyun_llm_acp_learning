import sys
sys.path.append("..")
from chatbot import rag, llm

# 加载索引并设置query engine
query_engine = rag.create_query_engine(index = rag.load_index())

prompt = '''
【角色背景】
你是一个问题分类路由器，负责判断用户问题的类型，并将其归入下列三类之一：
1. 公司内部文档查询
2. 内容翻译
3. 文档审查

【任务要求】
你的任务是根据用户的输入内容，判断其意图并仅选择一个最贴切的分类。请仅输出分类名称，不需要多余的解释。判断依据如下：

- 如果问题涉及公司政策、流程、内部工具或职位描述与职责等内容，内部职位包括：“技术内容工程师”，选择“公司内部文档查询”。
- 如果问题涉及任意一门非中文的语言，且输入中没有任何出现任何外语或出现“翻译”等字眼，选择“内容翻译”。
- 如果问题涉及检查或总结外部文档或链接内容，选择“文档审查”。
- 用户的前后输入与问题分类并没有任何关系，请单独为每次对话考虑分类类别。

【Few-shot 示例】
示例1：用户输入：“公司内部有哪些常用的项目管理工具？”
分类：公司内部文档查询

示例2：用户输入：“How can we finish the assignment on time?”
分类：内容翻译

示例3：用户输入：“https://help.aliyun.com/zh/model-studio/user-guide/long-context-qwen-long”
分类：文档审查

示例4：用户输入：“技术内容工程师需要设计和开发⾼质量的教育教材和课程吗？”
分类：文档审查

示例5：用户输入：“技术内容工程师核心职责是什么？”
分类：公司内部文档查询

示例6：用户输入：“帮我检查下这段文档：技术内容工程师有需要进行内容优化与更新与跨部门合作吗？“”
分类：公司内部文档查询

【用户输入】
以下是用户的输入，请判断分类：

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

def ask_v4(question):
  question_type = get_question_type(question)
  print(f"question: {question}\ntype: {question_type}")
  if question_type == '文档审查':
    return llm.invoke('你是文档纠错专家，负责找出文档中或网页内容的明显错误，并且言简意赅的回复。如果没有明显问题，请直接回复没有问题\n' + question)
  elif question_type == '公司内部文档查询':
    return rag.ask(question, query_engine = query_engine)
  elif question_type == '内容翻译':
    return llm.invoke(f"你是一名翻译专家，你要识别不同语言的文本，并翻译为中文。\n{question}")
  else:
    return "你是一名翻译专家，你要识别不同语言的文本，并翻译为中文。"

if __name__ == "__main__":
  print(ask_v4('https://www.promptingguide.ai/zh/techniques/fewshot'))
  print('')
  print(ask_v4('请帮我检查下这段文档：技术内容工程师有需要进行内容优化与更新与跨部门合作吗？'))
  print('')
  print(ask_v4('技术内容工程师有需要进行内容优化与更新与跨部门合作吗？'))
  print('')
  print(ask_v4('A true master always carries the heart of a student.'))
