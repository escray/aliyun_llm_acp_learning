import os
from typing import List
from dashscope import Generation

def questions(user_input: str) -> str:
  """使用智能客服模型对用户输入进行问题分类"""
  messages = [
    {'role': 'system', 'content': '''
      你是智能客服，你需要根据用户的问题给出问题的分类。
      分类只能是"价格过高","售后服务不好","产品体验不佳"三种，
      不能输出其他内容。
      输出的格式必须是：
      \n用户问题: <这里是用户的问题> -- 问题分类: <这里是问题分类> \n'''
    },
    {'role': 'user', 'content': user_input}
  ]

  try:
    response = Generation.call(
      api_key = os.getenv("DASHSCOPE_API_KEY"),
      model = "qwen-plus",
      messages = messages,
      result_format = "message",
      stream = True,
      # 增量式流式输出
      incremental_output = True
    )

    full_response = ""
    for chunk in response:
      if chunk.output and chunk.output.choices:
        full_response += chunk.output.choices[0].message.content

    # 返回最后一个有效的分类结果
    for line in reversed(full_response.split('\n')):
      if ' -- 问题分类: ' in line:
        return line.strip()

    return full_response.strip()
  except Exception as e:
    return f"Exception: {str(e)}"

def process_feedback_file(filename: str):
  """处理用户反馈文件,对每行内容进行分类"""
  try:
    with open(filename, "r", encoding="utf-8") as file:
      for line in file:
        if line.strip():
          print(questions(line.strip()))
  except Exception as e:
    print(f"File Process Error: {str(e)}")


if __name__ == "__main__":
  os.chdir(os.path.dirname(os.path.abspath(__file__)))
  process_feedback_file("用户反馈.txt")
