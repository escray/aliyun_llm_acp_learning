import os
from openai import OpenAI

client = OpenAI(
  api_key=os.getenv("DASHSCOPE_API_KEY"),
  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

system_message = '''你是一个标签提取专家，请按照下面的要求，从用户问题中提取标签信息。
---
【目前支持的标签】
- 岗位名称
- 部门名称
---
【输出要求】
1. 请用 json 输出，如：[{"key": "部门名称", "value": "教研部"}]、[]
2. 识别不出来时，value 可以为 null
---
用户的问题如下：
'''

def extract_tags(question):
  completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen-plus",
    messages=[
      {'role': 'system', 'content': system_message},
      {'role': 'user', 'content': question}
    ],
    # 确保生成结果是 JSON 格式，以便后续流程解析使用
    response_format={"type": "json_object"}
  )
  return completion.choices[0].message.content

print(extract_tags('内容工程师的工作职责是什么'))
print(extract_tags('课程开发部的内容工程师的工作职责是什么'))
print(extract_tags('张伟是哪个部门的'))
print(extract_tags('张伟是谁'))
