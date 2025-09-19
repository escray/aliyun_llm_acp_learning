from openai import OpenAI
from datetime import datetime
import json
import os

client = OpenAI(
  # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
  api_key=os.getenv("DASHSCOPE_API_KEY"),
  # 填写DashScope SDK的base_url
  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 定义工具列表，模型在选择使用哪个工具时会参考工具的name和description
tools = [
  # 工具1 获取当前时刻的时间
  {
    "type": "function",
    "function": {
      "name": "get_current_time",
      "description": "获取当前的时间，时间格式为 年-月-日 时,",
      # 因为获取当前时间无需输入参数，因此parameters为空字典
      "parameters": {}
    }
  },
  # 工具2 获取指定城市的天气
  {
    "type": "function",
    "function": {
      "name": "get_current_weather",
      "description": "获取指定城市的天气，天气信息包括温度、湿度、风速等。",
      "parameters": {
        "type": "object",
        "properties": {
          # 查询天气时需要提供位置，因此参数设置为location
          "location": {
            "type": "string",
            "description": "要查询的城市名称，例如：北京、上海、广州等"
          }
        },
        "required": ["location"]
      }
    }
  }
]

# 模拟天气查询工具。返回结果示例：“北京今天是雨天。”
def get_current_weather(location):
  # 这里为了演示方便，直接返回一个固定的天气信息
  return f"{location}今天是雨天。"

# 查询当前时间的工具。返回结果示例：“当前时间：2024-04-15 17:15:18。“
def get_current_time():
  # 获取当前日期和时间
  current_time = datetime.now()
  # 格式化当前日期和时间
  formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
  # 返回格式化后的当前时间
  return f"当前时间：{formatted_time}。"

# 封装模型响应函数
def get_response(messages):
  print("-"*60)
  completion = client.chat.completions.create(
    # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen-plus",
    messages=messages,
    tools=tools,
  )
  return completion.model_dump()

def call_with_messages():
  print('\n')
  messages = [
    {
      # 提问示例："现在几点了？" "一个小时后几点" "北京天气如何？"
      "content": input('请输入您的问题：'),
      "role": "user"
    }
  ]
  print("-"*60)
  # 模型的第一轮调用
  i = 1
  first_response = get_response(messages)
  assistant_output = first_response['choices'][0]['message']
  print(f"第{i}轮模型输出：{assistant_output}")
  if assistant_output['content'] is None:
    assistant_output['content'] = ""
  messages.append(assistant_output)
  # 如果不需要调用工具，则直接返回最终答案
  if assistant_output['tool_calls'] == None:
    print(f"无需调用工具，我可以直接回复：{assistant_output['content']}")
    return
  # 如果需要调用工具，则进行模型的多轮调用，直到模型判断无需调用工具
  while assistant_output['tool_calls'] != None:
    tool_call = assistant_output['tool_calls'][0]
    tool_name = tool_call['function']['name']
    tool_args = tool_call['function']['arguments']
    print(f"模型调用了工具：{tool_name}，参数是：{tool_args}")
    # 根据工具名称调用相应的工具函数
    if tool_name == "get_current_time":
      tool_info = {"name": "get_current_time", "role": "tool"}
      tool_info['content'] = get_current_time()
    elif tool_name == "get_current_weather":
      tool_info = {"name": "get_current_weather", "role": "tool"}
      # 提取位置参数信息
      location = json.loads(tool_args)
      tool_info['content'] = get_current_weather(location)
    else:
      tool_info['content'] = f"未知的工具：{tool_name}"
    print(f"工具返回的结果是：{tool_info['content']}")
    # 将工具调用结果添加到消息列表中，继续调用模型
    messages.append(tool_info)

    assistant_output = get_response(messages)['choices'][0]['message']

    if assistant_output['content'] is None:
      assistant_output['content'] = ""
    messages.append(assistant_output)
    i += 1
    print(f"第{i}轮模型输出：{assistant_output}")

  print(f"最终回复是：{assistant_output['content']}")

if __name__ == '__main__':
  call_with_messages()
