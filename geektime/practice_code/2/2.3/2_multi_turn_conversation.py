import os
from dashscope import Generation

def get_response(messages):
  """使用DashScope API获取响应"""
  try:
    response = Generation.call(
      api_key = os.getenv("DASHSCOPE_API_KEY"),
      model = "qwen-plus",
      messages = messages,
      result_format = "message"
    )
    return response

  except Exception as e:
    print(f"Call LLM API Error: {str(e)}")
    return None

messages = [
  {
    'role': 'system',
    'content': '''你是一名百炼手机商店的店员，你负责给用户推荐手机。
        手机有两个参数：屏幕尺寸（包括6.1英寸、6.5英寸、6.7英寸）、分辨率（包括2K、4K）。
        你一次只能向用户提问一个参数。如果用户提供的信息不全，你需要反问他，让他提供没有提供的参数。
        如果参数收集完成，你要说：我已了解您的购买意向，请稍等。''',
  }
]

# 初始化tokens计数
total_tokens = 0

assistant_output = "欢迎光临百炼手机商店，您需要购买什么尺寸的手机呢？"
print(f"LLM OUTPUT: {assistant_output}\n")

while "我已了解您的购买意向" not in assistant_output:
  user_input = input("Please enter: ")
  messages.append({'role': 'user', 'content': user_input})

  response = get_response(messages)
  assistant_output = response.output.choices[0]["message"]["content"]
  # 获取本轮tokens
  input_tokens = response.usage["input_tokens"]
  output_tokens = response.usage["output_tokens"]
  round_tokens = input_tokens + output_tokens
  # 累计总tokens
  total_tokens += round_tokens

  messages.append({'role': 'assistant', 'content': assistant_output})
  print(f"LLM output: {assistant_output}")
  print(f"this round expense: ")
  print(f" Input tokens: {input_tokens}")
  print(f" Output tokens: {output_tokens}")
  print(f" Round tokens: {round_tokens}")
  print(f"Total tokens: {total_tokens}")
  print("\n")
