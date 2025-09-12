import os
from dashscope import Generation

def invoke(user_message, stream=False):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    
    messages = [
        {"role": "system", "content": "你是智能助理贾维斯，用中文回答我的全部问题。"},
        {"role": "user", "content": user_message}
    ]
    
    if stream:
        responses = Generation.call(
            model='qwen-plus',
            messages=messages,
            api_key=api_key,
            stream=True
        )
        stream_result = ""
        for response in responses:
            if response.status_code == 200:
                content = response.output.text
                print(content, end="")
                stream_result += content
        return stream_result
    else:
        response = Generation.call(
            model='qwen-plus',
            messages=messages,
            api_key=api_key
        )
        return response.output.text

if __name__ == "__main__":
    print(invoke("写个100字的故事", stream=True))