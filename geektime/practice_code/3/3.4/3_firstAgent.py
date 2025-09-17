# 1. 环境配置
import os
import sys
sys.path.append('..')
# 获取dashscope的api key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# 设置代理
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
# 禁用nltk下载
os.chdir(os.path.dirname(__file__))
os.environ['NLTK_DATA'] = './nltk_cache'

# 2. 导入依赖
from llama_index.llms.dashscope import DashScope
from llama_index.core.base.llms.types import MessageRole, ChatMessage

from chatbot.rag import load_index
from dashscope import Assistants, Messages, Runs, Threads
import json

# 3. 核心工具函数
def query_employee_info(query):
  '''
  输入用户提问，输出员工信息查询结果
  '''
  # 1. 首先根据用户提问，使用NL2SQL生成SQL语句
  llm = DashScope(model_name="qwen_plus")
  messages = [
    ChatMessage(role=MessageRole.SYSTEM, content='''你有一个表叫employees，记录公司的员工信息，这个表有department（部门）、name（姓名）、HR三个字段。
    你需要根据用户输入生成sql语句进行查询,你一定不能生成sql语句之外的内容，也不要把```sql```这个信息加上。'''),
    ChatMessage(role=MessageRole.USER, content=query)
  ]
  SQL_output = llm.chat(messages).message.content
  # 打印出SQL语句
  print(f'SQL statement is: {SQL_output}')
  # 2. 根据SQL语句去查询数据库（此处为模拟查询），并返回结果
  if SQL_output == "SELECT COUNT(*) FROM employees WHERE department = '教育部门'":
    return "教育部门共有66名员工"
  if SQL_output == "SELECT HR FROM employees WHERE name = '张三'":
    return "张三的HR是李四"
  if SQL_output == "SELECT department FROM employees WHERE name = '王五'":
    return "王五的部门是后勤部"
  else:
    return "抱歉，我暂时无法回答您的问题"

def send_leave_application(date):
  '''
  输入请假时间，输出请假申请发送结果
  '''
  return f'已为你发送请假申请，请假日期是{date}'

def query_company_info(query):
  '''
  输入用户提问，输出公司信息查询结果
  '''
  # 使用封装好的函数加载索引
  index = load_index()
  query_engine = index.as_query_engine(
    llm=DashScope(model_name="qwen-plus")
  )
  return query_engine.query(query).response

# 4. Agent定义和配置
ChatAssistant = Assistants.create(
  # 在此指定模型名称
  model = "qwen-plus",
  # 在此指定Agent名称
  name = "贾维斯",
  # 在此指定Agent的描述信息
  description="一个智能助手，能够查询员工信息，帮助员工发送请假申请，或者查询公司规章制度。",
  # 用于提示大模型所具有的工具函数能力，也可以规范输出格式
  instructions='''你是贾维斯，你的功能有以下三个：
    1. 查询员工信息。例如：查询员工张三的HR是谁；
    2. 发送请假申请。例如：当员工提出要请假时，你可以在系统里帮他完成请假申请的发送；
    3. 查询公司规章制度。例如：我们公司项目管理的工具是什么？
    请准确判断需要调用哪个工具，并礼貌回答用户的提问。
  ''',
  # 将工具函数传入
  tools = [
    {
      # 定义工具函数类型，一般设置为function即可
      'type': 'function',
      'function': {
        # 定义工具函数名称，通过map方法映射到query_employee_info函数
        'name': '查询员工信息',
        # 定义工具函数的描述信息，Agent主要根据description来判断是否需要调用该工具函数
        'description': '当需要查询员工信息时非常有用，比如查询员工张三的HR是谁，查询教育部门总人数等。',
        # 定义工具函数的参数
        'parameters': {
          'type': 'object',
          'properties': {
            # 将用户的提问作为输入参数
            'query': {
              'type': 'str',
              # 对输入参数的描述
              'description': '用户的提问'
            },
          },
          # 在此声明该工具函数需要哪些函数
          'required': ['query']
        },
      },
    },
    {
      'type': 'function',
      'function': {
        'name': '发送请假申请',
        'description': '当需要帮助员工发送请假申请时非常有用',
        'parameters': {
          'type': 'object',
          'properties': {
            # 需要请假的时间
            'date': {
              'type': 'str',
              'description': '员工想要请假的时间'
            },
          },
          'required': ['date']
        },
      }
    },
    {
      'type': 'function',
      'function': {
        'name': '查询公司规章制度',
        'description': '当需要查询公司规章制度时非常有用，比如查询公司项目管理的工具是什么，查询公司都有哪些部门等',
        'parameters': {
          'type': 'object',
          'properties': {
            'query': {
              'type': 'str',
              'description': '用户的提问'
            }
          }
        }
      }
    }
  ]
)
print(f'{ChatAssistant.name}创建完成')

# 建立Agent Function name与工具函数的映射关系
function_mapper = {
  "查询员工信息": query_employee_info,
  "发送请假申请": send_leave_application,
  "查询公司规章制度": query_company_info
}
print('工具函数与function.name映射关系建立完成')

# 5. Agent响应处理
def get_agent_response(assistant, message=''):
  # 打印出输入Agent的信息
  thread = Threads.create()
  message = Messages.create(thread.id, content=message)
  run = Runs.create(thread.id, assistant_id=assistant.id)
  run_status = Runs.wait(run.id, thread_id=thread.id)
  # 如果响应失败，会打印出run failed
  if run_status.status == 'failed':
    print('run failed:')
  # 如果需要工具来辅助大模型输出，则进行以下流程
  if run_status.required_action:
    f = run_status.required_action.submit_tool_outputs.tool_calls[0].function
    # 获得function name
    func_name = f['name']
    # 获得function 的入参
    param = json.loads(f['arguments'])
    # 打印出工具信息
    print("function is: ", f)
    # 根据function name，通过function_mapper映射到函数，并将参数输入工具函数得到output输出
    if func_name in function_mapper:
      output = function_mapper[func_name](**param)
    else:
      output = ""
    tool_outputs = [{
      'output': output
    }]
    run = Runs.submit_tool_outputs(
      run.id,
      thread_id=thread.id,
      tool_outputs=tool_outputs
    )
    run_status = Runs.wait(run.id, thread_id=thread.id)
  run_status = Runs.get(run.id,thread_id=thread.id)
  msgs = Messages.list(thread.id)
  # 将Agent的输出返回
  return msgs['data'][0]['content'][0]['text']['value']

# 6. 测试函数
def test_three_tools():
  query_employee_info("教育部门有几个人")
  print(send_leave_application("明后两天"))
  print(query_company_info("我们公司项目管理应该用什么工具"))

# 7. 主函数测试
if __name__ == '__main__':
  # test_three_tools()

  query_stk = [
    "谁是张三的HR？",
    "教育部门一共有多少员工？",
    "王五在哪个部门？",
    "帮我提交下周三请假的申请",
    "我们公司应该用什么项目管理工具呢？"
  ]
  for query in query_stk:
    print("提问是：")
    print(query)
    print("思考过程与最终输出是：")
    print(get_agent_response(ChatAssistant, query))
    print("\n")
