# 导入所需库
from langchain_community.llms.tongyi import Tongyi
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, context_precision

# 构建示例数据集
# question: 待评估的问题
# answer: 系统给出的答案
# ground_truth: 标准答案
# contexts: 检索系统返回的相关上下文片段列表
data_samples = {
  'question': [
      '张伟是哪个部门的？',
      '张伟是哪个部门的？',
      '张伟是哪个部门的？'
  ],
  'answer': [
      '根据提供的信息，没有提到张伟所在的部门。如果您能提供更多关于张伟的信息，我可能能够帮助您找到答案。',
      '张伟是人事部门的',
      '张伟是教研部的'
  ],
  'ground_truth':[
      '张伟是教研部的成员',
      '张伟是教研部的成员',
      '张伟是教研部的成员'
  ],
  'contexts' : [
      ['提供⾏政管理与协调⽀持，优化⾏政⼯作流程。 ', '绩效管理部 韩杉 李⻜ I902 041 ⼈⼒资源'],
      ['李凯 教研部主任 ', '牛顿发现了万有引力'],
      ['牛顿发现了万有引力', '张伟 教研部工程师，他最近在负责课程研发'],
  ],
}

# 将字典数据转换为Dataset格式
dataset = Dataset.from_dict(data_samples)

# 使用ragas进行评估
# context_recall: 评估检索的上下文是否包含回答问题所需的全部信息
#   - 值越高表示检索的上下文越完整地包含了所需信息
#   - 计算方式是分析上下文是否包含支持答案的关键信息
#
# context_precision: 评估检索的上下文与问题的相关性
#   - 值越高表示检索的上下文越相关，噪声越少
#   - 计算方式是分析上下文中有多少内容是与问题相关的
score = evaluate(
  dataset = dataset,
  metrics = [context_recall, context_precision],
  llm = Tongyi(model_name = 'qwen-plus')
)

# 打印评估结果
print(score.to_pandas())
