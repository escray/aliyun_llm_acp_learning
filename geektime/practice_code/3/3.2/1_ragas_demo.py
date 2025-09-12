# =============== 导入必要的库 ===============
# Tongyi: 通义千问大语言模型,用于生成文本回答
from langchain_community.llms.tongyi import Tongyi

# DashScopeEmbeddings: 百炼文本向量化模型,用于将文本转换为向量
from langchain_community.embeddings import DashScopeEmbeddings

# Dataset: Hugging Face的数据集工具,用于创建和管理数据集
from datasets import Dataset

# ragas评估工具及相关指标
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision, answer_relevancy, answer_correctness

"""
ragas评估指标说明:
- answer_correctness: 评估答案的正确性
- context_recall: 评估上下文的召回率
- context_precision: 评估上下文的精确度
- faithfulness: 评估答案与上下文的一致性
- answer_relevancy: 评估答案与问题的相关性
"""

# =============== 准备示例数据 ===============
data_samples = {
  'question': [  # 测试问题列表
      '张伟是哪个部门的？',
      '张伟是哪个部门的？',
      '张伟是哪个部门的？'
  ],
  'answer': [  # 系统生成的不同答案,用于评估答案质量
      '根据提供的信息，没有提到张伟所在的部门。如果您能提供更多关于张伟的信息，我可能能够帮助您找到答案。',
      '张伟是人事部门的',
      '张伟是教研部的'
  ],
  'ground_truth':[  # 标准答案,用于对比评估
      '张伟是教研部的成员',
      '张伟是教研部的成员',
      '张伟是教研部的成员'
  ]
}

# =============== 创建数据集 ===============
# 将字典格式数据转换为Dataset对象,便于评估使用
dataset = Dataset.from_dict(data_samples)

# =============== 进行评估 ===============
score = evaluate(
  dataset = dataset, # 要评估的数据集
  metrics = [answer_correctness], # 使用answer_correctness指标评估答案的正确性
  llm = Tongyi(model_name = 'qwen-plus'),    # 使用通义千问plus模型作为评估模型
  embeddings = DashScopeEmbeddings(model = 'text-embedding-v3') # 使用DashScope的文本向量化模型进行文本编码
)

# 将评估结果转换为pandas数据框并打印
print(score.to_pandas())
