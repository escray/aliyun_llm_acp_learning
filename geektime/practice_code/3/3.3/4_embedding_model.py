from llama_index.embeddings.dashscope import DashScopeEmbedding
import numpy as np

# 定义余弦相似度计算函数
def cosine_similarity(a, b):
  a = np.array(a)
  b = np.array(b)
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

query = "张伟是哪个部门的"
chunk_1 = "核，提供⾏政管理与协调⽀持，优化⾏政⼯作流程。 ⾏政部 秦⻜ 蔡静 G705 034 ⾏政 ⾏政专员 13800000034 qinf@educompany.com 维护公司档案与信息系统，负责公司通知及公告的发布"
chunk_2 = "组织公司活动的前期准备与后期评估，确保公司各项⼯作的顺利进⾏。 IT部 张伟 ⻢云 H802 036 IT⽀撑 IT专员 13800000036 zhangwei036@educompany.com 进⾏公司⽹络及硬件设备的配置"

text_embedding_v2 = DashScopeEmbedding(model_name = "text-embedding-v2")
text_embedding_v3 = DashScopeEmbedding(model_name = "text-embedding-v3")

print(f"query: {query}")
print(f"chunk_1: {chunk_1}")
print(f"chunk_2: {chunk_2}")

print("/n=== text-embedding-v2 ===")
print(f"""query 和 chunk_1 的相似度：{cosine_similarity(
  text_embedding_v2.get_text_embedding(query),
  text_embedding_v2.get_text_embedding(chunk_1)
)}
""")
print(f"""query 和 chunk_2 的相似度：{cosine_similarity(
  text_embedding_v2.get_text_embedding(query),
  text_embedding_v2.get_text_embedding(chunk_2)
)}
""")

print("/n=== text-embedding-v3 ===")
print(f"""query 和 chunk_1 的相似度：{cosine_similarity(
  text_embedding_v3.get_text_embedding(query),
  text_embedding_v3.get_text_embedding(chunk_1)
)}
""")
print(f"""query 和 chunk_2 的相似度：{cosine_similarity(
  text_embedding_v3.get_text_embedding(query),
  text_embedding_v3.get_text_embedding(chunk_2)
)}
""")

import os
from openai import OpenAI

client = OpenAI(
  api_key=os.getenv("DASHSCOPE_API_KEY"),
  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

completion = client.embeddings.create(
  model="text-embedding-v4",
  input='衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买',
  dimensions=1024,
  encoding_format="float"
)



print(completion.model_dump_json())
