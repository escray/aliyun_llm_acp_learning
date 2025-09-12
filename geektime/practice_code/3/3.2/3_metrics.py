from datasets import Dataset
from ragas.metrics import answer_correctness
from ragas import evaluate

from langchain_community.llms.tongyi import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings

data_samples = {
  'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
}

dataset = Dataset.from_dict(data_samples)
score = evaluate(
  dataset = dataset,
  metrics=[answer_correctness],
  llm = Tongyi(model_name="Qwen-plus"),
  embeddings = DashScopeEmbeddings(model="text-embeddings-v3")
)
score.to_pandas()
