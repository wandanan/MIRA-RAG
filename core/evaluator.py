from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)


def evaluate_with_ragas(results_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    使用Ragas库对RAG系统生成的结果进行自动化评估
    
    Args:
        results_data: 包含评估数据的列表，每个元素是包含以下键的字典:
            - 'question': 用户问题
            - 'answer': 模型生成的答案
            - 'contexts': 检索到的上下文列表
            - 'ground_truth': 标准答案
            
    Returns:
        包含评估结果的Pandas DataFrame
    """
    # 将结果数据转换为Hugging Face Dataset对象
    dataset = Dataset.from_list(results_data)
    
    # 定义要使用的评估指标
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
    
    # 执行评估
    results = evaluate(dataset, metrics=metrics)
    
    # 将结果转换为DataFrame
    evaluation_df = results.to_pandas()
    
    return evaluation_df