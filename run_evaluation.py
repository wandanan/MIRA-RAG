# run_evaluation.py
import yaml
import json
import pandas as pd
from typing import Dict, List, Tuple
from core import (
    MiraRAGConfig,
    MiraRAGRetriever,
    DocumentProcessor,
    LLMGenerator,
    evaluate_with_ragas
)


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_test_set(test_set_path: str) -> List[Dict]:
    """加载测试集数据"""
    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)
    return test_set


def main(config_path: str = "config.yml"):
    """主评估流程，支持父文档检索结果"""
    print("===== MIRA-RAG 评估开始 =====")
    
    # 1. 加载配置
    print(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    
    # 2. 初始化文档处理器（使用父/子块参数）
    print("初始化文档处理器...")
    doc_processor = DocumentProcessor(
        parent_chunk_size=config['document_processor']['parent_chunk_size'],
        parent_chunk_overlap=config['document_processor']['parent_chunk_overlap'],
        chunk_size=config['document_processor']['chunk_size'],
        chunk_overlap=config['document_processor']['chunk_overlap']
    )
    
    # 3. 加载并处理知识库（返回子文档+父文档信息列表）
    print(f"加载知识库: {config['data']['corpus_path']}")
    documents = doc_processor.load_and_split(config['data']['corpus_path'])  # 文档结构: [{"sub_chunk_id": ..., "sub_chunk_text": ..., "parent_chunk_text": ...}, ...]
    
    # 4. 初始化检索器
    print("初始化检索引擎...")
    retriever_config = MiraRAGConfig(**config['retriever'])
    retriever = MiraRAGRetriever(retriever_config)
    retriever.build_index(documents)  # 检索器已适配子文档+父文档结构
    
    # 5. 加载测试集
    print(f"加载测试集: {config['data']['test_set_path']}")
    test_set = load_test_set(config['data']['test_set_path'])
    
    # 6. 初始化生成器
    print("初始化LLM生成器...")
    generator = LLMGenerator(
        api_key=config['generator']['api_key'],
        base_url=config['generator']['base_url'],
        model_name=config['generator']['model_name']
    )
    
    # 7. 执行评估（包含父文档信息）
    print("开始评估流程...")
    results_data = []
    for idx, test_case in enumerate(test_set, 1):
        question = test_case['question']
        ground_truth = test_case['ground_truth']
        
        print(f"\n处理测试用例 {idx}/{len(test_set)}: {question}")
        
        # 检索相关上下文（返回子文档+父文档信息）
        print("检索相关上下文...")
        # 检索结果格式: [(子文档ID, 分数, 子文档文本, 父文档文本), ...]
        retrieved_contexts = retriever.retrieve(question)
        
        # 提取子文档文本和父文档文本
        context_texts = [ctx[2] for ctx in retrieved_contexts]
        parent_texts = [ctx[3] for ctx in retrieved_contexts]
        
        # 生成答案（可将父文档文本作为额外上下文传入）
        print("生成答案...")
        answer = generator.generate(question, context_texts, parent_texts)  # 扩展生成器参数以支持父文档
        
        # 存储结果（包含父文档信息）
        results_data.append({
            'question': question,
            'answer': answer,
            'contexts': context_texts,
            'parent_contexts': parent_texts,  # 新增父文档上下文
            'ground_truth': ground_truth
        })
    
    # 8. 评估结果（可基于父文档优化评估逻辑）
    print("\n===== 评估结果 =====")
    evaluation_df = evaluate_with_ragas(results_data)
    
    # 9. 保存结果
    results_dir = config['data']['results_dir']
    results_path = f"{results_dir}/evaluation_results.csv"
    evaluation_df.to_csv(results_path, index=False, encoding='utf-8')
    print(f"评估结果已保存至: {results_path}")
    
    print("\n===== MIRA-RAG 评估完成 =====")


if __name__ == "__main__":
    main()