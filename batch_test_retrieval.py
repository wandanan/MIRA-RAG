# batch_test_retrieval.py

import yaml
import json
import csv
import time
import os
from typing import Dict, List, Any, Literal, Tuple
from tqdm import tqdm
from core import MiraRAGConfig, MiraRAGRetriever, DocumentProcessor

HitStatus = Literal["SUB_HIT", "PARENT_HIT", "MISS"]

def load_config(config_path: str = "config.yml") -> Dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_test_set(test_set_path: str) -> List[Dict]:
    """加载JSON格式的测试集"""
    with open(test_set_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def check_hit_levels(retrieved_contexts: List[Tuple[int, float, str, str]], keywords: List[str]) -> Tuple[HitStatus, int, int]:
    """
    分层检查命中情况：先检查子文档，再检查父文档。
    
    Returns:
        - "SUB_HIT": 关键词在子文档中找到。
        - "PARENT_HIT": 关键词未在子文档中找到，但在父文档中找到。
        - "MISS": 在子文档和父文档中都未找到。
    """
    if not keywords:
        return "SUB_HIT"  # 如果没有关键词，算作直接命中

    sub_context_text = " ".join([ctx[2] for ctx in retrieved_contexts])
    
    # 1. 优先检查子文档
    for keyword in keywords:
        if keyword in sub_context_text:
            sub_char_count = sum(len(ctx[2]) for ctx in retrieved_contexts)
            parent_char_count = sum(len(ctx[3]) for ctx in retrieved_contexts)
            return "SUB_HIT", sub_char_count, parent_char_count
            
    # 2. 如果子文档未命中，再检查父文档
    parent_context_text = " ".join([ctx[3] for ctx in retrieved_contexts])
    for keyword in keywords:
        if keyword in parent_context_text:
            sub_char_count = sum(len(ctx[2]) for ctx in retrieved_contexts)
            parent_char_count = sum(len(ctx[3]) for ctx in retrieved_contexts)
            return "PARENT_HIT", sub_char_count, parent_char_count
            
    # 3. 如果都未命中
    # 计算子文档和父文档的总字符数
    sub_char_count = sum(len(ctx[2]) for ctx in retrieved_contexts)
    parent_char_count = sum(len(ctx[3]) for ctx in retrieved_contexts)
    return "MISS", sub_char_count, parent_char_count

def run_retrieval_test(config_dict: Dict[str, Any], test_set: List[Dict], output_path: str) -> Dict[str, float]:
    """
    运行检索评估，并返回详细的命中率统计。
    """
    start_time = time.time()
    
    # 1. 初始化组件
    print("\n--- [1/3] 初始化组件 ---")
    doc_processor = DocumentProcessor(**config_dict['document_processor'])
    print(f"  - 加载知识库: {config_dict['data']['corpus_path']}")
    documents = doc_processor.load_and_split(config_dict['data']['corpus_path'])
    print(f"  - 成功分割为 {len(documents)} 个子文档")
    
    print("  - 初始化检索引擎...")
    retriever_config = MiraRAGConfig(**config_dict['retriever'])
    retriever = MiraRAGRetriever(retriever_config)
    print("  - 开始构建索引...")
    retriever.build_index(documents)
    print("  - 组件初始化完成。")
    
    # 2. 执行评估
    print("\n--- [2/3] 开始执行检索评估 ---")
    detailed_results = []
    hit_counts = {"SUB_HIT": 0, "PARENT_HIT": 0, "MISS": 0}
    total_sub_chars = 0
    total_parent_chars = 0
    
    test_iterator = tqdm(test_set, desc="  - 正在测试", unit="query")
    
    for test_case in test_iterator:
        query = test_case['query']
        keywords = test_case['expected_answer_keywords']
        test_iterator.set_postfix_str(f"Query: {query[:25]}...")
        
        # 确保query是字符串类型
        if not isinstance(query, str):
            print(f"警告: 查询参数类型错误，期望str，实际得到{type(query)}。尝试转换...")
            query = str(query)
        
        retrieved_contexts = retriever.retrieve(query)
        hit_status, sub_char_count, parent_char_count = check_hit_levels(retrieved_contexts, keywords)
        
        hit_counts[hit_status] += 1
        total_sub_chars += sub_char_count
        total_parent_chars += parent_char_count

        detailed_results.append({
            "query": query,
            "expected_keywords": ", ".join(keywords),
            "hit_status": hit_status,
            "sub_chars": sub_char_count,
            "parent_chars": parent_char_count
        })
    
    end_time = time.time()
    
    # 3. 计算结果
    total_count = len(test_set)
    if total_count == 0:
        return {"total_hit_rate": 0, "sub_hit_rate": 0, "parent_hit_rate": 0}

    sub_hit_rate = hit_counts["SUB_HIT"] / total_count
    parent_hit_rate = hit_counts["PARENT_HIT"] / total_count
    total_hit_rate = sub_hit_rate + parent_hit_rate # 总命中率 = 子文档命中 + 父文档命中

    print(f"\n  - 评估完成，耗时: {end_time - start_time:.2f} 秒")
    
    # 4. 保存结果到文件
    print(f"\n--- [3/3] 保存测试结果 ---")
    print(f"  - 正在将详细结果保存到: {output_path}")
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['query', 'expected_keywords', 'hit_status', 'sub_chars', 'parent_chars']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_results)
    print("  - 保存成功。")
    
    return {
        "total_hit_rate": total_hit_rate,
        "sub_hit_rate": sub_hit_rate,
        "parent_hit_rate": parent_hit_rate,
        "miss_rate": hit_counts["MISS"] / total_count,
        "avg_sub_chars": total_sub_chars / total_count if total_count > 0 else 0,
        "avg_parent_chars": total_parent_chars / total_count if total_count > 0 else 0
    }

def main():
    """主测试脚本"""

    print("===== 开始批量检索模式评估 (分层命中分析) =====")
    
    base_config = load_config("config.yml")

    # test_set_path = base_config['data'].get('test_set_path', 'data/novel_test_cases_full.json')
    test_set_path = r'D:\Small_APP\MIRA-RAG\data\novel_test_cases_full.json'
    test_set = load_test_set(test_set_path)
    results_dir = base_config['data'].get('results_dir', 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"创建结果目录: {results_dir}")
    
    test_scenarios = {
        "单头模式 (Single-Head)": {"use_multi_head": False},
        "多头模式 (Multi-Head)": {"use_multi_head": True}
    }
    
    all_results = {}
    total_scenarios = len(test_scenarios)

    for i, (scenario_name, scenario_params) in enumerate(test_scenarios.items(), 1):
        print(f"\n=============================================")
        print(f"  场景 [{i}/{total_scenarios}]: {scenario_name}")
        print(f"=============================================")
        
        import copy
        current_config = copy.deepcopy(base_config)
        
        for key, value in scenario_params.items():
            current_config['retriever'][key] = value
            
        output_path = f"{results_dir}/{scenario_name.replace(' ', '_').lower()}_detailed_results.csv"
        
        evaluation_metrics = run_retrieval_test(current_config, test_set, output_path)
        all_results[scenario_name] = evaluation_metrics
        if not base_config['retriever']['use_colbert_rerank']:
            scenario_name = '仅召回'

        print(f"\n--- {scenario_name} 测试结果 ---")
        print(f"  - 总命中率 (Total Hit Rate): {evaluation_metrics['total_hit_rate']:.2%}")
        print(f"    - 子文档命中率 (Sub-Chunk Hit): {evaluation_metrics['sub_hit_rate']:.2%}")
        print(f"    - 父文档命中率 (Parent-Chunk Hit): {evaluation_metrics['parent_hit_rate']:.2%}")
        print(f"  - 未命中率 (Miss Rate): {evaluation_metrics['miss_rate']:.2%}")
        print(f"  - 平均子文档字符数: {evaluation_metrics['avg_sub_chars']:.0f}")
        print(f"  - 平均父文档字符数: {evaluation_metrics['avg_parent_chars']:.0f}")

    print("\n\n===== 所有测试模式评估完成 =====")
    print("最终结果对比:")
    for name, metrics in all_results.items():
        print(f"\n  模式: {name}")
        print(f"    - 总命中率: {metrics['total_hit_rate']:.2%}")
        print(f"      (其中，子文档直接命中: {metrics['sub_hit_rate']:.2%}, 父文档补充命中: {metrics['parent_hit_rate']:.2%})")
        print(f"      平均文档长度 - 子文档: {metrics['avg_sub_chars']:.0f} 字符, 父文档: {metrics['avg_parent_chars']:.0f} 字符")
    
    print("\n评估结果的详细信息已保存到 'results' 文件夹中。")

if __name__ == "__main__":
    main()