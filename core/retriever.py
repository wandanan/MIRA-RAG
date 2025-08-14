import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from FlagEmbedding import FlagModel
from rank_bm25 import BM25Okapi

# --- 设备配置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class MiraRAGConfig:
    bge_model_path: str = "BAAI/bge-small-zh-v1.5"
    bm25_top_n: int = 100
    final_top_k: int = 10
    use_multi_head: bool = True   # <-- 开启多头
    num_heads: int = 8  
    embedding_dim: int = 512  # <-- 新增: 嵌入维度，对多头至关重要
    dense_top_n: int = 5  # 稠密召回的候选数量

    use_hybrid_serch: bool = True
    bm25_weight: float = 0.3
    colbert_weight: float = 0.9
    use_length_penalty: bool = True
    length_penalty_alpha: float = 0.45
    multi_channel_recall: bool = False

    use_stateful_reranking: bool = False
    context_memory_decay: float = 0.8  # 记忆衰减率
    context_influence: float = 0.3     # 上下文影响因子
    precompute_doc_tokens: bool = True
    encode_batch_size: int = 32 # 用于批量编码的批次大小
    use_colbert_rerank: bool = True
    # 可选值: 'colbert', 'mamba', 'none' (或未来任何你添加的)
    reranker_type: str = 'mamba' 
@dataclass
class ZipperV3State:
    # 存储原始查询，为未来扩展预留
    original_query: str
    # 核心：代表会话历史的上下文向量
    context_vector: torch.Tensor



class SimpleEncoder:
    """一个只使用BGE模型进行编码的简化版编码器"""
    def __init__(self, model_path: str = "BAAI/bge-small-zh-v1.5"):
        print(f"正在加载模型: {model_path}...")
        self.model = FlagModel(
            model_path, 
            normalize_embeddings=True,
            use_fp16=True if device.type == 'cuda' else False
        )
        self.tokenizer = self.model.tokenizer
        print("模型加载完成。")

    def tokenize(self, text: str) -> List[str]:
        """将文本分割成词元列表，用于BM25。"""
        return self.tokenizer.tokenize(text)

    def encode_tokens(self, text: str) -> torch.Tensor:
        """将单个文本编码为Token向量张量。"""
        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        model_device = self.model.model.device
        encoded = {k: v.to(model_device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model.model(**encoded, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state

        token_embeddings = last_hidden_state[0]
        attention_mask = encoded['attention_mask'][0]
        actual_length = attention_mask.sum().item()
        token_embeddings = token_embeddings[:actual_length]
        
        if token_embeddings.size(0) > 2:
            return token_embeddings[1:-1]
        return token_embeddings

    def encode_sentence(self, text: str) -> torch.Tensor:
        """将单个文本编码为单个句向量"""
        embedding = self.model.encode(
            text,
            convert_to_numpy=False
        )
        return torch.tensor(embedding, device=device).float()
    
    def encode_sentence_batch(self, text: List[str]) -> torch.Tensor:
        """将一批文本编码为句向量矩阵"""
        embedding = self.model.encode(
            text,
            convert_to_numpy=False
        )
        return torch.tensor(embedding, device=device).float()
    
    def encode_tokens_batch(self, texts: List[str], batch_size: int = 64) -> List[torch.Tensor]:
        """高效地对一批文本进行Token级编码，并包含OOM降级重试逻辑。"""
        all_token_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            try:
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                )
                model_device = self.model.model.device
                encoded = {k: v.to(model_device) for k, v in encoded.items()}

                with torch.no_grad():
                    outputs = self.model.model(**encoded, output_hidden_states=True)
                    last_hidden_state = outputs.last_hidden_state
                
                # 逐个处理批次中的结果
                for j in range(len(batch_texts)):
                    token_embeddings = last_hidden_state[j]
                    attention_mask = encoded['attention_mask'][j]
                    actual_length = attention_mask.sum().item()
                    valid_embeddings = token_embeddings[:actual_length]

                    if valid_embeddings.size(0) > 2:
                        all_token_embeddings.append(valid_embeddings[1:-1])
                    else:
                        all_token_embeddings.append(valid_embeddings)
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and len(batch_texts) > 1:
                    print(f"警告: 批次大小 {len(batch_texts)} 导致显存溢出。正在尝试减半处理...")
                    torch.cuda.empty_cache() # 清理显存
                    # 递归处理更小的批次
                    half_batch_size = len(batch_texts) // 2
                    all_token_embeddings.extend(self.encode_tokens_batch(batch_texts[:half_batch_size], half_batch_size))
                    all_token_embeddings.extend(self.encode_tokens_batch(batch_texts[half_batch_size:], half_batch_size))
                else:
                    raise e # 如果是其他错误或无法再拆分，则抛出
        
        return all_token_embeddings

class MiraRAGRetriever:
    def __init__(self, config: MiraRAGConfig):
        print("初始化 MIRA-RAG 检索引擎...")
        self.config = config
        #self.encoder = SimpleEncoder(model_path=config.bge_model_path)
                # --- 新增: 多头维度检查 ---
        if config.use_multi_head and config.embedding_dim % config.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({config.embedding_dim}) 必须能被 num_heads ({config.num_heads}) 整除。"
            )

        self.encoder = SimpleEncoder(model_path=config.bge_model_path)
        # BM25(稀疏)索引
        self.documents: Dict[int, str] = {}  # 这里的文档ID对应子文档ID
        self.bm25_index: BM25Okapi = None
        self.bm25_id_map: Dict[int, int] = {}  # 子文档ID到原始文档ID的映射
        # 稠密索引
        self.doc_sentence_embeddings: torch.Tensor = None
        self.dense_id_map: Dict[int, int] = {}  # 子文档ID到原始文档ID的映射
        self.doc_token_embeddings: Dict[int, torch.Tensor] = {}

    def build_index(self, documents: List[Dict]):
        """
        构建检索索引，包含详细的调试信息。
        """
        print("===== build_index: 开始执行 =====")
        
        # 1. 检查输入类型
        if not isinstance(documents, list):
            print(f"!!!!! FATAL ERROR: 输入的 'documents' 不是列表，而是 {type(documents)} !!!!!")
            raise TypeError("输入给 build_index 的必须是 List[Dict]")
        if documents and not isinstance(documents[0], dict):
            print(f"!!!!! FATAL ERROR: 'documents' 列表中的元素不是字典，而是 {type(documents[0])} !!!!!")
            raise TypeError("documents 列表中的元素必须是字典")
            
        print(f"输入了 {len(documents)} 个文档块。")

        # 2. 构建内部查找表
        try:
            # 确保文档文本是字符串类型
            self.documents = {}
            for doc in documents:
                if not isinstance(doc.get('sub_chunk_text', ''), str):
                    print(f"警告: 文档ID {doc.get('sub_chunk_id')} 的文本类型错误，期望str，实际得到{type(doc.get('sub_chunk_text'))}。尝试转换...")
                    doc['sub_chunk_text'] = str(doc.get('sub_chunk_text', ''))
                self.documents[doc["sub_chunk_id"]] = doc
            print(f"成功构建 self.documents 查找表，大小为: {len(self.documents)}")
        except (KeyError, TypeError) as e:
            print(f"!!!!! FATAL ERROR: 在构建 self.documents 时失败: {e} !!!!!")
            print("请检查输入的 'documents' 列表中字典的结构是否正确，是否包含 'sub_chunk_id'。")
            return

        # 3. 准备 corpus
        doc_ids = sorted(self.documents.keys())
        try:
            corpus = [self.documents[pid]['sub_chunk_text'] for pid in doc_ids]
            print(f"成功构建 corpus 字符串列表，大小为: {len(corpus)}")
        except (KeyError, TypeError) as e:
            print(f"!!!!! FATAL ERROR: 在构建 corpus 时失败: {e} !!!!!")
            print("请检查 self.documents 中字典的结构是否正确，是否包含 'sub_chunk_text'。")
            return

        # 4. 检查 corpus 内容
        for i, item in enumerate(corpus):
            if not isinstance(item, str):
                print(f"!!!!! FATAL ERROR: Corpus 列表在索引 {i} 处包含了一个非字符串元素 !!!!!")
                print(f"类型为: {type(item)}")
                print(f"内容为: {item}")
                # 我们在这里直接终止，因为这就是错误的根源
                raise TypeError(f"Corpus 列表在索引 {i} 处必须是字符串，但收到了 {type(item)}")

        print("Corpus 内容检查通过，所有元素都是字符串。")

        # 5. 构建 BM25 索引
        print('正在创建BM25索引...')
        try:
            tokenized_corpus = [self.encoder.tokenize(text_chunk) for text_chunk in corpus]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            self.bm25_id_map = {i: pid for i, pid in enumerate(doc_ids)}
            print("BM25索引构建成功。")
        except Exception as e:
            print(f"!!!!! FATAL ERROR: 在构建 BM25 索引时发生未知错误: {e} !!!!!")
            # 打印出导致错误的具体文本块
            for i, text_chunk in enumerate(corpus):
                try:
                    self.encoder.tokenize(text_chunk)
                except Exception as inner_e:
                    print(f"在tokenize第 {i} 个文本块时失败，错误: {inner_e}")
                    print(f"该文本块内容: {text_chunk}")
                    break
            return

        # 6. 构建稠密索引
        if self.config.multi_channel_recall:
            print('正在构建稠密向量索引...')
            self.doc_sentence_embeddings = self.encoder.encode_sentence_batch(corpus)
            self.dense_id_map = {i: pid for i, pid in enumerate(doc_ids)}
            print('稠密向量索引构建成功。')

        if self.config.precompute_doc_tokens:
            print("正在预计算所有文档的Token级向量 (用于精排)...")
            # 批量获取所有Token向量
            all_token_vecs = self.encoder.encode_tokens_batch(
                corpus, 
                batch_size=self.config.encode_batch_size
            )
            # 存入字典
            self.doc_token_embeddings = {pid: vec for pid, vec in zip(doc_ids, all_token_vecs)}
            print(f"已预计算并缓存了 {len(self.doc_token_embeddings)} 个文档的Token向量。")

        print("多路召回索引构建完成！")



    def _calculate_colbert_score(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> float:
        if query_emb.nelement() == 0 or doc_emb.nelement() == 0:
            return 0.0

        # --- 多头逻辑 ---
        if self.config.use_multi_head:
            # print(f"多头查询模式")
            # 1. 获取维度信息
            num_query_tokens, q_dim = query_emb.shape
            num_doc_tokens, d_dim = doc_emb.shape
            num_heads = self.config.num_heads
            head_dim = self.config.embedding_dim // num_heads

            # 2. 安全检查：如果实际维度与配置不符，则回退到单头模式
            if q_dim != self.config.embedding_dim or d_dim != self.config.embedding_dim:
                # 这是一个安全网，防止因模型输出维度与配置不符导致崩溃
                print(f"警告: Token维度({q_dim}/{d_dim})与配置({self.config.embedding_dim})不符，回退到单头计算。")
                # 执行原来的单头逻辑
                sim_matrix = F.cosine_similarity(query_emb.unsqueeze(1), doc_emb.unsqueeze(0), dim=-1)
                score = sim_matrix.max(dim=1).values.sum().item()
            else:
                # 3. 变形 (Reshape)
                # [num_tokens, embedding_dim] -> [num_tokens, num_heads, head_dim]
                query_heads = query_emb.view(num_query_tokens, num_heads, head_dim)
                doc_heads = doc_emb.view(num_doc_tokens, num_heads, head_dim)

                total_score = 0.0
                # 4. 循环计算每个头的分数
                for i in range(num_heads):
                    # 取出当前头的所有token向量
                    q_head = query_heads[:, i, :]
                    d_head = doc_heads[:, i, :]
                    
                    # 在当前头内部执行 MaxSim + Sum
                    sim_matrix_head = F.cosine_similarity(q_head.unsqueeze(1), d_head.unsqueeze(0), dim=-1)
                    score_head = sim_matrix_head.max(dim=1).values.sum().item()
                    total_score += score_head
                
                score = total_score

        # --- 单头逻辑 (如果多头关闭) ---
        else:
            # print(f'单头查询模式...')
            sim_matrix = F.cosine_similarity(query_emb.unsqueeze(1), doc_emb.unsqueeze(0), dim=-1)
            score = sim_matrix.max(dim=1).values.sum().item()

        # --- 长度惩罚 (逻辑不变，作用于最终分数) ---
        if self.config.use_length_penalty:
            num_doc_tokens = doc_emb.size(0)
            penalty = 1.0 + self.config.length_penalty_alpha * np.log(num_doc_tokens + 1)
            score /= penalty

        return score

    def update_state(self, state:ZipperV3State,results:List[Tuple[int,float,str]])->ZipperV3State:
        """根据本次检索结果，更新会话状态"""
        if not self.config.use_stateful_reranking or not results:
            return state
        print("正在更新会话状态...")
        # 1. 获取排名第一的文档
        top_doc_pid = results[0][0]
        top_doc_text = self.documents[top_doc_pid]
        # 2. 计算其句向量
        top_doc_emb = self.encoder.encode_sentence(top_doc_text)
        # 3.使用EMA公式更新上下文向量
        decay = self.config.context_memory_decay
        state.context_vector = decay*state.context_vector+(1-decay)*top_doc_emb

        # 4. 更新状态中的查询记录 (可选，但好习惯)
        # 注意：我们这里没有用到 state.original_query，但保留这个字段是为未来扩展
        state.original_query = " | ".join([state.original_query, results[0][2][:20]]) # 简单记录
        
        print("状态更新完成。")
        return state
    
    # 在 AdvancedZipperQueryEngineV5 类中
    def _rerank_with_colbert(self, query: str, final_candidate_pids: List[int], bm25_all_scores: np.ndarray, state: Optional[ZipperV3State] = None) -> List[Tuple[int, float, str]]:
        """使用ColBERT进行精排。"""
        print(f"开始ColBERT精排，处理 {len(final_candidate_pids)} 个候选...")
        
        # --- 1. 获取所有候选的原始分数 (这部分不变) ---
        query_colbert_emb = self.encoder.encode_tokens(query)
        if state and self.config.use_stateful_reranking and state.context_vector.sum() != 0:
            adjustment = state.context_vector.unsqueeze(0) * self.config.context_influence
            query_colbert_emb += adjustment

        colbert_scores_list = []
        candidate_docs_data = [] 
        bm25_scores_map = {self.bm25_id_map[i]: score for i, score in enumerate(bm25_all_scores)}
        if self.config.use_multi_head:
            print('多头查询模式...')
        else:
            print('单头查询模式...')
        for pid in final_candidate_pids:
            doc = self.documents[pid]
            doc_text = doc['sub_chunk_text']
            parent_text = doc.get('parent_chunk_text', '')
            if pid in self.doc_token_embeddings:
                doc_emb = self.doc_token_embeddings[pid]
            else:
                doc_emb = self.encoder.encode_tokens(doc_text)
            colbert_score = self._calculate_colbert_score(query_colbert_emb, doc_emb)
            colbert_scores_list.append(colbert_score)
            bm25_score = bm25_scores_map.get(pid, 0.0)
            candidate_docs_data.append({'id': pid, 'text': doc_text, 'parent_text': parent_text, 'bm25_score': bm25_score})

        # --- 2. 核心修改: 使用Z-Score标准化代替Min-Max ---
        bm25_scores_in_candidates = np.array([d['bm25_score'] for d in candidate_docs_data])
        colbert_scores_in_candidates = np.array(colbert_scores_list)

        # 计算均值和标准差
        bm25_mean, bm25_std = np.mean(bm25_scores_in_candidates), np.std(bm25_scores_in_candidates)
        colbert_mean, colbert_std = np.mean(colbert_scores_in_candidates), np.std(colbert_scores_in_candidates)

        # --- 3. 融合与结果生成 ---
        final_results = []
        for i, data in enumerate(candidate_docs_data):
            # 计算Z-Score
            # 防止标准差为0（当所有值都相同时）
            norm_bm25 = 0.0
            if bm25_std > 1e-9:
                norm_bm25 = (bm25_scores_in_candidates[i] - bm25_mean) / bm25_std
            
            norm_colbert = 0.0
            if colbert_std > 1e-9:
                norm_colbert = (colbert_scores_in_candidates[i] - colbert_mean) / colbert_std
            
            # 打印出来观察一下
            # print(f"PID: {data['id']}, norm_bm25: {norm_bm25:.4f}, norm_colbert: {norm_colbert:.4f}")
            
            fused_score = (self.config.bm25_weight * norm_bm25 +
                        self.config.colbert_weight * norm_colbert)
            final_results.append((data['id'], fused_score, data['text'], data['parent_text']))
        
        return final_results

    # 在 AdvancedZipperQueryEngineV5 类中
    def _rerank_with_mamba_style(self, query: str, final_candidate_pids: List[int], bm25_all_scores: np.ndarray) -> List[Tuple[int, float, str]]:
        """使用Mamba风格的循环状态模型进行精排。"""
        print(f"开始Mamba风格精排，处理 {len(final_candidate_pids)} 个候选...")
        
        bm25_scores_map = {self.bm25_id_map[i]: score for i, score in enumerate(bm25_all_scores)}
        sorted_pids = sorted(final_candidate_pids, key=lambda pid: bm25_scores_map.get(pid, 0.0), reverse=True)

        query_emb = self.encoder.encode_sentence(query)
        state_vector_h = torch.zeros_like(query_emb)

        decay_factor = 0.9
        selection_strength = 2.0
        context_influence = 0.5

        reranked_results = []
        for pid in sorted_pids:
            doc_text = self.documents[pid]['sub_chunk_text']
            doc_emb = self.encoder.encode_sentence(doc_text)
            
            # 确保输入是2D张量 [1, embedding_dim]
            query_emb_2d = query_emb.unsqueeze(0)
            doc_emb_2d = doc_emb.unsqueeze(0)
            gate = torch.sigmoid(selection_strength * F.cosine_similarity(query_emb_2d, doc_emb_2d))
            state_vector_h = decay_factor * state_vector_h + (1 - decay_factor) * gate * doc_emb
            
            # 确保所有相似度计算使用2D张量
            direct_score = F.cosine_similarity(query_emb_2d, doc_emb_2d)
            context_score = F.cosine_similarity(query_emb_2d, state_vector_h.unsqueeze(0))
            final_score = direct_score + context_influence * context_score
            
            reranked_results.append((pid, final_score.item(), doc_text, self.documents[pid]['parent_chunk_text']))
            
        return reranked_results

    def retrieve(self, query: str, state: Optional[ZipperV3State] = None) -> List[Tuple[int, float, str]]:
        """
        统一的检索入口。执行多路召回，然后根据配置分发给不同的精排器。
        """
        if not self.config.use_colbert_rerank:
            print(f"\n开始检索：'{query}' (跳过ColBERT精排)")
            self.config.reranker_type = 'none'
        else:
            print(f"\n开始检索：'{query}' (使用 '{self.config.reranker_type}' 精排器)")
        
        # --- 步骤 1: 多路召回 (此部分逻辑保持统一和默认) ---
        query_tokens = self.encoder.tokenize(query)
        bm25_all_scores = self.bm25_index.get_scores(query_tokens)
        bm25_candidate_indices = np.argsort(bm25_all_scores)[::-1][:self.config.bm25_top_n]
        final_candidate_pids = {self.bm25_id_map[idx] for idx in bm25_candidate_indices}

        if self.config.multi_channel_recall:
            query_sentence_emb = self.encoder.encode_sentence(query)
            dense_scores = torch.matmul(query_sentence_emb, self.doc_sentence_embeddings.T)
            dense_candidate_indices = torch.topk(dense_scores, k=self.config.dense_top_n).indices
            dense_pids = {self.dense_id_map[idx.item()] for idx in dense_candidate_indices}
            final_candidate_pids.update(dense_pids)
        
        final_candidate_pids = list(final_candidate_pids)
        if not final_candidate_pids:
            return []
        
        # --- 步骤 2: 根据配置分发到不同的精排器 ---
        if not self.config.use_colbert_rerank:
            print(f"跳过精排，共 {len(final_candidate_pids)} 个候选，仅使用BM25分数排序...")
            bm25_scores_map = {self.bm25_id_map[i]: score for i, score in enumerate(bm25_all_scores)}
            final_results = [
                (pid, bm25_scores_map.get(pid, 0.0),
                self.documents[pid]['sub_chunk_text'],
                self.documents[pid]['parent_chunk_text']
            ) for pid in final_candidate_pids]
        elif self.config.reranker_type == 'colbert':
            final_results = self._rerank_with_colbert(query, final_candidate_pids, bm25_all_scores, state)
        
        elif self.config.reranker_type == 'mamba':
            final_results = self._rerank_with_mamba_style(query, final_candidate_pids, bm25_all_scores)
            bm25_scores_map = {self.bm25_id_map[i]: score for i, score in enumerate(bm25_all_scores)}
            final_results = [
                (pid, bm25_scores_map.get(pid, 0.0),
                self.documents[pid]['sub_chunk_text'],
                self.documents[pid]['parent_chunk_text']
            ) for pid in final_candidate_pids]
            
        else:
            raise ValueError(f"未知的精排器类型: {self.config.reranker_type}")

        # --- 步骤 3: 最终排序和返回 ---
        # 确保所有结果都是四元组
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:self.config.final_top_k]


if __name__ == '__main__':
    sample_docs = {
        # ... (sample_docs 内容不变) ...
        1: "深度学习彻底改变了自然语言处理领域。",
        2: "BM25是一种基于词频的经典信息检索算法。",
        3: """苹果公司发布的这款新款 iPhone，除了配备强大的 A 系列芯片...""",
        4: "如何使用PyTorch构建一个简单的神经网络？",
        5: "ColBERT通过晚期交互机制，实现了更精细的语义匹配。",
        6: "传统的搜索引擎严重依赖关键词匹配。",
        7: "iPhone的摄像头系统非常出色，尤其在低光环境下表现优异。" # 新增一条相关文档
    }

    print('\n---创建引擎配置---')
    config = MiraRAGConfig(
        embedding_dim=512,
        multi_channel_recall=True,
        use_stateful_reranking=True # <-- 确保开启状态管理
    )
    engine = MiraRAGRetriever(config) # <-- 使用 V5 引擎
    engine.build_index(sample_docs)
    
    # --- 模拟对话式搜索 ---

    # 1. 第一次查询
    query1 = "苹果最新的手机是什么"
    print(f"\n\n=============== 对话轮次 1 ===============\n查询: '{query1}'")
    
    # 初始化一个空的会话状态
    initial_context_vector = torch.zeros(config.embedding_dim, device=device)
    state = ZipperV3State(original_query=query1, context_vector=initial_context_vector)
    
    results1 = engine.retrieve(query1, state)
    print("\n--- 第一次查询结果 ---")
    for doc_id, score, text in results1:
        print(f"  ID: {doc_id}, Score: {score:.4f}, Text: {text[:50]}...")
    
    # 2. 更新状态
    state = engine.update_state(state, results1)

    # 3. 第二次查询 (模糊查询，依赖上下文)
    query2 = "它的拍照效果怎么样"
    print(f"\n\n=============== 对话轮次 2 ===============\n查询: '{query2}'")
    
    results2 = engine.retrieve(query2, state)
    print("\n--- 第二次查询结果 (已应用上下文) ---")
    for doc_id, score, text in results2:
        print(f"  ID: {doc_id}, Score: {score:.4f}, Text: {text[:50]}...")