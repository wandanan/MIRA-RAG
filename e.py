# 在 AdvancedZipperQueryEngineV4.retrieve 方法中
def retrieve(self, query: str) -> List[Tuple[int, float, str]]:
    print(f"\n开始检索：'{query}'")
    
    num_docs = len(self.documents)
    if num_docs == 0:
        return []

    # --- 1. BM25 稀疏召回 (始终执行) ---
    k_for_bm25 = min(self.config.bm25_top_n, num_docs) # 安全的 k 值
    query_tokens = self.encoder.tokenize(query)
    bm25_all_scores = self.bm25_index.get_scores(query_tokens)
    bm25_candidate_indices = np.argsort(bm25_all_scores)[::-1][:k_for_bm25] # 使用安全的k
    bm25_candidate_pids = {self.bm25_id_map[idx] for idx in bm25_candidate_indices}
    print(f"BM25 召回了 {len(bm25_candidate_pids)} 个候选。")
    
    final_candidate_pids = set(bm25_candidate_pids)

    # --- 2. (可选) 稠密向量召回 ---
    if self.config.multi_channel_recall and self.doc_sentence_embeddings is not None:
        print("执行稠密向量召回...")
        k_for_dense = min(self.config.dense_top_n, num_docs) # 安全的 k 值
        if k_for_dense > 0:
            query_sentence_emb = self.encoder.encode_sentence(query)
            dense_scores = torch.matmul(query_sentence_emb, self.doc_sentence_embeddings.T)
            dense_candidate_indices = torch.topk(dense_scores, k=k_for_dense).indices
            dense_candidate_pids = {self.dense_id_map[idx.item()] for idx in dense_candidate_indices}
            print(f"稠密向量召回了 {len(dense_candidate_pids)} 个候选。")
            final_candidate_pids.update(dense_candidate_pids)