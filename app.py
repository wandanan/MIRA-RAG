# app.py
import streamlit as st
import yaml
from typing import List, Dict
from core import MiraRAGConfig, MiraRAGRetriever, DocumentProcessor, LLMGenerator


@st.cache_resource
def load_components(config: Dict):
    """加载并缓存应用所需的核心组件"""
    # 初始化文档处理器
    doc_processor = DocumentProcessor(
        parent_chunk_size=config['document_processor']['parent_chunk_size'],
        parent_chunk_overlap=config['document_processor']['parent_chunk_overlap'],
        chunk_size=config['document_processor']['chunk_size'],
        chunk_overlap=config['document_processor']['chunk_overlap']
    )
    
    # 加载并分割知识库（使用文档处理器生成结构化文档）
    print(f"加载知识库: {config['data']['corpus_path']}")
    documents = doc_processor.load_and_split(config['data']['corpus_path'])  # 生成结构化文档列表
    print(f"成功分割为 {len(documents)} 个子文档")
    
    # 初始化检索器
    retriever_config = MiraRAGConfig(**config['retriever'])
    retriever = MiraRAGRetriever(retriever_config)
    retriever.build_index(documents)  # 传入结构化文档列表
    
    # 初始化生成器
    generator = LLMGenerator(
        api_key=config['generator']['api_key'],
        base_url=config['generator']['base_url'],
        model_name=config['generator']['model_name']
    )
    
    return doc_processor, retriever, generator


def load_config(config_path: str = "config.yml") -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """Streamlit应用主函数"""
    st.set_page_config(page_title="MIRA-RAG Demo", page_icon="🔍")
    st.title("MIRA-RAG (多交互检索增强生成) Demo")
    
    # 加载配置
    config = load_config()
    
    # 侧边栏配置
    with st.sidebar:
        st.header("配置参数")
        parent_chunk_size = st.slider("父文本块大小", 500, 5000, config['document_processor']['parent_chunk_size'])
        chunk_size = st.slider("子文本块大小", 100, 1000, config['document_processor']['chunk_size'])

        chunk_overlap = st.slider("子文本块重叠", 0, 200, config['document_processor']['chunk_overlap'])
        bm25_top_n = st.slider("BM25候选数", 1, 100, config['retriever']['bm25_top_n'])
        dense_top_n = st.slider("稠密向量候选数", 1, 100, config['retriever']['dense_top_n'])
        final_top_k = st.slider("最终检索结果数", 1, 10, config['retriever']['final_top_k'])
        bge_model_path = st.text_input("BGE模型路径", config['retriever']['bge_model_path'])
        multi_channel_recall = st.checkbox("启用多通道检索", config['retriever']['multi_channel_recall'])
        bm25_weight = st.slider("BM25权重", 0.0, 1.0, config['retriever']['bm25_weight'], 0.1)
        colbert_weight = st.slider("ColBERT权重", 0.0, 1.0, config['retriever']['colbert_weight'], 0.1)
        use_length_penalty = st.checkbox("使用长度惩罚", config['retriever']['use_length_penalty'])
        length_penalty_alpha = st.slider("长度惩罚系数", 0.0, 1.0, config['retriever']['length_penalty_alpha'], 0.05)
        # 新增：Retriever额外参数
        use_multi_head = st.checkbox("启用多头检索", config['retriever']['use_multi_head'])
        num_heads = st.slider("头的数量", 1, 32, config['retriever']['num_heads'])
        embedding_dim = st.slider("嵌入维度", 128, 1024, config['retriever']['embedding_dim'])
        model_name = st.text_input("模型名称", config['generator']['model_name'])
        # 新增：Generator额外参数
        api_key = st.text_input("API密钥", config['generator']['api_key'])
        base_url = st.text_input("API基础URL", config['generator']['base_url'])
        
        if st.button("更新配置"):
            # 更新配置
            config['document_processor']['parent_chunk_size'] = parent_chunk_size   
            config['document_processor']['chunk_size'] = chunk_size

            config['document_processor']['chunk_overlap'] = chunk_overlap
            config['retriever']['bm25_top_n'] = bm25_top_n
            config['retriever']['dense_top_n'] = dense_top_n
            config['retriever']['final_top_k'] = final_top_k
            config['generator']['model_name'] = model_name
            # 更新Retriever额外参数
            config['retriever']['use_multi_head'] = use_multi_head
            config['retriever']['num_heads'] = num_heads
            config['retriever']['embedding_dim'] = embedding_dim
            # 更新Generator额外参数
            config['generator']['api_key'] = api_key
            config['generator']['base_url'] = base_url
            
            # 重新加载组件
            doc_processor, retriever, generator = load_components(config)
            st.success("配置已更新，组件已重新加载")
    
    # 加载核心组件
    doc_processor, retriever, generator = load_components(config)
    
    # 用户查询输入
    query = st.text_input("请输入您的问题:")
    
    if query:
        with st.spinner("正在处理您的查询..."):
            # 检索相关上下文（包含父文档信息）
            # 检索结果格式: [(子文档ID, 分数, 子文档文本, 父文档文本), ...]
            retrieved_contexts = retriever.retrieve(query)
            
            # 提取子文档文本和父文档文本
            context_texts = [ctx[2] for ctx in retrieved_contexts]
            parent_texts = [ctx[3] for ctx in retrieved_contexts]
            # 展示父文档
            # 生成答案（传入父文档文本）
            answer = generator.generate(query, context_texts, parent_texts)
            
            # 显示结果
            st.subheader("检索到的上下文与父文档:")
            for i, (_, score, sub_text, parent_text) in enumerate(retrieved_contexts, 1):
                with st.expander(f"上下文 {i} (分数: {score:.4f})"):
                    st.write("**子文档内容:**")
                    st.write(sub_text)
                    st.divider()
                    st.write("**父文档内容:**")
                    st.write(parent_text)
            
            st.subheader("生成的答案:")
            st.write(answer)


if __name__ == "__main__":
    main()