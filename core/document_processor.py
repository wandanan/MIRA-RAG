#document_processor.py
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """
    文档处理器类，负责加载和切分知识库文档，支持父文档检索技术
    """
    def __init__(self, parent_chunk_size: int, parent_chunk_overlap: int, chunk_size: int, chunk_overlap: int):
        """
        初始化文档处理器
        
        Args:
            parent_chunk_size: 父文档块大小
            parent_chunk_overlap: 父文档块重叠大小
            chunk_size: 子文档块大小
            chunk_overlap: 子文档块重叠大小
        """
        # 父文档分割器
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""]
        )
        # 子文档分割器
        self.sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""]
        )

    def load_and_split(self, file_path: str) -> List[Dict]:
        """
        从文件加载文本并进行两级拆分（父文档->子文档）
        
        Args:
            file_path: 输入文件路径
            
        Returns:
            包含子文档和父文档信息的列表，每个元素为:
            {
                "sub_chunk_id": 子文档ID,
                "sub_chunk_text": 子文档内容,
                "parent_chunk_id": 父文档ID,
                "parent_chunk_text": 父文档内容
            }
        """
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 1. 分割为父文档块
        parent_chunks = self.parent_splitter.split_text(text)
        
        # 2. 对每个父文档块分割为子文档块，并记录对应父文档
        sub_chunk_list = []
        global_sub_chunk_id = 0 #全局计数器
        for parent_idx, parent_chunk in enumerate(parent_chunks):
            # 分割为子文档块
            sub_chunks = self.sub_splitter.split_text(parent_chunk)
            # 为每个子文档块添加父文档信息
            for sub_chunk in sub_chunks:  # 直接迭代文本内容
                sub_chunk_list.append({
                    "sub_chunk_id": global_sub_chunk_id,
                    "sub_chunk_text": sub_chunk,
                    "parent_chunk_id": parent_idx,
                    "parent_chunk_text": parent_chunk
                })

                global_sub_chunk_id +=1
        return sub_chunk_list