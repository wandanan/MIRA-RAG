from openai import OpenAI
from typing import List, Dict, Any


class LLMGenerator:
    """
    LLM生成器类，负责调用大语言模型生成答案，支持父文档上下文
    """
    def __init__(self, api_key: str, base_url: str, model_name: str):
        """
        初始化LLM生成器
        
        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL
            model_name: LLM模型名称
        """
        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate(self, query: str, contexts: List[str], parent_contexts: List[str] = None) -> str:
        """
        调用LLM生成答案，支持父文档上下文
        
        Args:
            query: 用户查询问题
            contexts: 检索到的子文档文本列表
            parent_contexts: 对应的父文档文本列表（可选）
            
        Returns:
            生成的答案字符串
        """
        # 构建上下文文本（包含子文档和父文档）
        context_str = "\n\n".join([f"子文档: {ctx}" for ctx in contexts])
        if parent_contexts:
            context_str += "\n\n" + "\n\n".join([f"父文档: {pctx}" for pctx in parent_contexts])
        char_count = 0          # 初始化计数器
        for _ in context_str:          # 遍历文本串中的每个字符
            char_count += 1     # 每个字符对应计数器+1
        print(f'上下文总数：{len(context_str)}') 
        with open("results/output.txt", "w", encoding="utf-8") as file:
            file.write(context_str)  # 将字符串写入文件
        # 构建提示词
        prompt = f"""
        基于以下上下文信息回答用户问题。如果上下文信息不足以回答问题，请说明"无法基于提供的信息回答该问题"。
        
        上下文信息:
        {context_str}
        
        用户问题: {query}
        
        回答:
        """
        return 'a'
        # 调用OpenAI API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # 设置为0.0以获得更确定的结果
            max_tokens=1000
        )
        
        # 提取并返回答案
        answer = response.choices[0].message.content.strip()
        return answer