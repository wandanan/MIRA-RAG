# MIRA-RAG (多交互检索增强生成)

一个用于RAG管道端到端评估的可配置框架。

## 项目概述

MIRA-RAG实现了完整的RAG评估工作流：
1. 参数配置
2. 文档分割
3. 多交互检索(MIRA)
4. LLM生成答案
5. Ragas自动化评估

整个项目由中央YAML配置文件驱动，并将实验结果输出到指定目录。

## 项目结构

```
MIRA-RAG/
├── .gitignore           # Git忽略文件
├── README.md            # 项目文档
├── requirements.txt     # 依赖项列表
├── config.yml           # 核心配置文件
├── run_evaluation.py    # 批量评估主脚本
├── app.py               # Streamlit交互式演示应用
├── configs/             # 实验配置文件目录
│   └── experiment_advanced.yml  # 高级实验配置示例
├── core/                # 核心逻辑模块目录
│   ├── __init__.py      # 包初始化文件
│   ├── document_processor.py  # 文档处理器
│   ├── retriever.py     # 检索器
│   ├── generator.py     # LLM生成器
│   └── evaluator.py     # 评估器
├── data/                # 数据存放目录
│   ├── corpus.txt       # 知识库源文件
│   └── test_set.json    # 测试问答对
└── results/             # 实验结果存放目录
    └── .gitkeep         # 保持目录存在的空文件
```

## 安装方法

```bash
pip install -r requirements.txt
```

## 使用方法

执行批量评估：
```bash
python run_evaluation.py --config config.yml
```

运行Streamlit演示界面：
```bash
streamlit run app.py
```

## 许可证

[MIT许可证](LICENSE)