"""文档分级 Agent —— 三路并行匹配 + 融合决策 + 人工复核。

模块：
    parse     文档解析（PaddleOCR 的公开替身：txt/docx/pdf → 段落 → 句子）
    index     倒排索引（ES 替身）+ TF-IDF 向量索引（Milvus 替身）+ RRF
    channels  三个通道：敏感词 / 敏感句 / 语义规则
    fusion    权重融合 + 冲突检测 + 转人工策略
    agent     编排（并行、降级、结构化输出、JSONL 日志）
"""

__version__ = "0.1.0"
