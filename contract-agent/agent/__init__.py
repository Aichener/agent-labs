"""合同/条款核查 Agent —— 自研执行循环 + 评测。

模块划分：
    llm     模型接入（OpenAI 兼容 / 离线 Mock），零第三方依赖
    tools   4 个工具：条款切分、规则核查、检索、修复建议
    policy  规则库、计划兜底、结论聚合（确定性逻辑，便于测试）
    loop    自研执行循环：规划 → 调用 → 观测 → 收敛，含步数/成本护栏
    graph   LangGraph 等价实现（需安装 langgraph）
    cli     命令行 demo
"""

__version__ = "0.1.0"
