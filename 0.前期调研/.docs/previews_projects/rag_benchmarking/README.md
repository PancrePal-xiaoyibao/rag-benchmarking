# RAG 基准测试系统

一个专用于评估医疗领域检索增强生成（RAG）系统的综合基准测试平台。

## 🎯 项目目标

- **标准化RAG评估**：为医疗RAG系统创建统一的基准测试框架
- **医疗领域专注**：专门针对中文医疗应用的评估
- **开源共建**：构建社区驱动的RAG评估标准
- **全面指标**：多维度评估RAG性能

## 🏗️ 系统架构

```
rag_benchmarking/
├── src/                    # 源代码
│   ├── retrieval/         # 检索评估组件
│   ├── generation/        # 生成评估组件  
│   ├── evaluation/        # 核心评估逻辑
│   ├── datasets/          # 数据集管理
│   ├── metrics/           # 评估指标
│   ├── visualization/     # 结果可视化
│   └── reports/           # 报告生成
├── tests/                 # 测试套件
├── data/                  # 数据存储
│   ├── datasets/          # 测试数据集
│   ├── cache/             # 缓存结果
│   └── results/           # 评估结果
├── docs/                  # 文档
├── configs/               # 配置文件
└── examples/              # 使用示例
```

## 🚀 快速开始

### 安装

```bash
cd rag_benchmarking
pip install -e .
```

### 基础使用

```python
from rag_benchmarking import RAGEvaluator

# 初始化评估器
evaluator = RAGEvaluator()

# 运行评估
results = evaluator.evaluate(
    rag_system="your_rag_system",
    dataset="medical_qa",
    metrics=["retrieval", "generation", "end_to_end"]
)

# 生成报告
report = evaluator.generate_report(results)
```

## 📊 评估维度

### 检索评估
- **Precision@K**：前K个检索文档的精确率
- **Recall@K**：前K个检索文档的召回率
- **MRR**：平均倒数排名
- **NDCG**：归一化折损累积增益
- **Latency**：检索时间性能

### 生成评估
- **答案相关性**：答案对问题的匹配程度
- **忠实度**：与检索上下文的事实一致性
- **完整性**：相关信息覆盖度
- **医疗准确性**：领域特定的正确性
- **幻觉检测**：识别虚构信息

### 端到端评估
- **用户满意度**：整体质量评估
- **上下文利用**：有效使用检索信息
- **临床安全性**：医疗应用的安全性
- **专业性**：医疗专业标准

## 🏥 医疗领域特性

- **中文医疗数据**：专门针对中文医疗文献
- **临床指南**：基于医疗标准的评估
- **多粒度**：从句子级到文档级的评估
- **专业指标**：医疗特定的评估标准

## 🔧 配置

配置文件位于 `configs/` 目录：

- `evaluation.yaml`：评估参数
- `datasets.yaml`：数据集配置
- `metrics.yaml`：指标定义
- `models.yaml`：模型配置

## 📈 报告功能

- **HTML报告**：交互式可视化报告
- **JSON结果**：机器可读的结果
- **PDF导出**：可发布的报告
- **API访问**：程序化访问结果

## 🤝 贡献指南

我们欢迎社区贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细指南。

## 📄 许可证

本项目采用 MIT 许可证 - 详情请查看 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- XXB 社区对本项目的支持
- 医疗领域专家提供的评估指导
- 开源RAG框架的启发 