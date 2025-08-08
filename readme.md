
# RAG Benchmarking 项目

## 项目简介

RAG Benchmarking 是一个专注于检索增强生成（Retrieval-Augmented Generation）技术研究与实践的开源项目。项目致力于探索和优化RAG系统的各个环节，包括文档转化、质量评估、智能触发机制等核心技术。

### 项目特色

- **全链路RAG技术栈**：覆盖从文档预处理到质量评估的完整RAG流程
- **多模型支持**：支持OpenAI GPT、Claude、LLaMA、DeepSeek等主流大模型
- **实用工具集**：提供文档转化、QA生成、质量评估等实用工具
- **医疗领域应用**：结合胰腺癌等医疗场景的实际应用案例

## 目录结构

```
rag-benchmarking/
├── readme.md                           # 项目说明文档
├── LLM_benchmarking/                  # LLM基准测试相关研究
│   └── llm_benchmarking_cn.md        # LLM置信度评估与RAG触发机制研究
├── QA_sets/                           # 标准QA数据集合
│   └── InfiniFlow:medical_QA/         # InfiniFlow医疗QA数据集
│       ├── Internal medicine_QA_all.csv      # 内科QA数据
│       ├── Medical Oncology_QA_all.csv       # 肿瘤科QA数据
│       ├── OB GYN_QA_all.csv                 # 妇产科QA数据
│       ├── Pediatrics_QA_all.csv             # 儿科QA数据
│       ├── QA for Andrology.csv              # 男科QA数据
│       ├── Surgical_QA_all.csv               # 外科QA数据
│       ├── README.md                         # 数据集说明
│       └── extract_urls.py                   # URL提取工具
├── Tools/                             # 工具集合
│   ├── MinerU_xyb/                    # MinerU文档转化工具
│   │   └── readme.md                  # MinerU工具使用说明
│   └── QA_auto_gen/                   # QA对自动生成工具
│       ├── auto_gen_QAv3.py          # 基于智谱AI的QA生成脚本
│       └── QA.json                   # 生成的QA数据示例
└── xiaoyibao_Benching_V1.0/          # 小胰宝基准测试集v1.0
    ├── Hornoring_Contributors.md      # 贡献者致谢
    ├── XiaoYiBao 知识库评测集 v1.0.xlsx # 评测数据集
    └── XiaoYiBao 知识库评测集 v1.0 - 一期评测结果.csv # 评测结果
```

## 核心技术模块

### 1. LLM基准测试与智能RAG触发机制 (LLM_benchmarking/)

**技术亮点：**
- 基于置信度评估的动态RAG触发机制
- 支持多种置信度计算方法（Token级概率、序列生成概率、自洽性校验）
- 分层阈值策略（常识类、领域类、实时类）
- 主流大模型适配方案（OpenAI GPT、Claude、LLaMA、DeepSeek等）

**核心功能：**
- 多模型性能对比测试
- 置信度评估能力验证
- 领域知识掌握程度评估
- 推理能力基准测试

**应用场景：**
- 医疗问答系统
- 法律咨询助手
- 技术文档问答

### 2. 标准QA数据集 (QA_sets/)

**数据集特色：**
- **InfiniFlow医疗QA数据集**：涵盖多个医疗专科领域
  - 内科、外科、肿瘤科、妇产科、儿科、男科等
  - 标准化的CSV格式，便于批量测试
  - 支持Apache 2.0开源协议

**数据规模：**
- 多专科医疗问答覆盖
- 结构化数据格式
- 支持自动化评测流程

**应用价值：**
- RAG系统性能基准测试
- 医疗AI模型评估
- 领域知识准确性验证

### 3. 文档转化工具 (Tools/MinerU_xyb/)

**技术特点：**
- 基于上海OpenDataLab的MinerU项目改造
- 支持PDF、Word等多种文档格式
- 图文混排处理能力
- 集成S3存储，支持长期有效的图片链接
- 适配GPU云服务资源

**核心优势：**
- 批量处理能力
- 低成本部署方案
- 高质量文档解析
- 开源项目：[MinerU-xyb](https://github.com/PancrePal-xiaoyibao/MinerU-xyb)

### 4. QA自动生成工具 (Tools/QA_auto_gen/)

**功能特性：**
- 基于智谱AI API的智能QA生成
- 支持单文件和批量处理
- 自动生成文档摘要
- JSON格式标准化输出
- 可配置的生成参数

**技术实现：**
```python
class QAGenerator:
    """
    QA对生成器
    功能：根据文档内容自动生成问答对
    支持：Markdown文件批量处理
    输出：JSON格式的QA数据
    """
```

### 5. 小胰宝基准测试集 (xiaoyibao_Benching_V1.0/)

**数据集特点：**
- 专注胰腺癌医疗领域的专业问答
- 包含标准评测集和评测结果
- 支持RAG系统性能基准测试
- 提供Excel和CSV格式的数据

**评估维度：**
- 准确率 (Accuracy)
- 召回率 (Recall)
- F1值 (F1-Score)
- 专业性评估
- 安全性评估

## 技术栈

### 核心依赖
- **大模型API**：OpenAI GPT、Claude、智谱AI、DeepSeek
- **文档处理**：MinerU、PyPDF2、python-docx
- **向量检索**：FAISS、Chroma、Pinecone
- **评估工具**：RAGAS、自研评估框架

### 部署环境
- **云服务**：腾讯云、阿里云、AWS
- **容器化**：Docker、Kubernetes
- **存储**：MinIO、S3兼容存储
- **GPU支持**：CUDA、ROCm

## 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone https://github.com/your-org/rag-benchmarking.git
cd rag-benchmarking

# 安装依赖
pip install -r requirements.txt
```

### 2. 使用标准QA数据集进行测试
```bash
# 查看医疗QA数据集
cd QA_sets/InfiniFlow:medical_QA/
ls *.csv  # 查看各专科QA数据

# 使用Python加载数据集
import pandas as pd
df = pd.read_csv('Internal medicine_QA_all.csv')
print(f"内科QA数据集包含 {len(df)} 条问答对")
```

### 3. QA自动生成示例
```bash
cd Tools/QA_auto_gen/
python auto_gen_QAv3.py --input_path your_document.md
```

### 4. 文档转化示例
```bash
# 参考 Tools/MinerU_xyb/readme.md
cd Tools/MinerU_xyb/
# 查看详细使用说明
```

### 5. LLM基准测试
```bash
# 查看LLM置信度评估机制
cd LLM_benchmarking/
# 参考 llm_benchmarking_cn.md 了解实现方法
```

## 应用案例

### 医疗领域应用
- **胰腺癌临床药物咨询**：基于最新临床试验数据的智能问答
- **药物副作用评估**：结合置信度评估的安全性分析
- **治疗方案推荐**：多模态信息融合的个性化建议

### 技术文档问答
- **API文档智能检索**：基于语义理解的精准匹配
- **代码示例生成**：结合上下文的代码片段推荐
- **故障诊断助手**：基于历史案例的问题解决方案

## 贡献者

### 胡卿老师
- **贡献内容**: Agent中LLM置信度评估与RAG触发机制研究
- **文档**: L.md
- **主要成果**: 
  - 提出了基于置信度评估的RAG触发机制
  - 分析了主流大模型对置信度评估的支持情况
  - 提供了OpenAI GPT、Claude、LLaMA等模型的实现方案
  - 医疗和奢侈品行业的实际应用案例

### IamCartsman
- **贡献内容**: 小胰宝基准测试集v1.0创建
- **目录**: xiaoyibao_Benching_V1.0/
- **主要成果**:
  - 构建了胰腺癌医疗领域的专业评测数据集
  - 完成了RAG能力测评的重要基础工作
  - 为后续项目发展奠定了数据基础

### 小X宝团队
- **贡献内容**: MinerU工具RAG转化实践
- **目录**: Tools/MinerU_xyb/
- **主要成果**:
  - 基于MinerU的文档转化工具优化
  - 图文混排RAG解决方案
  - 批量处理和云服务集成
  - 开源项目：[MinerU-xyb](https://github.com/PancrePal-xiaoyibao/MinerU-xyb)

### 技术团队
- **贡献内容**: QA自动生成工具和LLM基准测试
- **目录**: Tools/QA_auto_gen/, LLM_benchmarking/
- **主要成果**:
  - 基于智谱AI的QA对自动生成工具
  - 支持批量处理Markdown文件
  - JSON格式输出标准化
  - LLM性能基准测试框架

### InfiniFlow团队
- **贡献内容**: 医疗QA标准数据集贡献
- **目录**: QA_sets/InfiniFlow:medical_QA/
- **主要成果**:
  - 多专科医疗问答数据（内科、外科、肿瘤科、妇产科、儿科、男科等）
  - 标准化CSV格式数据集
  - Apache 2.0开源协议
  - 为医疗AI模型评估提供基础数据

## 发展规划

### 短期目标
- [ ] 完善LLM评价模块
- [ ] 优化文档转化工具的UI界面
- [ ] 扩展更多评估指标和方法
- [ ] 增加更多大模型支持

### 中期目标
- [ ] 构建完整的RAG评估基准测试集
- [ ] 开发可视化的RAG性能监控面板
- [ ] 集成更多领域的专业知识库
- [ ] 建立RAG系统的自动化优化流程

### 长期目标
- [ ] 打造业界领先的RAG技术标准
- [ ] 建设开放的RAG技术社区
- [ ] 推动RAG技术在更多行业的应用
- [ ] 贡献RAG相关的学术研究成果

## 参与贡献

我们欢迎各种形式的贡献，包括但不限于：

- **代码贡献**：新功能开发、Bug修复、性能优化
- **文档完善**：技术文档、使用教程、最佳实践
- **测试用例**：单元测试、集成测试、性能测试
- **应用案例**：实际项目应用、效果评估、经验分享

### 贡献流程
1. Fork 项目到个人仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 联系我们

- **项目维护**：RAG技术团队
- **技术交流**：[点击加入志愿者](https://uei55ql5ok.feishu.cn/wiki/LDFCw3sPPiOZ3EkwFVTcxtksnNx)
- **问题反馈**：GitHub Issues
- **商务合作**：请通过志愿者链接联系

## 致谢

感谢曾经的贡献者在小x宝社区推动RAG benchmarking能力的多次迭代，为开源社区类似垂直应用的benchmarking实践，走出了一条独一无二的实践之路

- 特别感谢发起人IamCartsman对项目的贡献，走完了RAG能力测评的重要一步，并进而衔接到本项目的推进：
- 特别感谢胡卿老师发起LLM benchmarking，为项目的发展做出了实质落地和模式总结。
- 感谢Sam推动和实践了RAG benchmarking，在最简陋的阶段，仍然铭记RAG质量和核心能力必须掌握自主的原则，希望社区接力力量保持，为RAG能力的测评和提升贡献力量。

感谢以下开源项目和团队的支持：

- [OpenDataLab MinerU](https://github.com/opendatalab/MinerU) - 文档解析核心技术
- [RAGAS](https://github.com/explodinggradients/ragas) - RAG评估框架
- [LangChain](https://github.com/langchain-ai/langchain) - LLM应用开发框架
- 小X宝社区 - 提供实践平台和用户反馈

## 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

---

**让我们一起推动RAG技术的发展，构建更智能的知识问答系统！** 🚀