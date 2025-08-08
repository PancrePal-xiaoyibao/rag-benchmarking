---
AIGC:
  Label: '1'
  ContentProducer: '001191310104MACE0YX63918020'
  ProduceID: '142569260085903360'
  ReservedCode1: '{"SecurityData":{"Type":"TC260PG","Version":1,"PubSD":[{"Type":"DS","AlgID":"1.2.156.10197.1.501","TBSData":{"Type":"LabelMataData"},"Signature":"3046022100AE6B84B3342128AE301A597596FCD9EECF3EA0FC2DB6C250BAF234E167298E9002210097223ABB647B786A90D0A1B93CB9382EA879563B2C7F025FA006BE16AE926C14"},{"Type":"PubKey","AlgID":"1.2.156.10197.1.501","KeyValue":"3059301306072A8648CE3D020106082A811CCF5501822D034200045728F08C58D9C72150A705693EF7337859F8FF216E802630041AFAE5C036F0A67CB1FDF8C57F5F8B623284AE09E4E545464B7140B26BBDA51BAE965F854C57AA"}]}}'
  ContentPropagator: '001191310104MACE0YX63918020'
  PropagateID: '142569260085903360'
  ReservedCode2: '{"SecurityData":{"Type":"TC260PG","Version":1,"PubSD":[{"Type":"DS","AlgID":"1.2.156.10197.1.501","TBSData":{"Type":"LabelMataData"},"Signature":"3046022100CFB0BFFC83C97AA5B09493295A7A054C082BBBB0E007AC139ACAA3BCD23D62DF022100FFDF70599E0E5D7DC6727825C5717DE0A9726D9F3119611E5E303FADA8792A5C"},{"Type":"PubKey","AlgID":"1.2.156.10197.1.501","KeyValue":"3059301306072A8648CE3D020106082A811CCF5501822D034200045728F08C58D9C72150A705693EF7337859F8FF216E802630041AFAE5C036F0A67CB1FDF8C57F5F8B623284AE09E4E545464B7140B26BBDA51BAE965F854C57AA"}]}}'
---

# 开源RAG基准测试项目深度调研报告

>  本报告由 阶跃AI 生成 · 2025/08/08 10:31:25


## 总体概述

当前开源RAG评测领域呈现出三大主要流派：以RAGAS为代表的组件化评估框架（专注于提供多维度评估指标）、以RAGBench为代表的大规模标注数据集（提供统一评估标准），以及以MIRAGE和CRUD-RAG为代表的领域/语言特化基准（分别针对医疗和中文场景）。这些项目共同推动了RAG系统的标准化评估，但在中文医疗领域的专业评测仍存在明显空白。

## 识别的核心项目

### RAGAS

- **一句话总结**: RAGAS是一个专为RAG系统设计的自动化评估框架，提供了一套全面的无参考评估指标，并与主流RAG开发框架（如LangChain和LlamaIndex）无缝集成。

- **GitHub链接**: https://github.com/explodinggradients/ragas

- **核心特性**: 
  1. 无参考评估：大多数评估指标不需要人工标注的参考答案，通过LLM-as-judge方式进行评估[<sup>[13]</sup>](https://hub.baai.ac.cn/view/31713)
  2. 组件化评估：分别评估RAG系统的检索和生成两个核心环节[<sup>[12]</sup>](https://www.modb.pro/db/1824260472640647168)
  3. 框架集成：与LlamaIndex和LangChain等主流RAG开发框架深度集成，便于开发者在工作流中直接使用[<sup>[12]</sup>](https://www.modb.pro/db/1824260472640647168)
  4. 可扩展性：支持自定义评估指标和评估流程[<sup>[11]</sup>](https://zhuanlan.zhihu.com/p/691820367)
  5. 丰富的评估维度：涵盖RAG、智能代理和多模态等多种应用场景[<sup>[10]</sup>](https://developer.aliyun.com/article/1490291)

- **评估指标**: 
  - **检索评估指标**:
    - Context Precision：评估检索内容中与问题相关信息的精确度[<sup>[9]</sup>](https://aclanthology.org/2024.eacl-demo.16/)
    - Context Recall：衡量检索内容是否包含回答问题所需的所有信息[<sup>[10]</sup>](https://developer.aliyun.com/article/1490291)
    - Context Entities Recall：特别关注检索内容中实体的召回率[<sup>[10]</sup>](https://developer.aliyun.com/article/1490291)
    - Noise Sensitivity：测量系统对检索内容中噪声的敏感程度[<sup>[10]</sup>](https://developer.aliyun.com/article/1490291)
    - Context Relevancy：评估检索的上下文与问题的相关性[<sup>[10]</sup>](https://developer.aliyun.com/article/1490291)
  
  - **生成评估指标**:
    - Faithfulness：衡量生成的回答是否忠实于检索的上下文[<sup>[8]</sup>](https://agijuejin.feishu.cn/wiki/ZqXTwt1e4id9vpklSMacqzf9npd)
    - Answer Relevancy：评估生成的回答与问题的相关性[<sup>[7]</sup>](https://cloud.tencent.com/developer/article/2467491?from=15425&frompage=seopage&policyId=20240001&traceId=01jqb437g6n2xsr4nx8ffnk772)
    - Factual Correctness：评估回答的事实准确性[<sup>[10]</sup>](https://developer.aliyun.com/article/1490291)
    
  - **多模态指标**:
    - Multimodal Faithfulness：扩展到多模态内容的忠实度评估[<sup>[10]</sup>](https://developer.aliyun.com/article/1490291)
    - Multimodal Relevance：评估多模态内容的相关性[<sup>[10]</sup>](https://developer.aliyun.com/article/1490291)

- **中文/领域支持**: 
  - **中文支持**：RAGAS的评估依赖于底层LLM的能力，如使用支持中文的模型（如GPT-4、Claude等），理论上可以支持中文评估，但文档中未明确提及对中文的专门优化。
  - **领域支持**：提供了针对SQL和摘要等特定任务的专门指标[<sup>[10]</sup>](https://developer.aliyun.com/article/1490291)，但未见对医疗等特定领域的专门支持。框架设计允许用户根据需求修改现有指标或创建新指标[<sup>[10]</sup>](https://developer.aliyun.com/article/1490291)，为领域适配提供了可能性。

- **简要评述**: 
  RAGAS作为一个成熟的RAG评估框架，其组件化的评估方法和丰富的指标体系对我们的中文医疗RAG-bench项目有很高的参考价值。特别是其无参考评估的设计理念可以降低人工标注的成本。然而，RAGAS缺乏对中文和医疗领域的专门优化，我们需要基于RAGAS的框架，添加中文医疗特定的评估指标（如医疗术语准确性、诊断相关性等），并优化评估提示以更好地处理中文医疗文本。此外，RAGAS的LLM-as-judge方法在医疗领域可能需要使用专业医疗大模型来提高评估的专业性和准确性。

### RAGBench

- **一句话总结**: RAGBench是首个大规模、跨行业的RAG基准数据集，包含10万个样本，并提出了TRACe评估框架，用于全面评估RAG系统的检索器和生成器组件。

- **GitHub链接**: https://huggingface.co/datasets/rungalileo/ragbench

- **核心特性**: 
  1. 大规模数据集：包含10万个样本，涵盖五个特定行业领域[<sup>[6]</sup>](https://arxiv.org/html/2407.11005v2)
  2. 多领域覆盖：包括生物医学研究、通用知识、法律合同、客户支持和金融领域[<sup>[6]</sup>](https://arxiv.org/html/2407.11005v2)
  3. TRACe评估框架：提出了一套统一的、可解释的RAG评估指标[<sup>[6]</sup>](https://arxiv.org/html/2407.11005v2)
  4. 真实数据源：样本来源于行业语料库，如用户手册，更贴近实际应用场景[<sup>[6]</sup>](https://arxiv.org/html/2407.11005v2)
  5. 可解释性：提供详细的标注，便于分析RAG系统的具体问题[<sup>[6]</sup>](https://arxiv.org/html/2407.11005v2)

- **评估指标**: 
  - **TRACe评估框架**:
    - Utilization (利用率)：评估生成器对检索内容的使用程度
    - Relevance (相关性)：评估检索内容与查询的相关性
    - Adherence (遵从性)：评估生成的回答是否忠实于提供的上下文(与文献中的"faithfulness"、"groundedness"和"attribution"概念相似)
    - Completeness (完整性)：评估生成的回答是否完整

- **中文/领域支持**: 
  - **中文支持**：RAGBench主要基于英文数据集构建，包括CovidQA、PubmedQA、HotpotQA、MS Marco等，没有明确提到对中文的支持。
  - **领域支持**：RAGBench确实包含医疗领域的数据集，特别是生物医学研究领域的PubmedQA和CovidQA。PubmedQA包含来自研究摘要的文档，CovidQA-RAG包含来自研究论文的文档。这表明RAGBench支持医疗领域的评估，但主要集中在生物医学研究文献方面，而不是更广泛的临床医疗应用。

- **简要评述**: 
  RAGBench的TRACe评估框架提供了一套全面且可解释的评估指标，对我们构建中文医疗RAG-bench有很好的借鉴意义。特别是其在生物医学领域的数据集（如PubmedQA和CovidQA）可以作为医疗领域评估的参考。然而，RAGBench主要基于英文数据，缺乏对中文的支持，且其医疗数据主要集中在研究文献而非临床应用。因此，我们需要在保留TRACe框架优点的基础上，构建中文医疗领域的专用数据集，包括临床诊断、治疗方案、药物信息等多方面内容，并考虑中文医疗文本的特点（如专业术语翻译、中医概念等）进行评估指标的调整。

### MIRAGE

- **一句话总结**: MIRAGE是首个专为医疗问答领域的RAG系统设计的基准测试，包含来自五个常用医疗问答数据集的7,663个问题，并提供MedRAG工具包进行全面评估。

- **GitHub链接**: https://github.com/Teddy-XiongGZ/MIRAGE

- **核心特性**: 
  1. 医疗专用：专为医疗领域RAG系统评估设计，包含医学考试和生物医学研究问题[<sup>[5]</sup>](https://zhuanlan.zhihu.com/p/691686155)
  2. 多样化数据集：整合五个常用医疗问答数据集，总计7,663个问题[<sup>[5]</sup>](https://zhuanlan.zhihu.com/p/691686155)
  3. 四种评估设置：零样本学习(ZSL)、多选题评估(MCE)、检索增强生成(RAG)和仅问题检索(QOR)[<sup>[5]</sup>](https://zhuanlan.zhihu.com/p/691686155)
  4. MedRAG工具包：提供专门的工具评估不同组件组合的性能[<sup>[5]</sup>](https://zhuanlan.zhihu.com/p/691686155)
  5. 最佳实践指南：基于大规模实验结果提供医学RAG系统实施的建议[<sup>[5]</sup>](https://zhuanlan.zhihu.com/p/691686155)

- **评估指标**: 
  - **准确率**：评估不同LLM和RAG系统在各数据集上的答案准确性
  - **检索质量**：评估不同语料库和检索器组合的检索效果
  - **规模效应**：研究检索片段数量与性能之间的关系(发现对数线性缩放特性)
  - **综合性能**：跨数据集的平均性能作为系统总体评价

- **中文/领域支持**: 
  - **中文支持**：MIRAGE主要基于英文医疗数据集构建，包括MMLU-Med、MedQA-US、MedMCQA、PubMedQA和BioASQ-Y/N，未明确提及对中文的支持。
  - **领域支持**：MIRAGE专为医疗领域设计，涵盖医学考试问答和生物医学研究问答两大类[<sup>[5]</sup>](https://zhuanlan.zhihu.com/p/691686155)，包含解剖学、临床知识、专业医学等领域的问题，非常适合医疗领域的RAG系统评估。

- **简要评述**: 
  MIRAGE作为专为医疗RAG系统设计的基准测试，其医疗领域的专业性与我们的中文医疗RAG-bench项目高度相关。特别是其MedRAG工具包和对不同医疗语料库、检索器组合的系统评估方法，为我们构建医疗RAG评测框架提供了宝贵参考。然而，MIRAGE主要基于英文医疗数据集，缺乏对中文医疗文本的支持。因此，我们可以借鉴MIRAGE的评估框架和方法，但需要构建中文医疗数据集，包括中文医学教材、临床指南、病例报告等资源，并考虑中西医结合的特点。此外，我们还可以参考MIRAGE的四种评估设置，特别是"仅问题检索"设置，它更符合真实医疗问答场景。

### CRUD-RAG

- **一句话总结**: CRUD-RAG是首个全面的中文RAG基准测试框架，基于CRUD(创建、读取、更新、删除)操作分类，涵盖四种主要应用场景，包含36,166个样本。

- **GitHub链接**: 未找到公开的GitHub仓库链接，但论文已在arXiv发布：https://arxiv.org/abs/2401.17043

- **核心特性**: 
  1. 中文专用：专为中文RAG系统设计的基准测试，基于中国主要新闻网站的高质量数据[<sup>[4]</sup>](https://www.researchgate.net/publication/385078877_CRUD-RAG_A_Comprehensive_Chinese_Benchmark_for_Retrieval-Augmented_Generation_of_Large_Language_Models)
  2. CRUD分类法：将RAG应用场景分为Create(创建)、Read(读取)、Update(更新)和Delete(删除)四类[<sup>[3]</sup>](https://arxiv.org/html/2401.17043v2)
  3. 多场景评估：包括文本续写、问答(单文档和多文档)、幻觉修改和开放域多文档摘要四个评估任务
  4. 大规模数据集：包含36,166个样本，规模远超其他类似基准
  5. 系统性评估：全面评估RAG系统的各个组件，包括检索模型、外部知识库构建和语言模型

- **评估指标**: 
  - **ROUGE**：用于评估生成文本与参考文本的重叠程度
  - **BLEU**：评估生成文本的质量
  - **BertScore**：基于BERT的语义相似度评分
  - **RAGQuestEval**：专门为RAG系统设计的评估指标
  - **检索和生成一致性评估**：评估检索内容与生成内容的一致性

- **中文/领域支持**: 
  - **中文支持**：CRUD-RAG是专门为中文RAG系统设计的基准测试[<sup>[4]</sup>](https://www.researchgate.net/publication/385078877_CRUD-RAG_A_Comprehensive_Chinese_Benchmark_for_Retrieval-Augmented_Generation_of_Large_Language_Models)，通过爬取中国主要新闻网站的最新高质量新闻数据构建，完全支持中文评估。
  - **领域支持**：CRUD-RAG主要基于新闻数据构建，未见对医疗等特定领域的专门支持。但其CRUD分类法和评估方法可以应用于不同领域。

- **简要评述**: 
  CRUD-RAG作为首个全面的中文RAG基准测试框架，其对中文的专门支持与我们的中文医疗RAG-bench项目高度相关。特别是其CRUD分类法将RAG应用场景分为四类的方法，为我们构建多场景的医疗RAG评测提供了框架参考。然而，CRUD-RAG主要基于新闻数据，缺乏对医疗等特定领域的专门支持。因此，我们可以借鉴CRUD-RAG的中文评估方法和CRUD分类框架，但需要构建专门的中文医疗数据集，并根据医疗场景的特点调整评估任务和指标。例如，在医疗RAG的"Create"场景中，可以评估系统生成医疗报告或治疗计划的能力；在"Read"场景中，可以评估系统回答医学问题的准确性；在"Update"场景中，可以评估系统纠正医疗信息错误的能力；在"Delete"场景中，可以评估系统简化复杂医学文献的能力。

### LlamaIndex评估套件

- **一句话总结**: LlamaIndex提供了一套集成在RAG开发框架中的评估工具，包括响应评估和检索评估两大类，支持多种评估指标和与社区评估工具的集成。

- **GitHub链接**: https://github.com/run-llama/llama_index

- **核心特性**: 
  1. 开发框架集成：评估功能直接集成在RAG开发框架中，便于在开发过程中进行持续评估
  2. 双重评估焦点：分别关注响应评估和检索评估两个核心方面
  3. LLM评估器：使用基于LLM的评估模块(如GPT-4)来衡量生成结果的质量[<sup>[2]</sup>](https://docs.llamaindex.ai/en/stable/examples/cookbooks/oreilly_course_cookbooks/Module-3/Evaluating_RAG_Systems/)
  4. 社区工具集成：集成了多个社区评估工具，包括Ragas、RAGChecker、Cleanlab等
  5. 问题生成功能：支持从数据中自动生成问题用于评估[<sup>[1]</sup>](https://docs.llamaindex.ai/en/v0.10.22/examples/data_connectors/GithubRepositoryReaderDemo/)

- **评估指标**: 
  - **响应评估指标**:
    - 正确性(Correctness)：评估生成的答案是否与参考答案匹配(需要标签)
    - 语义相似性(Semantic Similarity)：评估预测答案与参考答案的语义相似度(需要标签)
    - 忠实度(Faithfulness)：评估答案是否忠于检索的上下文，即是否存在幻觉
    - 上下文相关性(Context Relevancy)：评估检索的上下文是否与查询相关
    - 回答相关性(Answer Relevancy)：评估生成的答案是否与查询相关
    - 指南遵守性(Guideline Adherence)：评估预测答案是否遵循特定指南
  
  - **检索评估指标**:
    - 平均倒数排名(MRR)：评估检索结果的排名质量
    - 命中率：评估检索是否返回相关文档
    - 精确度：评估检索结果的准确性

- **中文/领域支持**: 
  - **中文支持**：LlamaIndex的评估功能依赖于底层LLM的能力，如使用支持中文的模型，理论上可以支持中文评估，但文档中未明确提及对中文的专门优化。
  - **领域支持**：LlamaIndex提供了灵活的评估框架，允许用户自定义评估指标和方法，为领域适配提供了可能性，但未见对医疗等特定领域的专门支持。

- **简要评述**: 
  LlamaIndex评估套件作为集成在RAG开发框架中的评估工具，其无缝集成的特点对我们构建易用的中文医疗RAG-bench有参考价值。特别是其响应评估和检索评估的双重焦点，以及与社区评估工具的集成能力，可以帮助我们构建全面的评估框架。然而，LlamaIndex的评估功能缺乏对中文和医疗领域的专门优化。因此，我们需要在借鉴LlamaIndex评估框架的基础上，添加中文医疗特定的评估指标和方法，并考虑与现有中文医疗大模型的集成。此外，LlamaIndex的问题生成功能也值得借鉴，我们可以基于中文医疗文献自动生成评估问题，减少人工标注的成本。

## 综合分析与建议

基于对上述五个开源RAG评测/基准测试框架的深入调研，我们可以得出以下综合分析和建议，以指导中文医疗领域RAG基准测试框架的构建：

1. **评估指标体系**：
   - 借鉴RAGAS的组件化评估方法，分别评估检索和生成两个环节
   - 采用RAGBench的TRACe框架中的四个核心指标：利用率、相关性、遵从性和完整性
   - 结合MIRAGE的医疗专用评估指标，如医疗准确性、临床相关性等
   - 参考CRUD-RAG的多场景评估方法，覆盖医疗RAG的不同应用场景

2. **中文支持**：
   - 借鉴CRUD-RAG的中文评估方法和技术
   - 构建中文医疗专用数据集，包括医学教材、临床指南、病例报告等
   - 优化评估提示以更好地处理中文医疗文本
   - 考虑中西医结合的特点，包括中医术语和概念的评估

3. **医疗领域适配**：
   - 参考MIRAGE的医疗RAG评估框架和方法
   - 扩展评估范围，覆盖临床诊断、治疗方案、药物信息等多方面内容
   - 引入医疗专业性评估指标，如医疗术语准确性、诊断相关性等
   - 考虑使用专业医疗大模型作为评估器，提高评估的专业性和准确性

4. **系统架构**：
   - 采用模块化设计，便于扩展和定制
   - 集成多种评估方法，包括LLM-as-judge和基于标注数据集的评估
   - 提供API和命令行接口，便于集成到开发流程中
   - 支持批量评估和结果可视化

5. **社区生态**：
   - 开源代码和数据集，促进社区参与和贡献
   - 提供详细的文档和教程，降低使用门槛
   - 建立评估基准和排行榜，促进技术竞争和进步
   - 与现有RAG框架（如LangChain、LlamaIndex）集成，提高可用性

通过综合借鉴这些优秀开源项目的经验和技术，我们可以构建一个专为中文医疗领域设计的、全面且实用的RAG基准测试框架，填补当前评测领域的空白，推动中文医疗RAG技术的发展和应用。
## 关键引用:
1. [Github Repo Reader - LlamaIndex](https://docs.llamaindex.ai/en/v0.10.22/examples/data_connectors/GithubRepositoryReaderDemo/)
2. [Evaluating RAG Systems - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/cookbooks/oreilly_course_cookbooks/Module-3/Evaluating_RAG_Systems/)
3. [CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval ...](https://arxiv.org/html/2401.17043v2)
4. [CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval ...](https://www.researchgate.net/publication/385078877_CRUD-RAG_A_Comprehensive_Chinese_Benchmark_for_Retrieval-Augmented_Generation_of_Large_Language_Models)
5. [arXiv 2402 | Benchmarking Retrieval-Augmented Generation for Medicine 论文分享](https://zhuanlan.zhihu.com/p/691686155)
6. [Explainable Benchmark for Retrieval-Augmented Generation Systems](https://arxiv.org/html/2407.11005v2)
7. [使用自定义LLM：RAGAs评估](https://cloud.tencent.com/developer/article/2467491?from=15425&frompage=seopage&policyId=20240001&traceId=01jqb437g6n2xsr4nx8ffnk772)
8. [RAGAS 框架：快速自动评估 RAG 质量，还便于集成](https://agijuejin.feishu.cn/wiki/ZqXTwt1e4id9vpklSMacqzf9npd)
9. [RAGAs: Automated Evaluation of Retrieval Augmented Generation](https://aclanthology.org/2024.eacl-demo.16/)
10. [【AI大模型应用开发】【RAG评估】0. 综述：一文了解RAG评估方法、工具与指标](https://developer.aliyun.com/article/1490291)
11. [RAGAS metrics：评估指标分析 - 知乎专栏](https://zhuanlan.zhihu.com/p/691820367)
12. [RAGAS 框架：快速自动评估 RAG 质量，还便于集成](https://www.modb.pro/db/1824260472640647168)
13. [再看大模型RAG检索增强如何评估：RAGAS开源自动化评估框架](https://hub.baai.ac.cn/view/31713)