## 背景

在春节小X宝社区产品开发团队贡献了第一个开源项目Fastgpt-on-wechat之后，昨天我们也开源了第二个适合RAG文档转化，基于上海Opendata的MinerU开源项目改造的独立项目。

在我们的RAG实践中，优化RAG生成质量是长期迭代的任务，一方面我们完善了RAG的测评环节，引入了自评估体系和RAGAS测评，对于目前的偏差校准开始实践和优化，另外一方面也开始对

## 项目介绍

开源项目地址：
<https://github.com/PancrePal-xiaoyibao/MinerU-xyb>



---


**感谢上游项目 Opendatalab-MinerU团队的开源和不断更新**
<https://github.com/opendatalab/MinerU> 

<div style="display: flex;">
  <img src="https://webpub.shlab.tech/dps/opendatalab-web/xlab_v5.1922/assets/logo-odl-b2631aad.svg" alt="ODL Logo" style="max-width: 200px; height: auto; margin-right: 10px;">
  <img src="https://webpub.shlab.tech/dps/opendatalab-web/mineru-prod.1907/assets/minerU-CF_69mGL.svg" alt="MinerU Logo" style="max-width: 50%; height: auto;">
</div>






---
** 针对问题 **
1. 原生RAG工具的知识库导入中，原始pdf和其它文档损耗大，文件中的图片，公式比较难
2. API能力

**主要特点**：
1.  批量处理文档，支持多种文件类型；
2. 结合Sealos，Minio，加入S3配置，解析后图链直接长期有效，被RAG引用，实现图文混排
2. 适配GPU云服务资源，结合腾讯IDE的免费资源，低成本构建





**后续开发**
1. 结合RAG平台工具的API能力增强，实现批量转化后的自动导入
2. UI能力：方便使用者（现在MinerU已经提供了UI，项目中有，但在测试中）
**图片

![](https://fastly.jsdelivr.net/gh/bucketio/img19@main/2025/02/26/1740530116843-f5fba9b1-a509-40ce-ba5b-7e1532e8724c.png)



截图1  代码示例

![](https://fastly.jsdelivr.net/gh/bucketio/img2@main/2025/02/26/1740530549234-1814b841-5faa-45f1-816e-a38acdc6a4f8.png)




截图2  图文混排的效果


![](https://fastly.jsdelivr.net/gh/bucketio/img12@main/2025/02/26/1740530645672-6624c35a-74ff-4b1c-adbc-2e127cd47e74.png)



## 加入我们，实践最新RAG技术

RAG团队和开发团队还在持续努力，在小X宝社区的平台上，不仅提供了最新的RAG技术的落地土壤和研讨/实践环境，也结合病友和患者/家属的抗癌努力中的实际需求，带动技术向善，在AI coding的高效率助力之下，让我们重新找回主导权，驾驭AI工具，不断创新，输出有价值的应用。

我们的RAG团队/技术和产品团队也邀请有热情，有公益精神的你加入！欢迎联络

## [点击加入志愿者](https://uei55ql5ok.feishu.cn/wiki/LDFCw3sPPiOZ3EkwFVTcxtksnNx)


