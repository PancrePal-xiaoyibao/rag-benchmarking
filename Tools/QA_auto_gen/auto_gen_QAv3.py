import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from zhipuai import ZhipuAI

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_generation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QAGenerator:
    def __init__(self, api_key: str, model_name: str = "glm-4"):
        self.client = ZhipuAI(api_key=api_key)
        self.model = model_name
        self.qa_data = []  # 直接存储问答对列表
        logger.info(f"初始化GLM模型: {model_name}")

    def generate_response(self, prompt: str) -> str:
        """调用API生成响应"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API调用失败: {str(e)}")
            return ""

    def process_file(self, file_path: Path) -> None:
        """处理单个文件"""
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 生成摘要
            summary_prompt = f"""请分析以下文本，提供关键内容总结：
            {content[:3000]}...
            """
            summary = self.generate_response(summary_prompt)
            logger.info("已生成文档摘要")

            # 基于摘要生成问答对
            qa_prompt = f"""基于以下文章摘要：
            {summary}
            
            请生成10个问答对，要求：
            1. 问题要覆盖摘要中的所有关键点
            2. 问答要由简到难，循序渐进
            3. 确保问题和答案准确反映摘要内容
            
            请严格按照以下格式返回，不要包含任何其他内容：
            {{
                "results": [
                    {{
                        "query_id": "000",
                        "query": "第一个问题",
                        "gt_answer": "第一个答案"
                    }},
                    {{
                        "query_id": "001",
                        "query": "第二个问题",
                        "gt_answer": "第二个答案"
                    }}
                ]
            }}
            """
            
            qa_response = self.generate_response(qa_prompt)
            logger.info("已生成问答对")
            logger.debug(f"原始问答响应: {qa_response}")
            
            # 处理返回的JSON
            try:
                qa_json = json.loads(qa_response)
                self.qa_data.extend(qa_json["results"])
                logger.info(f"成功处理文件: {file_path.name}")
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败 {file_path.name}: {str(e)}")
                logger.error(f"问题的JSON内容: {qa_response}")
                
                # 尝试修复JSON格式
                try:
                    # 清理响应文本
                    qa_response = qa_response.strip()
                    # 提取JSON部分
                    import re
                    json_pattern = r'\{[\s\S]*\}'
                    match = re.search(json_pattern, qa_response)
                    if match:
                        qa_response = match.group()
                        qa_json = json.loads(qa_response)
                        self.qa_data.extend(qa_json["results"])
                        logger.info(f"JSON修复成功: {file_path.name}")
                except Exception as e:
                    logger.error(f"JSON修复失败: {str(e)}")
                
        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {str(e)}")

    def process_directory(self, source_dir: str = "./documents") -> None:
        """处理目录下的所有文件"""
        directory = Path(source_dir)
        if not directory.exists():
            logger.error(f"目录不存在: {source_dir}")
            return

        # 获取所有.md文件（可以根据需要修改文件类型）
        files = list(directory.glob("*.md"))
        total_files = len(files)
        logger.info(f"找到 {total_files} 个文件待处理")

        for index, file_path in enumerate(files, 1):
            logger.info(f"正在处理第 {index}/{total_files} 个文件: {file_path.name}")
            self.process_file(file_path)
            
            # 添加间隔时间
            if index < total_files:
                wait_time = 10  # 10秒间隔
                logger.info(f"等待 {wait_time} 秒后处理下一个文件...")
                time.sleep(wait_time)
            
            # 每处理3个文件保存一次临时结果
            if index % 3 == 0:
                self.save_qa_data("QA_temp.json")
                logger.info(f"已保存临时结果到 QA_temp.json")

    def save_qa_data(self, output_file: str = "QA.json") -> None:
        """保存QA数据到JSON文件"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "results": self.qa_data
                }, f, ensure_ascii=False, indent=2)
                
            logger.info(f"已保存QA数据到: {output_file}")
            logger.info(f"总共生成问答对数量: {len(self.qa_data)}")
        except Exception as e:
            logger.error(f"保存JSON文件失败: {str(e)}")

def main():
    # 配置
    config = {
        "api_key": "b6a20560a7472f1c2d477a2d296d5e5b.BdpeuTxciElVYuzi",  # 替换为你的实际API密钥
        "model_name": "glm-4",
        "source_dir": "./documents",  # 源文件目录
        "output_file": "QA.json"  # 输出文件
    }
    
    try:
        qa_generator = QAGenerator(
            api_key=config["api_key"],
            model_name=config["model_name"]
        )
        
        # 处理目录
        qa_generator.process_directory(config["source_dir"])
        
        # 保存最终结果
        qa_generator.save_qa_data(config["output_file"])
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()