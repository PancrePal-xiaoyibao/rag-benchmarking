import re
import os

def extract_urls(input_file, output_file):
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式提取URL
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
    
    # 将URL写入Markdown文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for url in urls:
            f.write(f'{url}\n')
    
    print(f'已提取 {len(urls)} 个URL并保存到 {output_file}')

# 使用示例
input_file = '输入文件路径.txt'  # 替换为您的输入文件路径
output_file = '输出文件.md'  # 替换为您想要的输出文件名

extract_urls(input_file, output_file)
