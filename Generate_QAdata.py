#coding:utf-8
import sys
import os
import random
import json
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="5178e6350c7242e995c5f20d54e64e9f.pntyniBzt8tYrxMQ")

def return_random_prompt(content):
    input_length = random.randint(80, 120)
    output_length = random.randint(512, 1024)
    
    system_prompt = '''根据下面提供有关煤矿安全领域文本，请你仔细通读全文，你需要依据该文本：######{}######尽可能给出多样化的问题和对应的回答。我们将用于微调deepseek_1.5b模型对问答对数据的完成情况。要求:
    1. 生成问题有价值且遵守该文本信息，回答准确专业。
    2. 生成问答对10-20条，每个问答对不能重复。
    3. 问题多样化，同个问题可以换成不同表述方式，但意思保持不变。
    4. 为问题生成作为<input>，不应该只包含简单的占位符。<input>应提供实质性的内容问题，具有挑战性。字数不超过{}字。
    5. <output>应该是对问题的适当且真实的回答，不能只回复答应或拒绝请求。如果需要额外信息才能回复时，请努力预测用户意图并尝试回复，但不能胡编乱造。<output>的内容应少于{}字。
    6. 严格按照JSON格式输出，不要添加任何额外的文字说明、前缀或后缀。
    7. 输出必须是有效的JSON数组格式
    8. JSON中的字符串必须正确转义，特别是引号和换行符。
    9. 不要使用markdown代码块包装JSON。
    10. JSON模板格式数据（严格按此格式输出）:
    [
        {{
        "input": "请提供新建矿井立井井筒冻结法施工的具体要求。",
        "output": "新建矿井立井井筒冻结法施工需要遵守以下要求：冻结深度必须穿过风化带延深至稳定的基岩10m以上，第一个冻结孔必须全孔取芯，钻孔时必须测定钻孔的方向和偏斜度，偏斜度超过规定时必须及时纠正，冻结管必须采用无缝钢管并焊接或螺纹连接，开始冻结后必须经常观察水文观测孔的水位变化，并在确定冻结壁已交圈后才能进行试挖。"
        }},
        ...
    ]
    '''.format(content, input_length, output_length)
    return system_prompt

def process_file(file_path):
    """处理单个文件，生成问答数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 如果文件内容太短，跳过
        if len(content.strip()) < 100:
            print(f"跳过文件 {file_path}：内容太短")
            return None
            
        print(f"正在处理文件: {os.path.basename(file_path)}")
        
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {
                    "role": "user",
                    "content": return_random_prompt(content)
                }
            ],
            top_p=0.7,
            temperature=0.9,
            stream=False,
            max_tokens=20000,
        )
        
        response_content = response.choices[0].message.content
        
        # 尝试解析JSON响应
        try:
            qa_data = json.loads(response_content)
            # 为每个问答对添加来源文件信息
            # for item in qa_data:
            #     item['source_file'] = os.path.basename(file_path)
            return qa_data
        except json.JSONDecodeError:
            print(f"文件 {file_path} 的响应不是有效的JSON格式，尝试清理后重新解析")
            
            # 尝试清理响应内容
            cleaned_content = response_content.strip()
            
            # 移除可能的markdown代码块标记
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.startswith('```'):
                cleaned_content = cleaned_content[3:]
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]
            
            # 查找JSON数组的开始和结束
            start_idx = cleaned_content.find('[')
            end_idx = cleaned_content.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_part = cleaned_content[start_idx:end_idx+1]
                try:
                    qa_data = json.loads(json_part)
                    return qa_data
                except json.JSONDecodeError:
                    pass
            
            print(f"清理后仍无法解析JSON，原始响应内容：")
            print(response_content)
            return None
            
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None

if __name__ == "__main__":
    data_dir = './data/'
    output_file_path = "QAdata_all.json"
    
    # 存储所有问答数据
    all_qa_data = []
    
    # 获取data目录下所有txt文件
    txt_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.txt') and not file.startswith('.'):
            txt_files.append(os.path.join(data_dir, file))
    
    print(f"找到 {len(txt_files)} 个txt文件")
    
    # 处理每个文件
    for i, file_path in enumerate(txt_files, 1):
        print(f"\n进度: {i}/{len(txt_files)}")
        qa_data = process_file(file_path)
        
        if qa_data:
            all_qa_data.extend(qa_data)
            print(f"成功生成 {len(qa_data)} 条问答数据")
        else:
            print("处理失败")
    
    # 保存所有问答数据到JSON文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_qa_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成！总共生成 {len(all_qa_data)} 条问答数据")
    print(f"结果已保存到: {output_file_path}")
