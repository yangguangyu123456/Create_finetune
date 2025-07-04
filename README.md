# DeepSeek-R1-Distill-Qwen-1.5B 煤矿安全问答微调项目

## 项目简介

本项目基于 DeepSeek-R1-Distill-Qwen-1.5B 模型，针对煤矿安全领域进行专业问答微调。通过收集煤矿安全相关的法规、规程、技术标准等文档，构建高质量的问答数据集，使用 QLoRA 技术对模型进行高效微调，打造专业的煤矿安全知识问答助手。

## 项目结构
Create_finetune/

├── DeepSeek-R1-Distill-Qwen-1.5B/     # 基础模型文件

│   ├── config.json

│   ├── generation_config.json

│   ├── model.safetensors

│   ├── tokenizer.json

│   └── tokenizer_config.json

├── data/                              # 原始文档数据

│   ├── 煤矿安全规程相关文档.txt

│   ├── 煤矿建设安全规范.txt

│   ├── 防治煤与瓦斯突出细则.txt

│   └── ... (共40+个煤矿安全相关文档)

├── Generate_QAdata.py                 # 数据集构造脚本

├── deepseek_QA.py                     # 问答生成提示模板

├── QAdata_all.json                    # 生成的问答数据集

├── train.py                           # 微调训练脚本

├── inference.py                       # 推理测试脚本

└── output/                            # 训练输出目录

## 数据集构造

### 1. 原始数据收集

项目收集了40多个煤矿安全领域的权威文档，包括：
- 煤矿安全规程
- 煤矿建设安全规范
- 防治煤与瓦斯突出细则
- 煤矿防治水规定
- 矿山安全法及实施条例
- 各类技术规范和标准

### 2. 问答数据生成

使用 `Generate_QAdata.py` 脚本自动生成问答数据：

**核心功能：**
- 调用智谱AI GLM-4模型，基于原始文档生成多样化问答对
- 每个文档生成10-20条高质量问答数据
- 问题长度控制在80-120字，答案长度控制在512-1024字
- 严格的JSON格式输出和错误处理

**生成策略：**

### 问答生成提示模板
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

## 微调训练方法

### 1. 技术方案
使用 QLoRA (Quantized Low-Rank Adaptation) 技术进行高效微调：
- 4-bit量化 ：显著降低显存占用
- LoRA适配器 ：只训练少量参数，保持原模型完整性
- 梯度累积 ：支持大批次训练
### 2. 训练配置
train.py 中的关键配置：

模型配置

class ModelArguments:

    model_name_or_path: str = "./DeepSeek-R1-Distill-Qwen-1.5B"
    
    lora_rank: int = 64              # LoRA秩
    
    lora_alpha: int = 16             # LoRA缩放参数
    
    lora_dropout: float = 0.1        # LoRA dropout
    
    use_4bit: bool = True            # 4-bit量化
    
    bnb_4bit_compute_dtype: str = "bfloat16"
    
    bnb_4bit_quant_type: str = "nf4"

训练配置
class TrainingArguments_Custom:

    num_train_epochs: int = 6        # 训练轮数
    
    per_device_train_batch_size: int = 4
    
    gradient_accumulation_steps: int = 4
    
    learning_rate: float = 2e-4      # 学习率
    
    weight_decay: float = 0.01
    
    warmup_ratio: float = 0.1
    
    lr_scheduler_type: str = "cosine"

### 3. 数据处理
对话格式化：
def format_conversation(self, input_text: str, output_text: str) -> str:

    return f"system: 你是一个煤矿安全领域的知识达人，你对相关煤矿安全规章规程制度、技术等文档非常熟悉。请你专业正确地解答用户想问的煤矿安全相关问题。\nuser: {input_text}\nresponse: {output_text}"

LoRA目标模块：

target_modules=[

    "q_proj", "k_proj", "v_proj", "o_proj",
    
    "gate_proj", "up_proj", "down_proj"
    
]

### 4. 训练流程
1. 模型加载 ：加载基础模型并配置4-bit量化
2. LoRA配置 ：设置LoRA适配器参数
3. 数据准备 ：加载问答数据并进行tokenization
4. 训练执行 ：使用Trainer进行微调训练
5. 模型保存 ：保存LoRA适配器和训练配置
### 5. 训练启动
python train.py

训练完成后，模型将保存在 ./output/deepseek-coal-safety-{timestamp}/ 目录中
### 6. 推理使用
#### 1. 推理脚本
inference.py 提供多种推理模式：

交互式对话：

python inference.py --peft_model ./output/deepseek-coal-safety-xxx

单个问题：

python inference.py --peft_model ./output/deepseek-coal-safety-xxx --question "煤矿瓦斯防治的基本要求是什么？"

批量问题：

python inference.py --peft_model ./output/deepseek-coal-safety-xxx --questions_file questions.json --output_file results.json

#### 2. 推理特性
- 4-bit量化推理 ：降低显存占用
- LoRA模型加载 ：快速加载微调适配器
- 多种输入格式 ：支持交互式、单问题、批量推理
- 可配置参数 ：温度、top_p、最大长度等
## 环境要求
torch>=2.0.0

transformers>=4.35.0

peft>=0.6.0

datasets>=2.14.0

bitsandbytes>=0.41.0

accelerate>=0.24.0

zhipuai  # 用于数据生成

## 使用说明
### 1. 数据集构造
python Generate_QAdata.py
### 2. 微调训练
python train.py
### 3. 推理测试
交互式问答

python inference.py --peft_model ./output/your_model_path

单个问题测试

python inference.py --peft_model ./output/your_model_path --question "你的问题"

批量问题测试

python inference.py --peft_model./output/your_model_path --questions_file questions.json --output_file results.json

## 项目特点
1. 专业领域定制 ：专门针对煤矿安全领域进行微调
2. 高质量数据 ：基于权威文档生成的专业问答数据
3. 高效训练 ：使用QLoRA技术，显存占用低
4. 易于使用 ：提供完整的训练和推理脚本
5. 灵活部署 ：支持多种推理模式和参数配置
## 注意事项
1. 确保有足够的GPU显存（建议8GB以上）
2. 数据生成需要智谱AI API密钥
3. 训练时间根据硬件配置而定（通常几小时到十几小时）
4. 推理时可根据需要调整量化设置以平衡速度和质量
本项目为煤矿安全领域提供了一个完整的AI问答解决方案，从数据构造到模型训练再到实际应用，形成了完整的技术链路。

## 致谢
感谢智谱AI和 Hugging Face 等开源社区的贡献，为项目提供了强大的技术支持。

感谢https://github.com/yaosenJ/CoalQA的数据源
