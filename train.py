#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-R1-Distill-Qwen-1.5B 煤矿安全问答微调脚本
使用QLoRA技术进行高效微调
"""

import os
import json
import logging
import torch
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: str = "./DeepSeek-R1-Distill-Qwen-1.5B"
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    trust_remote_code: bool = True


@dataclass
class DataArguments:
    """数据相关参数"""
    data_path: str = "./QAdata_all.json"
    max_length: int = 512
    train_split: float = 0.9
    ignore_data_skip: bool = False


@dataclass
class TrainingArguments_Custom(TrainingArguments):
    """训练相关参数"""
    output_dir: str = field(default="./output")
    num_train_epochs: int = 6
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    save_total_limit: int = 3
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    bf16: bool = True
    dataloader_num_workers: int = 0  # 改为0，禁用多进程
    remove_unused_columns: bool = False
    report_to: str = "none"
    run_name: str = field(default_factory=lambda: f"deepseek-coal-safety-{datetime.now().strftime('%Y%m%d_%H%M%S')}")


class CoalSafetyDataset:
    """煤矿安全问答数据集处理类"""

    def __init__(self, data_args: DataArguments, tokenizer):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.data = self.load_data()

    def load_data(self):
        """加载训练数据"""
        try:
            # 首先尝试加载主数据文件
            if os.path.exists(self.data_args.data_path):
                logger.info(f"加载主数据文件: {self.data_args.data_path}")
                with open(self.data_args.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                raise FileNotFoundError(
                    f"找不到数据文件: {self.data_args.data_path}")

            logger.info(f"成功加载 {len(data)} 条数据")
            return data

        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise

    def format_conversation(self, input_text: str, output_text: str) -> str:
        """格式化对话数据为DeepSeek格式"""
        return f"system: 你是一个煤矿安全领域的知识达人，你对相关煤矿安全规章规程制度、技术等文档非常熟悉。请你专业正确地解答用户想问的煤矿安全相关问题。\nuser: {input_text}\nresponse: {output_text}"

    def tokenize_function(self, examples):
        """数据tokenization"""
        conversations = []
        for input_text, output_text in zip(examples['input'], examples['output']):
            conversation = self.format_conversation(input_text, output_text)
            conversations.append(conversation)

        # Tokenize
        model_inputs = self.tokenizer(
            conversations,
            max_length=self.data_args.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        # 设置labels（用于计算loss）
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        return model_inputs

    def create_datasets(self):
        """创建训练和验证数据集"""
        # 转换为Dataset格式
        dataset = Dataset.from_list(self.data)

        # 分割训练和验证集
        if self.data_args.train_split < 1.0:
            split_dataset = dataset.train_test_split(
                train_size=self.data_args.train_split,
                seed=42
            )
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
        else:
            train_dataset = dataset
            eval_dataset = None

        # Tokenize数据
        logger.info("开始tokenize训练数据...")
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train dataset"
        )

        if eval_dataset is not None:
            logger.info("开始tokenize验证数据...")
            eval_dataset = eval_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc="Tokenizing eval dataset"
            )

        return train_dataset, eval_dataset


def setup_model_and_tokenizer(model_args: ModelArguments):
    """设置模型和tokenizer"""
    logger.info(f"加载模型: {model_args.model_name_or_path}")

    # 配置量化
    if model_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=model_args.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, model_args.bnb_4bit_compute_dtype)
        )
    else:
        bnb_config = None

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right"
    )

    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16 if model_args.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
    )

    # 准备模型用于k-bit训练
    if model_args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def setup_lora(model, model_args: ModelArguments):
    """设置LoRA配置"""
    logger.info("配置LoRA...")

    # LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        inference_mode=False
    )

    # 应用LoRA
    model = get_peft_model(model, lora_config)

    # 打印可训练参数
    model.print_trainable_parameters()

    return model


def save_training_config(output_dir: str, model_args: ModelArguments,
                         data_args: DataArguments, training_args: TrainingArguments_Custom):
    """保存训练配置"""
    import json
    from datetime import datetime

    # 过滤掉不可序列化的对象
    def is_json_serializable(obj):
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False

    # 安全地转换training_args
    training_args_dict = {}
    for k, v in training_args.__dict__.items():
        if not k.startswith('_') and k not in ['device', 'accelerator_config', 'fsdp_config', 'deepspeed_config']:
            if is_json_serializable(v):
                training_args_dict[k] = v
            else:
                # 对于不可序列化的对象，尝试转换为字符串
                training_args_dict[k] = str(v)

    config = {
        "model_args": model_args.__dict__,
        "data_args": data_args.__dict__,
        "training_args": training_args_dict,
        "timestamp": datetime.now().isoformat()
    }

    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"训练配置已保存到: {config_path}")


def main():
    """主训练函数"""
    logger.info("开始DeepSeek-R1-Distill-Qwen-1.5B煤矿安全问答微调")

    # 初始化参数
    model_args = ModelArguments()
    data_args = DataArguments()

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"./output/deepseek-coal-safety-{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments_Custom(
        output_dir=output_dir,
        run_name=f"deepseek-coal-safety-{timestamp}"
    )

    # 保存训练配置
    save_training_config(output_dir, model_args, data_args, training_args)

    try:
        # 设置模型和tokenizer
        model, tokenizer = setup_model_and_tokenizer(model_args)

        # 设置LoRA
        model = setup_lora(model, model_args)

        # 准备数据
        dataset_processor = CoalSafetyDataset(data_args, tokenizer)
        train_dataset, eval_dataset = dataset_processor.create_datasets()

        logger.info(f"训练集大小: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"验证集大小: {len(eval_dataset)}")

        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            return_tensors="pt"
        )

        # 创建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )

        # 开始训练
        logger.info("开始训练...")
        trainer.train()

        # 保存最终模型
        logger.info("保存最终模型...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        # 保存训练状态
        trainer.save_state()

        logger.info(f"训练完成！模型已保存到: {output_dir}")

        # 显示训练总结
        if trainer.state.log_history:
            final_loss = trainer.state.log_history[-1].get('train_loss', 'N/A')
            logger.info(f"最终训练损失: {final_loss}")

            if eval_dataset:
                final_eval_loss = trainer.state.log_history[-1].get('eval_loss', 'N/A')
                logger.info(f"最终验证损失: {final_eval_loss}")

    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise

    logger.info("微调任务完成！")
    logger.info(f"\n使用以下命令进行推理测试:")
    logger.info(f"python inference.py --peft_model {output_dir}")


if __name__ == "__main__":
    main()