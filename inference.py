#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-R1-Distill-Qwen-1.5B 煤矿安全问答推理脚本
支持加载微调后的LoRA模型进行推理
"""

import os
import json
import argparse

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoalSafetyInference:
    """煤矿安全问答推理类"""

    def __init__(self, base_model_path: str, peft_model_path: str = None, use_4bit: bool = True):
        self.base_model_path = base_model_path
        self.peft_model_path = peft_model_path
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None

        self.load_model()

    def load_model(self):
        """加载模型和tokenizer"""
        logger.info(f"加载基础模型: {self.base_model_path}")

        # 配置量化（如果需要）
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            padding_side="right"
        )

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not self.use_4bit else None
        )

        # 如果有LoRA模型，加载它
        if self.peft_model_path and os.path.exists(self.peft_model_path):
            logger.info(f"加载LoRA模型: {self.peft_model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.peft_model_path)
            logger.info("LoRA模型加载完成")
        else:
            logger.info("使用原始基础模型（未加载LoRA）")

        # 设置为评估模式
        self.model.eval()
        logger.info("模型加载完成")

    def format_prompt(self, question: str) -> str:
        """格式化输入提示"""
        return f"system: 你是一个煤矿安全领域的知识达人，你对相关煤矿安全规章规程制度、技术等文档非常熟悉。请你专业正确地解答用户想问的煤矿安全相关问题。\nuser: {question}\nresponse: "

    def generate_response(self, question: str, max_length: int = 512,
                          temperature: float = 0.7, top_p: float = 0.9,
                          do_sample: bool = True) -> str:
        """生成回答"""
        # 格式化输入
        prompt = self.format_prompt(question)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )

        # 移动到模型设备
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取助手回答部分
        if "response:" in response:
            response = response.split("response:")[-1]

        return response.strip()

    def interactive_chat(self):
        """交互式对话"""
        print("\n=== 煤矿安全问答系统 ===")
        print("输入您的问题，输入 'quit' 或 'exit' 退出")
        print("输入 'clear' 清屏")
        print("=" * 50)

        while True:
            try:
                question = input("\n🤔 您的问题: ").strip()

                if question.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break

                if question.lower() in ['clear', '清屏']:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue

                if not question:
                    continue

                print("\n🤖 正在思考...")
                response = self.generate_response(question)
                print(f"\n💡 回答: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 生成回答时出错: {e}")

    def batch_inference(self, questions: list) -> list:
        """批量推理"""
        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"处理问题 {i}/{len(questions)}: {question[:50]}...")
            try:
                response = self.generate_response(question)
                results.append({
                    "question": question,
                    "answer": response
                })
            except Exception as e:
                logger.error(f"处理问题时出错: {e}")
                results.append({
                    "question": question,
                    "answer": f"错误: {str(e)}"
                })
        return results


def main():
    parser = argparse.ArgumentParser(description="DeepSeek煤矿安全问答推理")
    parser.add_argument(
        "--base_model",
        type=str,
        default="./DeepSeek-R1-Distill-Qwen-1.5B",
        help="基础模型路径"
    )
    parser.add_argument(
        "--peft_model",
        type=str,
        default=None,
        help="LoRA模型路径"
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=True,
        help="是否使用4bit量化"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="单个问题（非交互模式）"
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default=None,
        help="问题文件路径（JSON格式）"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="输出文件路径"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="最大生成长度"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成温度"
    )

    args = parser.parse_args()

    # 初始化推理器
    inferencer = CoalSafetyInference(
        base_model_path=args.base_model,
        peft_model_path=args.peft_model,
        use_4bit=args.use_4bit
    )

    # 单个问题模式
    if args.question:
        print(f"问题: {args.question}")
        response = inferencer.generate_response(
            args.question,
            max_length=args.max_length,
            temperature=args.temperature
        )
        print(f"回答: {response}")

    # 批量问题模式
    elif args.questions_file:
        if not os.path.exists(args.questions_file):
            print(f"错误: 问题文件不存在: {args.questions_file}")
            return

        with open(args.questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)

        # 支持不同的JSON格式
        if isinstance(questions_data, list):
            if isinstance(questions_data[0], str):
                questions = questions_data
            else:
                questions = [item.get('question', item.get('input', '')) for item in questions_data]
        else:
            questions = questions_data.get('questions', [])

        results = inferencer.batch_inference(questions)

        # 保存结果
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"结果已保存到: {args.output_file}")
        else:
            for result in results:
                print(f"\n问题: {result['question']}")
                print(f"回答: {result['answer']}")
                print("-" * 50)

    # 交互模式
    else:
        inferencer.interactive_chat()


if __name__ == "__main__":
    main()