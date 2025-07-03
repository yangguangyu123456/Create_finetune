#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
煤矿安全问答模型API调用示例
演示如何通过HTTP请求调用部署的模型服务
"""

import requests
import json
import time
from typing import List, Dict, Any


class CoalSafetyAPIClient:
    """煤矿安全问答API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def check_status(self) -> Dict[str, Any]:
        """检查服务状态"""
        try:
            response = self.session.get(f"{self.base_url}/api/status")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def ask_question(self, question: str, max_length: int = 512, 
                    temperature: float = 0.7, top_p: float = 0.9) -> Dict[str, Any]:
        """提问"""
        try:
            data = {
                "question": question,
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p
            }
            response = self.session.post(
                f"{self.base_url}/api/ask",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def batch_inference(self, questions: List[str]) -> Dict[str, Any]:
        """批量推理"""
        try:
            data = {"questions": questions}
            response = self.session.post(
                f"{self.base_url}/api/batch",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def wait_for_ready(self, timeout: int = 300, check_interval: int = 5) -> bool:
        """等待服务就绪"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.check_status()
            if status.get('status') == 'ready':
                return True
            print(f"服务状态: {status.get('message', '未知')}")
            time.sleep(check_interval)
        return False


def main():
    """示例用法"""
    # 创建客户端
    client = CoalSafetyAPIClient()
    
    print("=== 煤矿安全问答API调用示例 ===")
    
    # 1. 健康检查
    print("\n1. 健康检查:")
    health = client.health_check()
    print(json.dumps(health, indent=2, ensure_ascii=False))
    
    # 2. 检查服务状态
    print("\n2. 检查服务状态:")
    status = client.check_status()
    print(json.dumps(status, indent=2, ensure_ascii=False))
    
    # 3. 等待服务就绪
    if status.get('status') != 'ready':
        print("\n3. 等待服务就绪...")
        if not client.wait_for_ready():
            print("服务启动超时，请检查服务状态")
            return
        print("服务已就绪！")
    
    # 4. 单个问题示例
    print("\n4. 单个问题示例:")
    questions = [
        "煤矿井下作业的安全要求有哪些？",
        "瓦斯爆炸的条件是什么？",
        "煤矿防治水的主要措施有哪些？",
        "井下电气设备的安全要求是什么？"
    ]
    
    for i, question in enumerate(questions[:2], 1):  # 只演示前2个问题
        print(f"\n问题 {i}: {question}")
        result = client.ask_question(question)
        if result.get('success'):
            print(f"回答: {result['answer']}")
        else:
            print(f"错误: {result.get('error')}")
        print("-" * 50)
    
    # 5. 批量推理示例
    print("\n5. 批量推理示例:")
    batch_questions = [
        "煤矿安全监控系统的作用是什么？",
        "井下通风系统的重要性？"
    ]
    
    batch_result = client.batch_inference(batch_questions)
    if batch_result.get('success'):
        print(f"批量处理完成，共处理 {batch_result['total']} 个问题:")
        for i, result in enumerate(batch_result['results'], 1):
            print(f"\n问题 {i}: {result['question']}")
            print(f"回答: {result['answer']}")
    else:
        print(f"批量推理失败: {batch_result.get('error')}")
    
    print("\n=== 示例完成 ===")


if __name__ == "__main__":
    main()